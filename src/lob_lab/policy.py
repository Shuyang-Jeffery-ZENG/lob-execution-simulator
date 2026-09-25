"""Current-information decisions; no event stream or future outcomes are inputs."""

from dataclasses import dataclass
from decimal import Decimal, localcontext

from .costs import Level, _number
from .execution import ParentOrder, _accounting, _eligible_sides
from .valuation import estimate_cost

ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class LiquidityRule:
    max_cost_bps: Decimal

    def __post_init__(self):
        _number(self.max_cost_bps, "max_cost_bps", allow_zero=True)


@dataclass(frozen=True, slots=True)
class PolicyView:
    time_ns: int
    target_qty: Decimal
    parent: ParentOrder
    filled_qty: Decimal
    reserved_qty: Decimal
    available_qty: Decimal
    raw_bids: tuple[Level, ...]
    raw_asks: tuple[Level, ...]
    bids: tuple[Level, ...]
    asks: tuple[Level, ...]
    limit_price: Decimal | None
    unusable_reason: str | None
    is_catch_up: bool


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    time_ns: int
    target_qty: Decimal
    candidate_qty: Decimal | None
    selected_qty: Decimal
    quoted_filled_qty: Decimal | None
    full_cost_bps: Decimal | None
    quote_mid: Decimal | None
    reason: str
    is_catch_up: bool


def decide(view: PolicyView, rule: LiquidityRule | None = None) -> PolicyDecision:
    """Decide from an engine-validated frozen view, using an optional soft gate.

    Gate costs use the current raw midpoint and effective protected depth. The
    gate is a decision estimate, never a fill guarantee or future price model.
    Baselines and common catch-up submit the scheduled deficit without gating.
    """
    def result(reason, quantity=ZERO, candidate=None, quote=None, mid=None):
        return PolicyDecision(
            view.time_ns, view.target_qty, candidate, quantity,
            quote.fill.filled_qty if quote else None,
            quote.full_cost_bps if quote else None, mid, reason, view.is_catch_up,
        )

    if view.unusable_reason is not None:
        return result(view.unusable_reason)
    with localcontext(_accounting([view.target_qty, view.filled_qty, view.reserved_qty, view.available_qty])):
        quantity = min(view.available_qty, max(ZERO, view.target_qty - view.filled_qty - view.reserved_qty))
    if quantity == ZERO:
        return result("target already filled or reserved", candidate=quantity)
    if rule is None or view.is_catch_up:
        return result("catch_up" if view.is_catch_up else "scheduled", quantity, quantity)
    if not view.raw_bids or not view.raw_asks:
        return result("missing_mid", candidate=quantity)
    with localcontext(_accounting([view.raw_bids[0][0], view.raw_asks[0][0]])):
        mid = (view.raw_bids[0][0] + view.raw_asks[0][0]) / Decimal("2")
    bids, asks = _eligible_sides(view.parent.side, view.bids, view.asks, view.limit_price)
    quote = estimate_cost(view.parent.side, quantity, bids, asks, reference_price=mid, fee_bps=view.parent.fee_bps)
    if not quote.fill.depth_sufficient:
        return result("insufficient_protected_depth", candidate=quantity, quote=quote, mid=mid)
    # Exact cash comparison; rounding the display bps must not change a decision.
    with localcontext(_accounting([quote.covered_cost, quantity, mid, rule.max_cost_bps])):
        permitted = quote.covered_cost * Decimal("10000") <= rule.max_cost_bps * quantity * mid
    if not permitted:
        return result("cost_above_cap", candidate=quantity, quote=quote, mid=mid)
    return result("cost_within_cap", quantity, quantity, quote, mid)
