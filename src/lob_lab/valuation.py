"""Fee-inclusive costs of the covered quantity against an explicit reference.

These remain read-only visible-book estimates, not fills or full parent-order
implementation shortfall. Uncovered quantity has no invented execution price.
"""

from dataclasses import dataclass
from decimal import Decimal, DecimalException, Inexact, localcontext

from .costs import (
    MAX_ACCOUNTING_PRECISION, RATIO_PRECISION, FillEstimate, Level,
    _context, _number, estimate_fill,
)
from collections.abc import Sequence

ZERO = Decimal("0")
BPS = Decimal("10000")


@dataclass(frozen=True, slots=True)
class CostEstimate:
    fill: FillEstimate
    reference_price: Decimal
    fee_bps: Decimal
    fee_amount: Decimal
    covered_cost: Decimal
    covered_cost_bps: Decimal | None
    full_cost_bps: Decimal | None


def estimate_cost(
    side: str, quantity: Decimal, bids: Sequence[Level], asks: Sequence[Level],
    *, reference_price: Decimal, fee_bps: Decimal = ZERO,
) -> CostEstimate:
    """Add a nonnegative ad-valorem fee, once, to covered signed price cost.

    Positive cost is worse than the reference for either BUY or SELL. Money is
    exact; bps ratios use 28 significant digits, independent of caller context.
    ``full_cost_bps`` is undefined for partial/empty/zero-size queries. The
    reference must be specified before evaluation; no future mid is selected.
    Fees have no currency/tick rounding here; a real adapter must specify that.
    """
    _number(reference_price, "reference_price")
    _number(fee_bps, "fee_bps", allow_zero=True)
    fill = estimate_fill(side, quantity, bids, asks)
    numbers = [reference_price, fee_bps, fill.filled_qty, fill.notional]
    nonzero = [v for v in numbers if v != ZERO]
    width = max(v.adjusted() for v in nonzero) - min(v.as_tuple().exponent for v in nonzero) + 1
    # Products, summation/cancellation and four fee decimal places fit exactly.
    precision = max(RATIO_PRECISION, 3 * width + 16)
    if precision > MAX_ACCOUNTING_PRECISION:
        raise ValueError("decimal scale spread exceeds supported cost accounting precision")
    accounting = _context(precision)
    accounting.traps[Inexact] = True
    try:
        with localcontext(accounting):
            fee = fill.notional * fee_bps / BPS
            reference_notional = fill.filled_qty * reference_price
            signed_price_cost = (fill.notional - reference_notional) * (1 if side == "BUY" else -1)
            covered_cost = signed_price_cost + fee
            numerator = covered_cost * BPS
        ratio = _context(RATIO_PRECISION)
        bps = ratio.divide(numerator, reference_notional) if fill.filled_qty else None
    except DecimalException as exc:
        raise ValueError("arithmetic exceeds supported cost decimal range or exact precision") from exc
    return CostEstimate(
        fill=fill, reference_price=reference_price, fee_bps=fee_bps,
        fee_amount=fee, covered_cost=covered_cost, covered_cost_bps=bps,
        full_cost_bps=bps if fill.depth_sufficient and quantity > ZERO else None,
    )
