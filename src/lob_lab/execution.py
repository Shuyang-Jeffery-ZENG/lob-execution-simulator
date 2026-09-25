"""Single-parent, static synthetic execution with shared residual liquidity.

Calls are explicit serial operations, not timed exchange events. Submission
reserves parent quantity, never depth. An execution is immediate-or-cancel at
call time; unfilled child quantity is released, not left resting. No market
updates, replenishment, latency, real fills or total parent-order IS are modelled.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, DecimalException, Inexact, localcontext

from .costs import Level, MAX_ACCOUNTING_PRECISION, RATIO_PRECISION, _context, _number
from .valuation import CostEstimate, estimate_cost

ZERO = Decimal("0")
BPS = Decimal("10000")


@dataclass(frozen=True, slots=True)
class ParentOrder:
    parent_id: str
    side: str
    quantity: Decimal
    reference_price: Decimal
    fee_bps: Decimal = ZERO


@dataclass(frozen=True, slots=True)
class ChildOrder:
    child_id: str
    requested_qty: Decimal
    limit_price: Decimal | None
    status: str = "reserved"
    filled_qty: Decimal = ZERO
    unfilled_qty: Decimal | None = None
    fills: tuple[Level, ...] = ()
    notional: Decimal = ZERO
    fee_amount: Decimal = ZERO
    covered_cost: Decimal = ZERO


@dataclass(frozen=True, slots=True)
class ExecutionState:
    parent: ParentOrder
    bids: tuple[Level, ...]
    asks: tuple[Level, ...]
    children: tuple[ChildOrder, ...]
    filled_qty: Decimal
    remaining_qty: Decimal
    reserved_qty: Decimal
    available_qty: Decimal
    notional: Decimal
    fees: Decimal
    covered_cost: Decimal
    average_price: Decimal | None
    full_cost_bps: Decimal | None
    version: int


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _accounting(numbers):
    nonzero = [v for v in numbers if v != ZERO]
    width = max(v.adjusted() for v in nonzero) - min(v.as_tuple().exponent for v in nonzero) + 1 if nonzero else 1
    precision = max(RATIO_PRECISION, 4 * width + len(str(len(numbers))) + 32)
    if precision > MAX_ACCOUNTING_PRECISION:
        raise ValueError("decimal scale spread exceeds supported execution precision")
    context = _context(precision)
    context.traps[Inexact] = True
    return context


def _limit(value):
    if value is not None:
        _number(value, "limit_price")


def _eligible_sides(side, bids, asks, limit):
    if limit is None:
        return bids, asks
    if side == "BUY":
        return bids, tuple(level for level in asks if level[0] <= limit)
    return tuple(level for level in bids if level[0] >= limit), asks


def _consume(levels, quantity):
    """Return exact residual depth and per-level fills, without mutation."""
    numbers = [quantity] + [v for level in levels for v in level]
    with localcontext(_accounting(numbers)):
        remaining = quantity
        residual, fills = [], []
        for price, available in levels:
            take = min(available, remaining)
            if take:
                fills.append((price, take))
            left = available - take
            if left:
                residual.append((price, left))
            remaining -= take
    return tuple(residual), tuple(fills)


class StaticExecution:
    """Own one valid immutable initial book and one parent in a serial sandbox.

    The input sides must already be sorted, positive, unique and noncrossed.
    There is deliberately no update/reset method: historical replenishment
    requires a separately specified model. State is published only after all
    validation, consumption and accounting succeed. Not thread-safe.
    """

    def __init__(self, parent: ParentOrder, bids: Sequence[Level], asks: Sequence[Level]):
        if not isinstance(parent, ParentOrder):
            raise TypeError("parent must be ParentOrder")
        _identifier(parent.parent_id, "parent_id")
        _number(parent.quantity, "parent quantity", allow_zero=True)
        estimate_cost(parent.side, ZERO, bids, asks, reference_price=parent.reference_price, fee_bps=parent.fee_bps)
        self._state = self._build(parent, tuple(tuple(v) for v in bids), tuple(tuple(v) for v in asks), (), 0)

    @property
    def state(self) -> ExecutionState:
        return self._state

    @staticmethod
    def _build(parent, bids, asks, children, version):
        numbers = [parent.quantity, parent.reference_price, parent.fee_bps]
        numbers += [v for level in bids + asks for v in level]
        for child in children:
            numbers += [child.requested_qty, child.filled_qty, child.notional, child.fee_amount, child.covered_cost]
            if child.limit_price is not None:
                numbers.append(child.limit_price)
        try:
            with localcontext(_accounting(numbers)):
                filled = sum((c.filled_qty for c in children), ZERO)
                reserved = sum((c.requested_qty for c in children if c.status == "reserved"), ZERO)
                remaining = parent.quantity - filled
                available = remaining - reserved
                notional = sum((c.notional for c in children), ZERO)
                fees = sum((c.fee_amount for c in children), ZERO)
                cost = sum((c.covered_cost for c in children), ZERO)
                denominator = parent.quantity * parent.reference_price
                numerator = cost * BPS
                if min(filled, reserved, remaining, available) < ZERO:
                    raise ValueError("parent quantity conservation failed")
            ratio = _context(RATIO_PRECISION)
            average = ratio.divide(notional, filled) if filled else None
            bps = ratio.divide(numerator, denominator) if parent.quantity > ZERO and remaining == ZERO else None
        except DecimalException as exc:
            raise ValueError("execution accounting exceeds supported decimal range or exact precision") from exc
        return ExecutionState(parent, bids, asks, children, filled, remaining, reserved, available, notional, fees, cost, average, bps, version)

    def _child_index(self, child_id):
        _identifier(child_id, "child_id")
        for index, child in enumerate(self.state.children):
            if child.child_id == child_id:
                if child.status != "reserved":
                    raise ValueError("child is already terminal")
                return index
        raise ValueError("unknown child_id")

    def quote(self, quantity: Decimal, limit_price: Decimal | None = None) -> CostEstimate:
        """Read-only diagnostic; quantity need not equal the parent's remainder."""
        _limit(limit_price)
        state = self.state
        bids, asks = _eligible_sides(state.parent.side, state.bids, state.asks, limit_price)
        return estimate_cost(state.parent.side, quantity, bids, asks, reference_price=state.parent.reference_price, fee_bps=state.parent.fee_bps)

    def submit(self, child_id: str, quantity: Decimal, limit_price: Decimal | None = None) -> ExecutionState:
        _identifier(child_id, "child_id")
        _number(quantity, "child quantity")
        _limit(limit_price)
        state = self.state
        if any(c.child_id == child_id for c in state.children):
            raise ValueError("child_id must be unique across the session")
        if quantity > state.available_qty:
            raise ValueError("child quantity exceeds unreserved parent remainder")
        child = ChildOrder(child_id, quantity, limit_price)
        candidate = self._build(state.parent, state.bids, state.asks, state.children + (child,), state.version + 1)
        self._state = candidate
        return candidate

    def cancel(self, child_id: str) -> ExecutionState:
        index = self._child_index(child_id)
        state = self.state
        children = list(state.children)
        children[index] = replace(children[index], status="cancelled", unfilled_qty=children[index].requested_qty)
        candidate = self._build(state.parent, state.bids, state.asks, tuple(children), state.version + 1)
        self._state = candidate
        return candidate

    def execute(self, child_id: str) -> ExecutionState:
        index = self._child_index(child_id)
        state = self.state
        child = state.children[index]
        estimate = self.quote(child.requested_qty, child.limit_price)
        try:
            # Consume exactly the eligible filled amount from the favorable
            # prefix of the original book. Protected levels remain untouched.
            opposite = state.asks if state.parent.side == "BUY" else state.bids
            residual, fills = _consume(opposite, estimate.fill.filled_qty)
            bids, asks = (state.bids, residual) if state.parent.side == "BUY" else (residual, state.asks)
            status = "filled" if estimate.fill.depth_sufficient else "partial" if estimate.fill.filled_qty else "unfilled"
            children = list(state.children)
            children[index] = replace(
                child, status=status, filled_qty=estimate.fill.filled_qty,
                unfilled_qty=estimate.fill.unfilled_qty, fills=fills,
                notional=estimate.fill.notional, fee_amount=estimate.fee_amount,
                covered_cost=estimate.covered_cost,
            )
            candidate = self._build(state.parent, bids, asks, tuple(children), state.version + 1)
        except DecimalException as exc:
            raise ValueError("execution arithmetic exceeds supported decimal range or exact precision") from exc
        self._state = candidate
        return candidate
