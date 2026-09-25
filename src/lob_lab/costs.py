"""Read-only sweep estimates for a validated, static, visible L2 book.

No fees, latency, price protection, hidden liquidity or market feedback are
modelled here. A query does not consume depth or place an order.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import (
    MAX_EMAX,
    MIN_EMIN,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DecimalException,
    Inexact,
    localcontext,
)

Level = tuple[Decimal, Decimal]
ZERO = Decimal("0")
RATIO_PRECISION = 28
MAX_ACCOUNTING_PRECISION = 10_000


@dataclass(frozen=True, slots=True)
class FillEstimate:
    """Conditional visible fills; None prices mean that no quantity is filled.

    Quantities and notional are exact. Average and ratio are rounded to 28
    significant decimal digits using ROUND_HALF_EVEN, independently of the
    caller's decimal context. ``notional`` excludes fees.
    """

    requested_qty: Decimal
    filled_qty: Decimal
    unfilled_qty: Decimal
    average_price: Decimal | None
    worst_price: Decimal | None
    levels_swept: int
    fill_ratio: Decimal | None
    depth_sufficient: bool
    notional: Decimal


def _number(value: Decimal, name: str, *, allow_zero: bool = False) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be Decimal constructed from a string")
    if not value.is_finite() or value < ZERO or (not allow_zero and value == ZERO):
        limit = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {limit}")


def _levels(levels: Sequence[Level], name: str, *, ascending: bool) -> tuple[Level, ...]:
    if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of (price, quantity) pairs")
    result = []
    previous = None
    for index, level in enumerate(levels):
        if (
            not isinstance(level, Sequence)
            or isinstance(level, (str, bytes))
            or len(level) != 2
        ):
            raise TypeError(f"{name}[{index}] must be a (price, quantity) pair")
        price, size = level
        _number(price, f"{name}[{index}].price")
        _number(size, f"{name}[{index}].quantity")
        if previous is not None and (
            (ascending and price <= previous) or (not ascending and price >= previous)
        ):
            order = "increasing" if ascending else "decreasing"
            raise ValueError(f"{name} prices must be strictly {order}, without duplicates")
        result.append((price, size))
        previous = price
    return tuple(result)


def _context(precision: int) -> Context:
    # A fresh Context also isolates the operation from caller traps/rounding.
    return Context(prec=precision, rounding=ROUND_HALF_EVEN, Emin=MIN_EMIN, Emax=MAX_EMAX)


def estimate_fill(
    side: str,
    quantity: Decimal,
    bids: Sequence[Level],
    asks: Sequence[Level],
) -> FillEstimate:
    """Estimate a BUY against asks or a SELL against bids without mutation.

    Validate the entire two-sided input even for zero orders and unused levels.
    Empty sides are allowed; a locked/crossed two-sided book is rejected.
    Invalid numeric types raise TypeError; invalid values/order raise ValueError.
    A scale spread requiring >10,000 accounting digits or arithmetic outside
    Decimal's exponent range is rejected instead of returning inexact money.
    """
    if not isinstance(side, str) or side not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    _number(quantity, "quantity", allow_zero=True)
    bid_levels = _levels(bids, "bids", ascending=False)
    ask_levels = _levels(asks, "asks", ascending=True)
    if bid_levels and ask_levels and bid_levels[0][0] >= ask_levels[0][0]:
        raise ValueError("book must have best bid strictly below best ask")

    numbers = [quantity] + [v for level in bid_levels + ask_levels for v in level]
    nonzero = [v for v in numbers if v != ZERO]
    if nonzero:
        width = max(v.adjusted() for v in nonzero) - min(
            v.as_tuple().exponent for v in nonzero
        ) + 1
    else:
        width = 1
    # Two widths cover price * size; extra digits cover summation carries.
    precision = max(RATIO_PRECISION, 2 * width + len(str(len(numbers))) + 2)
    if precision > MAX_ACCOUNTING_PRECISION:
        raise ValueError("decimal scale spread exceeds supported accounting precision")
    accounting = _context(precision)
    accounting.traps[Inexact] = True

    try:
        with localcontext(accounting):
            remaining = quantity
            filled = notional = ZERO
            worst = None
            count = 0
            for price, available in ask_levels if side == "BUY" else bid_levels:
                if remaining == ZERO:
                    break
                take = min(available, remaining)
                notional += price * take
                filled += take
                remaining -= take
                worst = price
                count += 1

        division = _context(RATIO_PRECISION)
        return FillEstimate(
            requested_qty=quantity,
            filled_qty=filled,
            unfilled_qty=remaining,
            average_price=division.divide(notional, filled) if filled else None,
            worst_price=worst,
            levels_swept=count,
            fill_ratio=division.divide(filled, quantity) if quantity else None,
            depth_sufficient=remaining == ZERO,
            notional=notional,
        )
    except DecimalException as exc:
        raise ValueError("arithmetic exceeds supported decimal range or exact precision") from exc
