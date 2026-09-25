"""Atomic updates to a synthetic, price-aggregated L2 book.

Sequence numbers belong to this normalized single-session protocol, not an
exchange feed. Updates contain absolute sizes; they do not identify trades,
cancellations, queue position or our own simulated executions. Single-threaded
use only: atomic means no partially applied batch is published by a method.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal

from .costs import FillEstimate, Level, estimate_fill

ZERO = Decimal("0")


class BookError(ValueError):
    """A message cannot be accepted; the book needs a fresh snapshot."""


class BookNotReadyError(BookError):
    """No current valid state is available for updates or cost estimates."""


@dataclass(frozen=True, slots=True)
class LevelUpdate:
    side: str
    price: Decimal
    quantity: Decimal


@dataclass(frozen=True, slots=True)
class BookState:
    bids: tuple[Level, ...] = ()
    asks: tuple[Level, ...] = ()
    sequence: int | None = None
    version: int = 0
    is_valid: bool = False
    invalid_reason: str | None = "snapshot required"
    required_snapshot_sequence: int = 0


def _sequence(value: int) -> None:
    if type(value) is not int or value < 0:
        raise BookError("sequence must be a non-negative integer (not bool)")


def _number(value: Decimal, name: str, *, allow_zero: bool = False) -> None:
    if not isinstance(value, Decimal):
        raise BookError(f"{name} must be Decimal")
    if not value.is_finite() or value < ZERO or (value == ZERO and not allow_zero):
        raise BookError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")


def _sequence_container(value: object, name: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BookError(f"{name} must be a sequence")


def _snapshot_side(levels: Sequence[Level], name: str) -> dict[Decimal, Decimal]:
    _sequence_container(levels, name)
    result = {}
    for level in levels:
        _sequence_container(level, f"{name} level")
        if len(level) != 2:
            raise BookError(f"{name} level must contain price and quantity")
        price, quantity = level
        _number(price, "price")
        _number(quantity, "snapshot quantity")
        if price in result:
            raise BookError(f"duplicate {name} snapshot price")
        result[price] = quantity
    return result


def _sorted_sides(bids: dict, asks: dict) -> tuple[tuple[Level, ...], tuple[Level, ...]]:
    bid_levels = tuple(sorted(bids.items(), reverse=True))
    ask_levels = tuple(sorted(asks.items()))
    if bid_levels and ask_levels and bid_levels[0][0] >= ask_levels[0][0]:
        raise BookError("final book must have best bid strictly below best ask")
    return bid_levels, ask_levels


class OrderBook:
    """Maintain one normalized session; recovery requires a full snapshot.

    Rejected messages never change last accepted depth, sequence or version.
    They *do* invalidate usability and may raise the recovery sequence floor.
    A new source session with restarted sequence numbering needs a new instance.
    """

    def __init__(self) -> None:
        self._state = BookState()

    @property
    def state(self) -> BookState:
        """Immutable current view; old views remain historical snapshots."""
        return self._state

    def _reject(self, sequence: object, reason: str) -> None:
        floor = self._state.required_snapshot_sequence
        if type(sequence) is int and sequence >= 0:
            floor = max(floor, sequence)
        self._state = replace(
            self._state, is_valid=False, invalid_reason=reason,
            required_snapshot_sequence=floor,
        )

    def _commit(self, sequence: int, bids: dict, asks: dict) -> BookState:
        bid_levels, ask_levels = _sorted_sides(bids, asks)
        # The only publication point: every field is replaced together.
        self._state = BookState(
            bids=bid_levels, asks=ask_levels, sequence=sequence,
            version=self._state.version + 1, is_valid=True,
            invalid_reason=None, required_snapshot_sequence=sequence + 1,
        )
        return self._state

    def apply_snapshot(
        self, *, sequence: int, bids: Sequence[Level], asks: Sequence[Level]
    ) -> BookState:
        """Replace both sides; unordered unique positive levels are sorted.

        Sequence must reach the recovery floor and exceed any accepted sequence.
        Empty sides are valid and differ from a missing/invalid snapshot.
        """
        try:
            _sequence(sequence)
            if sequence < self._state.required_snapshot_sequence:
                raise BookError(
                    f"snapshot sequence must be at least {self._state.required_snapshot_sequence}"
                )
            candidate_bids = _snapshot_side(bids, "bids")
            candidate_asks = _snapshot_side(asks, "asks")
            return self._commit(sequence, candidate_bids, candidate_asks)
        except BookError as exc:
            self._reject(sequence, str(exc))
            raise

    def apply_update(self, *, sequence: int, updates: Sequence[LevelUpdate]) -> BookState:
        """Apply unique (side, price) changes as one batch of absolute sizes.

        Zero removes a level (absent levels are harmless no-ops). Intermediate
        candidate states may be crossed; only the final two-sided state matters.
        Even an empty batch consumes the next normalized sequence number.
        """
        try:
            _sequence(sequence)
            if not self._state.is_valid:
                raise BookNotReadyError("fresh snapshot required before incremental updates")
            expected = self._state.sequence + 1
            if sequence != expected:
                raise BookError(f"update sequence must be {expected}, received {sequence}")
            _sequence_container(updates, "updates")
            bids, asks = dict(self._state.bids), dict(self._state.asks)
            seen = set()
            for update in updates:
                if not isinstance(update, LevelUpdate):
                    raise BookError("updates must contain LevelUpdate values")
                if not isinstance(update.side, str) or update.side not in ("BID", "ASK"):
                    raise BookError("update side must be BID or ASK")
                _number(update.price, "price")
                _number(update.quantity, "update quantity", allow_zero=True)
                key = (update.side, update.price)
                if key in seen:
                    raise BookError("duplicate side/price within update batch")
                seen.add(key)
                levels = bids if update.side == "BID" else asks
                if update.quantity == ZERO:
                    levels.pop(update.price, None)
                else:
                    levels[update.price] = update.quantity
            return self._commit(sequence, bids, asks)
        except BookError as exc:
            self._reject(sequence, str(exc))
            raise

    def estimate_fill(self, side: str, quantity: Decimal) -> FillEstimate:
        """Query only a valid current view; bad order inputs do not invalidate it."""
        state = self._state
        if not state.is_valid:
            raise BookNotReadyError(state.invalid_reason or "snapshot required")
        return estimate_fill(side, quantity, state.bids, state.asks)
