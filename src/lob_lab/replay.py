"""Deterministic, single-session replay of visible messages and cost queries.

This is an in-memory synthetic query engine. It has no order submission,
execution consumption, strategy callback, exchange clock or market feedback.
Times are integer nanoseconds since an explicitly chosen synthetic origin.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from .book import BookError, BookState, LevelUpdate, OrderBook
from .costs import Level
from .valuation import CostEstimate, estimate_cost


@dataclass(frozen=True, slots=True)
class SnapshotEvent:
    visible_time_ns: int
    event_index: int
    sequence: int
    bids: Sequence[Level]
    asks: Sequence[Level]
    source_time_ns: int | None = None


@dataclass(frozen=True, slots=True)
class UpdateEvent:
    visible_time_ns: int
    event_index: int
    sequence: int
    updates: Sequence[LevelUpdate]
    source_time_ns: int | None = None


@dataclass(frozen=True, slots=True)
class CostQuery:
    query_id: str
    visible_time_ns: int
    side: str
    quantity: Decimal
    reference_price: Decimal
    fee_bps: Decimal = Decimal("0")


@dataclass(frozen=True, slots=True)
class MessageResult:
    visible_time_ns: int
    event_index: int
    source_time_ns: int | None
    accepted: bool
    error: str | None
    state: BookState


@dataclass(frozen=True, slots=True)
class QueryResult:
    query_id: str
    visible_time_ns: int
    state: BookState
    last_accepted_visible_time_ns: int | None
    last_accepted_source_time_ns: int | None
    book_age_ns: int | None
    estimate: CostEstimate | None
    error: str | None


@dataclass(frozen=True, slots=True)
class ReplayResult:
    messages: tuple[MessageResult, ...]
    queries: tuple[QueryResult, ...]


def _integer(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer (not bool)")


def _container(value: object, name: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")


def run_replay(
    events: Sequence[SnapshotEvent | UpdateEvent], queries: Sequence[CostQuery],
) -> ReplayResult:
    """Evaluate each query using messages visible no later than its time.

    Event order is (visible time, recorded event_index), never source time or
    book sequence. Every message at a timestamp precedes all queries at that
    timestamp; tied queries retain input order. Indices and query IDs are unique.
    Malformed scheduling/query inputs fail the run. Book semantic errors become
    rejection records at arrival and invalidate queries until snapshot recovery.

    Book age is time since the last *accepted* message, not measured feed latency
    or a guarantee of freshness. There is no automatic age expiry in this layer.
    The batch result includes later records for analysis, not for earlier queries.
    """
    _container(events, "events")
    _container(queries, "queries")
    timeline = []
    indices = set()
    for event in events:
        if not isinstance(event, (SnapshotEvent, UpdateEvent)):
            raise TypeError("events must contain SnapshotEvent or UpdateEvent")
        _integer(event.visible_time_ns, "visible_time_ns")
        _integer(event.event_index, "event_index")
        if event.source_time_ns is not None:
            _integer(event.source_time_ns, "source_time_ns")
        if event.event_index in indices:
            raise ValueError("event_index must be unique across the session")
        indices.add(event.event_index)
        timeline.append((event.visible_time_ns, 0, event.event_index, event))
    query_ids = set()
    for index, query in enumerate(queries):
        if not isinstance(query, CostQuery):
            raise TypeError("queries must contain CostQuery")
        _integer(query.visible_time_ns, "query visible_time_ns")
        if not isinstance(query.query_id, str) or not query.query_id.strip():
            raise ValueError("query_id must be a nonempty string")
        if query.query_id in query_ids:
            raise ValueError("query_id must be unique")
        query_ids.add(query.query_id)
        # Validate even if there is no valid book at the query time.
        estimate_cost(query.side, query.quantity, (), (), reference_price=query.reference_price, fee_bps=query.fee_bps)
        timeline.append((query.visible_time_ns, 1, index, query))
    timeline.sort(key=lambda item: item[:3])

    book = OrderBook()
    messages, results = [], []
    last_visible = last_source = None
    for time, priority, _, item in timeline:
        if priority == 0:
            error = None
            try:
                if isinstance(item, SnapshotEvent):
                    book.apply_snapshot(sequence=item.sequence, bids=item.bids, asks=item.asks)
                else:
                    book.apply_update(sequence=item.sequence, updates=item.updates)
            except BookError as exc:
                error = str(exc)
            if error is None:
                last_visible, last_source = time, item.source_time_ns
            messages.append(MessageResult(time, item.event_index, item.source_time_ns, error is None, error, book.state))
        else:
            state = book.state
            estimate = None
            error = state.invalid_reason if not state.is_valid else None
            if state.is_valid:
                estimate = estimate_cost(
                    item.side, item.quantity, state.bids, state.asks,
                    reference_price=item.reference_price, fee_bps=item.fee_bps,
                )
            results.append(QueryResult(
                item.query_id, time, state, last_visible, last_source,
                time - last_visible if last_visible is not None else None,
                estimate, error,
            ))
    return ReplayResult(tuple(messages), tuple(results))
