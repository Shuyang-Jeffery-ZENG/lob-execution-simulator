"""Timed synthetic active execution under explicitly exogenous liquidity rules.

Visible-clock book observations stand in for arrival-time liquidity. They are
not an exchange clock or identified counterfactual market response. Execution
and confirmation share a timestamp; asynchronous acknowledgements are absent.
"""

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, replace
from decimal import Decimal, DecimalException, localcontext
import heapq

from .book import BookError, BookState, OrderBook
from .costs import RATIO_PRECISION, _context
from .execution import ParentOrder, ExecutionState, StaticExecution, _accounting, _limit
from .replay import SnapshotEvent, UpdateEvent, _container, _integer
from .policy import LiquidityRule, PolicyDecision, PolicyView, decide

ZERO = Decimal("0")
BPS = Decimal("10000")


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    start_time_ns: int
    deadline_ns: int
    observation_start_ns: int
    observation_end_ns: int
    order_latency_ns: int = 0
    max_book_age_ns: int | None = None
    liquidity_model: str = "persistent_debit"
    limit_price: Decimal | None = None


@dataclass(frozen=True, slots=True)
class StrategySpec:
    kind: str = "immediate"
    slices: int = 1
    catch_up: bool = True


@dataclass(frozen=True, slots=True)
class TraceRecord:
    time_ns: int
    event_type: str
    child_id: str | None
    reason: str | None
    market: BookState
    book_age_ns: int | None
    ledger: ExecutionState


@dataclass(frozen=True, slots=True)
class OrderTiming:
    child_id: str
    decision_time_ns: int
    arrival_time_ns: int
    processed_time_ns: int | None = None
    confirmation_time_ns: int | None = None
    status: str = "pending"
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionMetrics:
    target_qty: Decimal
    filled_qty: Decimal
    remaining_qty: Decimal
    completion_ratio: Decimal | None
    covered_cost: Decimal
    fees: Decimal
    terminal_mid: Decimal | None
    opportunity_cost: Decimal | None
    is_cash: Decimal | None
    is_bps: Decimal | None
    evaluation_available: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class TimedResult:
    parent: ParentOrder
    config: ExecutionConfig
    strategy: StrategySpec
    policy_label: str
    trace: tuple[TraceRecord, ...]
    orders: tuple[OrderTiming, ...]
    final_state: ExecutionState
    metrics: ExecutionMetrics
    decisions: tuple[PolicyDecision, ...] = ()
    liquidity_rule: LiquidityRule | None = None


class _ReplayLedger(StaticExecution):
    """Rebase liquidity without resetting child history or parent reservations."""

    def __init__(self, parent, model):
        super().__init__(parent, (), ())
        self._model = model
        self._debits = {}

    def observe(self, market, event):
        debits = dict(self._debits)
        if self._model == "refresh_touched":
            if isinstance(event, SnapshotEvent):
                debits.clear()
            else:
                for update in event.updates:
                    debits.pop((update.side, update.price), None)
        numbers = list(debits.values()) + [v for level in market.bids + market.asks for v in level]
        with localcontext(_accounting(numbers)):
            sides = []
            for side, levels in (("BID", market.bids), ("ASK", market.asks)):
                residual = []
                for price, quantity in levels:
                    available = max(ZERO, quantity - debits.get((side, price), ZERO))
                    if available:
                        residual.append((price, available))
                sides.append(tuple(residual))
        candidate = self._build(self.state.parent, sides[0], sides[1], self.state.children, self.state.version + 1)
        self._state, self._debits = candidate, debits

    def execute(self, child_id):
        # Compute the entire transition privately: an unsupported debit sum
        # must not publish fills before the liquidity-accounting error appears.
        candidate = copy(self)
        StaticExecution.execute(candidate, child_id)
        child = next(c for c in candidate.state.children if c.child_id == child_id)
        debits = dict(self._debits)
        side = "ASK" if self.state.parent.side == "BUY" else "BID"
        numbers = list(debits.values()) + [q for _, q in child.fills]
        with localcontext(_accounting(numbers)):
            for price, quantity in child.fills:
                key = (side, price)
                debits[key] = debits.get(key, ZERO) + quantity
        self._state, self._debits = candidate.state, debits
        return self.state


def _validate(events, config, strategy):
    _container(events, "events")
    if not isinstance(config, ExecutionConfig) or not isinstance(strategy, StrategySpec):
        raise TypeError("config and strategy must use their declared dataclasses")
    for field in ("start_time_ns", "deadline_ns", "observation_start_ns", "observation_end_ns", "order_latency_ns"):
        _integer(getattr(config, field), field)
    if not config.observation_start_ns <= config.start_time_ns <= config.deadline_ns <= config.observation_end_ns:
        raise ValueError("start/deadline must be ordered within the observation window")
    if config.max_book_age_ns is not None:
        _integer(config.max_book_age_ns, "max_book_age_ns")
    if config.liquidity_model not in ("persistent_debit", "refresh_touched"):
        raise ValueError("unknown liquidity model")
    _limit(config.limit_price)
    if strategy.kind not in ("immediate", "cumulative_twap", "cost_gated_twap"):
        raise ValueError("strategy must be immediate, cumulative_twap or cost_gated_twap")
    _integer(strategy.slices, "slices")
    if strategy.slices == 0 or (strategy.kind == "immediate" and strategy.slices != 1):
        raise ValueError("slices must be positive; immediate requires one slice")
    if type(strategy.catch_up) is not bool:
        raise ValueError("catch_up must be bool")
    if strategy.slices > 1 and strategy.slices > config.deadline_ns - config.start_time_ns:
        raise ValueError("slice ticks must be distinct at nanosecond resolution")
    indices = set()
    for event in events:
        if not isinstance(event, (SnapshotEvent, UpdateEvent)):
            raise TypeError("events must contain SnapshotEvent or UpdateEvent")
        _integer(event.visible_time_ns, "visible_time_ns")
        _integer(event.event_index, "event_index")
        if event.source_time_ns is not None:
            _integer(event.source_time_ns, "source_time_ns")
        if not config.observation_start_ns <= event.visible_time_ns <= config.observation_end_ns:
            raise ValueError("event outside declared observation window")
        if event.event_index in indices:
            raise ValueError("event_index must be unique")
        indices.add(event.event_index)


def _schedule(parent, config, strategy):
    ticks = {}
    count = strategy.slices
    duration = config.deadline_ns - config.start_time_ns
    for index in range(count):
        time = config.start_time_ns + index * duration // count
        if index == count - 1:
            target = parent.quantity
        else:
            with localcontext(_accounting([parent.quantity, Decimal(count)])):
                numerator = parent.quantity * Decimal(index + 1)
            target = _context(RATIO_PRECISION).divide(numerator, Decimal(count))
            target = min(target, parent.quantity)
        ticks[time] = (target, "scheduled")
    catch_time = config.deadline_ns - config.order_latency_ns
    if strategy.catch_up and catch_time >= config.start_time_ns:
        ticks[catch_time] = (parent.quantity, "scheduled_and_catch_up" if catch_time in ticks else "catch_up")
    return sorted((time, target, label) for time, (target, label) in ticks.items())


def _metrics(state, market, usable):
    parent = state.parent
    mid = None
    numbers = [parent.quantity, parent.reference_price, state.filled_qty, state.remaining_qty, state.covered_cost]
    numbers += [p for p, _ in market.bids[:1] + market.asks[:1]]
    with localcontext(_accounting(numbers)):
        if usable and market.bids and market.asks:
            mid = (market.bids[0][0] + market.asks[0][0]) / Decimal("2")
        if state.remaining_qty == ZERO:
            opportunity = ZERO
        elif mid is not None:
            opportunity = (1 if parent.side == "BUY" else -1) * state.remaining_qty * (mid - parent.reference_price)
        else:
            opportunity = None
        cash = state.covered_cost + opportunity if opportunity is not None else None
        numerator = cash * BPS if cash is not None else None
        denominator = parent.quantity * parent.reference_price
    ratio = _context(RATIO_PRECISION)
    bps = ratio.divide(numerator, denominator) if numerator is not None and parent.quantity > ZERO else None
    completion = ratio.divide(state.filled_qty, parent.quantity) if parent.quantity > ZERO else None
    return ExecutionMetrics(
        parent.quantity, state.filled_qty, state.remaining_qty, completion,
        state.covered_cost, state.fees, mid, opportunity, cash, bps, cash is not None,
        None if cash is not None else "remaining quantity requires a valid fresh two-sided terminal midpoint",
    )


def run_timed_execution(
    events: Sequence[SnapshotEvent | UpdateEvent], parent: ParentOrder,
    config: ExecutionConfig, strategy: StrategySpec,
    *, liquidity_rule: LiquidityRule | None = None,
) -> TimedResult:
    """Use only current visible book and progress at predeclared decision ticks.

    At a tied timestamp: all market messages, queued arrivals, decisions, cutoff.
    A zero-latency order generated by a decision arrives before the next tied
    decision. Arrivals at cutoff execute; later ones expire without fills.
    Invalid/stale decisions skip, invalid/stale arrivals reject and release.
    All arithmetic/metadata errors fail the run; semantic feed errors invalidate
    at their visible time and require snapshot recovery under the book contract.
    """
    _validate(events, config, strategy)
    if strategy.kind == "cost_gated_twap":
        if not isinstance(liquidity_rule, LiquidityRule):
            raise TypeError("cost_gated_twap requires LiquidityRule")
    elif liquidity_rule is not None:
        raise ValueError("liquidity_rule is only valid for cost_gated_twap")
    ledger = _ReplayLedger(parent, config.liquidity_model)
    book = OrderBook()
    queue = [(e.visible_time_ns, 0, e.event_index, e) for e in events if e.visible_time_ns <= config.deadline_ns]
    try:
        for index, (time, target, label) in enumerate(_schedule(parent, config, strategy)):
            queue.append((time, 2, index, (target, label)))
        queue.append((config.deadline_ns, 3, 0, None))
        heapq.heapify(queue)
        last_visible = None
        trace, orders, decisions = [], [], []

        def condition(time):
            age = time - last_visible if last_visible is not None else None
            if not book.state.is_valid:
                return "invalid", age
            if config.max_book_age_ns is not None and age > config.max_book_age_ns:
                return "stale", age
            return None, age

        def record(time, kind, child_id=None, reason=None):
            _, age = condition(time)
            trace.append(TraceRecord(time, kind, child_id, reason, book.state, age, ledger.state))

        while queue:
            time, priority, index, item = heapq.heappop(queue)
            if time > config.deadline_ns:
                break
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
                    last_visible = time
                    ledger.observe(book.state, item)
                record(time, "market_accepted" if error is None else "market_rejected", reason=error)
            elif priority == 1:
                timing = orders[index]
                error, _ = condition(time)
                if error is not None:
                    ledger.cancel(timing.child_id)
                    status = "rejected_" + error
                else:
                    ledger.execute(timing.child_id)
                    status = next(c.status for c in ledger.state.children if c.child_id == timing.child_id)
                orders[index] = replace(timing, processed_time_ns=time, confirmation_time_ns=time, status=status, reason=error)
                record(time, "arrival_" + status, timing.child_id, error)
            elif priority == 2:
                target, label = item
                error, _ = condition(time)
                state = ledger.state
                view = PolicyView(time, target, parent, state.filled_qty, state.reserved_qty,
                                  state.available_qty, book.state.bids, book.state.asks,
                                  state.bids, state.asks, config.limit_price, error,
                                  label in ("catch_up", "scheduled_and_catch_up"))
                decision = decide(view, liquidity_rule)
                decisions.append(decision)
                quantity = decision.selected_qty
                if quantity == ZERO:
                    record(time, "decision_skipped", reason=decision.reason)
                    continue
                child_id = f"child-{len(orders) + 1}"
                ledger.submit(child_id, quantity, config.limit_price)
                arrival = time + config.order_latency_ns
                orders.append(OrderTiming(child_id, time, arrival))
                heapq.heappush(queue, (arrival, 1, len(orders) - 1, None))
                record(time, "decision_submitted", child_id, label)
            else:
                for order_index, timing in enumerate(orders):
                    if timing.status == "pending":
                        ledger.cancel(timing.child_id)
                        orders[order_index] = replace(timing, processed_time_ns=time, status="expired", reason="arrival after deadline")
                        record(time, "deadline_expired", timing.child_id, "arrival after deadline")
                record(time, "deadline")
                break
        error, _ = condition(config.deadline_ns)
        metrics = _metrics(ledger.state, book.state, error is None)
    except DecimalException as exc:
        raise ValueError("timed execution arithmetic exceeds supported decimal range or exact precision") from exc
    label = strategy.kind + ("_with_catch_up" if strategy.catch_up else "_no_catch_up")
    return TimedResult(parent, config, strategy, label, tuple(trace), tuple(orders), ledger.state, metrics, tuple(decisions), liquidity_rule)
