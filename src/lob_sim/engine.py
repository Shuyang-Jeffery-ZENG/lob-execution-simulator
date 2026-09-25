"""Single-clock causal sandbox with separate market, observation and known orders.

Only explicitly delivered information reaches PolicyView. This module models
active IOC children on an exogenous reference, not a responsive exchange.
"""
from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal, DecimalException, localcontext
import heapq

from lob_lab.book import BookError, OrderBook
from lob_lab.execution import StaticExecution, _accounting, _identifier, _limit
from lob_lab.policy import LiquidityRule, PolicyView, decide
from lob_lab.replay import SnapshotEvent, UpdateEvent, _container, _integer
from lob_lab.timed_execution import StrategySpec, _ReplayLedger, _metrics, _schedule

from .models import (
    CausalConfig, CausalResult, CausalTrace, DecisionRecord, FeedDelivery,
    MarketMessage, OrderRecord, RecoveryConfig,
)
from .recovery import LocalRecovery

ZERO = Decimal("0")


def _event(message):
    """Bridge the unchanged baseline liquidity rule, never its clock."""
    if message.kind == "snapshot":
        return SnapshotEvent(message.state_time_ns, message.event_index,
                             message.sequence, message.bids, message.asks,
                             message.state_time_ns)
    return UpdateEvent(message.state_time_ns, message.event_index,
                       message.sequence, message.updates, message.state_time_ns)


def _apply(book, message):
    if message.kind == "snapshot":
        return book.apply_snapshot(sequence=message.sequence,
                                   bids=message.bids, asks=message.asks)
    return book.apply_update(sequence=message.sequence, updates=message.updates)


def _validate(messages, deliveries, config, strategy, liquidity_rule):
    _container(messages, "messages")
    _container(deliveries, "deliveries")
    if not isinstance(config, CausalConfig) or not isinstance(strategy, StrategySpec):
        raise TypeError("config and strategy must use the declared dataclasses")
    for name in ("start_time_ns", "deadline_ns", "market_start_ns", "market_end_ns",
                 "order_latency_ns", "response_latency_ns"):
        _integer(getattr(config, name), name)
    if not config.market_start_ns <= config.start_time_ns <= config.deadline_ns <= config.market_end_ns:
        raise ValueError("task must be contained in the declared market recording")
    _identifier(config.session_id, "session_id")
    for name in ("max_receive_age_ns", "max_state_age_ns"):
        if getattr(config, name) is not None:
            _integer(getattr(config, name), name)
    if config.liquidity_model not in ("persistent_debit", "refresh_touched"):
        raise ValueError("unknown liquidity model")
    _limit(config.limit_price)
    recovery = config.recovery
    if not isinstance(recovery, RecoveryConfig):
        raise TypeError("recovery must be RecoveryConfig")
    for name in ("max_buffer", "timeout_ns", "max_requests", "snapshot_response_latency_ns"):
        _integer(getattr(recovery, name), name)
    if recovery.max_buffer == 0 or recovery.timeout_ns == 0:
        raise ValueError("max_buffer and timeout_ns must be positive")
    if recovery.snapshot_read_latency_ns is not None:
        _integer(recovery.snapshot_read_latency_ns, "snapshot_read_latency_ns")
    if strategy.kind not in ("immediate", "cumulative_twap", "cost_gated_twap"):
        raise ValueError("unknown strategy")
    _integer(strategy.slices, "slices")
    if strategy.slices == 0 or (strategy.kind == "immediate" and strategy.slices != 1):
        raise ValueError("positive slices required; immediate requires one slice")
    if strategy.slices > 1 and strategy.slices > config.deadline_ns - config.start_time_ns:
        raise ValueError("slice ticks must be distinct")
    if type(strategy.catch_up) is not bool:
        raise ValueError("catch_up must be bool")
    if strategy.kind == "cost_gated_twap":
        if not isinstance(liquidity_rule, LiquidityRule):
            raise TypeError("cost_gated_twap requires LiquidityRule")
    elif liquidity_rule is not None:
        raise ValueError("liquidity_rule requires cost_gated_twap")
    by_index = {}
    for message in messages:
        if not isinstance(message, MarketMessage):
            raise TypeError("messages must contain MarketMessage")
        for name in ("state_time_ns", "event_index", "sequence"):
            _integer(getattr(message, name), name)
        if message.event_index >= 2**63:
            raise ValueError("source event_index must be below 2**63; upper namespace is reserved")
        if not config.market_start_ns <= message.state_time_ns <= config.market_end_ns:
            raise ValueError("market message outside recording")
        if message.session_id != config.session_id:
            raise ValueError("market message belongs to another session")
        if message.kind not in ("snapshot", "update"):
            raise ValueError("unknown message kind")
        if type(message.full_depth) is not bool or not message.full_depth:
            raise ValueError("v1 requires full-depth reference messages")
        # Freeze nested message containers: a caller cannot mutate historical views.
        for name in ("bids", "asks", "updates"):
            if not isinstance(getattr(message, name), tuple):
                raise TypeError(f"{name} must be an immutable tuple")
        if any(not isinstance(level, tuple) for level in message.bids + message.asks):
            raise TypeError("levels must be immutable tuples")
        if message.kind == "snapshot" and message.updates:
            raise ValueError("snapshot cannot contain updates")
        if message.kind == "update" and (message.bids or message.asks):
            raise ValueError("update cannot contain snapshot levels")
        if message.event_index in by_index:
            raise ValueError("event_index must be unique")
        by_index[message.event_index] = message
    for delivery in deliveries:
        if not isinstance(delivery, FeedDelivery):
            raise TypeError("deliveries must contain FeedDelivery")
        _integer(delivery.event_index, "delivery event_index")
        _integer(delivery.receive_time_ns, "receive_time_ns")
        if delivery.event_index not in by_index:
            raise ValueError("delivery references unknown message")
        if delivery.receive_time_ns < by_index[delivery.event_index].state_time_ns:
            raise ValueError("feed cannot arrive before its source state exists")
    return by_index


class _KnownOrders:
    """Submission reservations plus delivered terminal reports, no market access."""

    def __init__(self, parent, model):
        self._ledger = StaticExecution(parent, (), ())
        self._model = model
        self._receipts = {}  # child -> (terminal child, reference sequence at fill)
        self._refresh_all = -1
        self._refresh_levels = {}
        self._book = None

    @property
    def state(self):
        return self._ledger.state

    def submit(self, child_id, quantity, limit):
        self._ledger.submit(child_id, quantity, limit)

    def observe(self, view, applied):
        self._book = view.book
        for message in applied:
            if message.kind == "snapshot":
                self._refresh_all = max(self._refresh_all, message.sequence)
            else:
                for update in message.updates:
                    key = (update.side, update.price)
                    self._refresh_levels[key] = max(
                        self._refresh_levels.get(key, -1), message.sequence)
        self._rebuild()

    def receive(self, child, sequence):
        if child.child_id in self._receipts:
            if self._receipts[child.child_id] != (child, sequence):
                raise ValueError("conflicting duplicate order report")
            return False
        index = self._ledger._child_index(child.child_id)
        prior = self.state.children[index]
        if child.requested_qty != prior.requested_qty or child.limit_price != prior.limit_price:
            raise ValueError("report does not match submission")
        self._receipts[child.child_id] = (child, sequence)
        children = list(self.state.children)
        children[index] = child
        self._rebuild(tuple(children))
        return True

    def _rebuild(self, children=None):
        state = self.state
        children = state.children if children is None else children
        book = self._book
        sides = (book.bids, book.asks) if book is not None else ((), ())
        numbers = [q for levels in sides for _, q in levels]
        numbers += [q for child, _ in self._receipts.values() for _, q in child.fills]
        with localcontext(_accounting(numbers)):
            debits = {}
            side = "ASK" if state.parent.side == "BUY" else "BID"
            for child, sequence in self._receipts.values():
                for price, quantity in child.fills:
                    key = (side, price)
                    refresh = max(self._refresh_all, self._refresh_levels.get(key, -1))
                    debit = self._model == "persistent_debit" or sequence >= refresh
                    if debit:
                        debits[key] = debits.get(key, ZERO) + quantity
            residual = []
            for side, levels in zip(("BID", "ASK"), sides):
                residual.append(tuple((price, quantity - debits.get((side, price), ZERO))
                                      for price, quantity in levels
                                      if quantity > debits.get((side, price), ZERO)))
        self._ledger._state = StaticExecution._build(
            state.parent, residual[0], residual[1], children, state.version + 1)


def run_causal_execution(
    messages: Sequence[MarketMessage], deliveries: Sequence[FeedDelivery],
    parent, config: CausalConfig, strategy: StrategySpec,
    *, liquidity_rule: LiquidityRule | None = None,
) -> CausalResult:
    return _run_execution(messages, deliveries, parent, config, strategy, liquidity_rule=liquidity_rule)


def _run_execution(messages, deliveries, parent, config, strategy, *, liquidity_rule=None):
    """Run deterministic synthetic IOC execution; see docs/design.md.

    Malformed metadata fails before replay. Semantically bad book messages
    invalidate their recipient at their own event time. Unknown execution aborts
    without claiming zero fills, releasing reservations or imputing IS.
    """
    by_index = _validate(messages, deliveries, config, strategy, liquidity_rule)
    run_end = config.deadline_ns
    market = OrderBook()
    local = LocalRecovery(config.session_id, config.recovery)
    actual = _ReplayLedger(parent, config.liquidity_model)
    known = _KnownOrders(parent, config.liquidity_model)
    queue = []
    serial = 0

    def enqueue(time, priority, kind, payload=None):
        nonlocal serial
        heapq.heappush(queue, (time, priority, serial, kind, payload))
        serial += 1

    for message in sorted(messages, key=lambda m: (m.state_time_ns, m.event_index)):
        if message.state_time_ns <= run_end:
            enqueue(message.state_time_ns, 0, "market", message)
    for delivery in deliveries:
        if delivery.receive_time_ns <= run_end:
            enqueue(delivery.receive_time_ns, 2, "feed", by_index[delivery.event_index])
    for time, target, label in _schedule(parent, config, strategy):
        enqueue(time, 5, "decision", (target, label))
    # Initialize after same-time pushed messages, before the first decision.
    enqueue(config.market_start_ns, 2, "bootstrap")
    enqueue(config.deadline_ns, 7, "deadline")
    trace, decisions, orders = [], [], []
    requests = 0
    request_pending = False
    request_token = 0
    timeout_generation = None
    terminal_market = market.state
    execution_known = True
    failure = None
    closed = False
    # Generated snapshot identities never collide with supplied market indices.
    generated_index = 2**63

    def record(time, kind, detail=None):
        trace.append(CausalTrace(time, kind, detail, market.state, local.view,
                                known.state, actual.state, execution_known))

    def request_recovery(time):
        nonlocal requests, request_pending, request_token, timeout_generation
        if local.view.book.is_valid or time > config.deadline_ns:
            return
        since = local.view.recovering_since_ns
        if since is None:
            since = time
        if timeout_generation != (since, local.view.publication):
            timeout_generation = (since, local.view.publication)
            enqueue(max(time, since + config.recovery.timeout_ns), 6,
                    "timeout", timeout_generation)
        if (config.recovery.snapshot_read_latency_ns is not None
                and not request_pending and requests < config.recovery.max_requests):
            requests += 1
            request_token += 1
            request_pending = True
            enqueue(time + config.recovery.snapshot_read_latency_ns, 1,
                    "snapshot_read", request_token)
            record(time, "snapshot_requested", str(request_token))

    def local_condition(time):
        view = local.view
        if not view.book.is_valid:
            return "invalid"
        if (config.max_receive_age_ns is not None and
                (view.last_receive_time_ns is None or
                 time - view.last_receive_time_ns > config.max_receive_age_ns)):
            return "stale_receive"
        if (config.max_state_age_ns is not None and
                (view.state_time_ns is None or
                 time - view.state_time_ns > config.max_state_age_ns)):
            return "stale_state"
        return None

    try:
        while queue:
            time, _, _, kind, payload = heapq.heappop(queue)
            if closed and kind != "report":
                continue
            if time > config.deadline_ns and kind != "report":
                continue
            if kind == "market":
                try:
                    _apply(market, payload)
                except BookError as exc:
                    record(time, "market_invalid", str(exc))
                else:
                    actual.observe(market.state, _event(payload))
                    record(time, "market_applied", str(payload.event_index))
            elif kind == "feed":
                outcome = local.receive(payload, time)
                if outcome.published:
                    known.observe(local.view, outcome.applied)
                record(time, "feed_published" if outcome.published else "feed_received",
                       f"{payload.event_index}: {outcome.reason or 'accepted'}")
                request_recovery(time)
            elif kind == "bootstrap":
                # Initialization may have no pushed snapshot at all.
                local.check_timeout(time)
                request_recovery(time)
            elif kind == "snapshot_read":
                if payload != request_token or not request_pending:
                    continue
                if local.view.book.is_valid:
                    request_pending = False
                    continue
                if market.state.is_valid:
                    snapshot = MarketMessage(time, generated_index, market.state.sequence,
                                             "snapshot", market.state.bids, market.state.asks,
                                             session_id=config.session_id)
                    generated_index += 1
                    enqueue(time + config.recovery.snapshot_response_latency_ns, 2,
                            "snapshot_response", (payload, snapshot))
                    record(time, "snapshot_read", f"request={payload}, sequence={snapshot.sequence}, index={snapshot.event_index}")
                else:
                    record(time, "snapshot_unavailable", str(payload))
                    # Keep this request outstanding until bounded timeout/retry.
            elif kind == "snapshot_response":
                token, snapshot = payload
                if token != request_token or not request_pending:
                    record(time, "snapshot_response_ignored", "expired request")
                    continue
                request_pending = False
                if local.view.book.is_valid:
                    record(time, "snapshot_response_ignored", "already recovered")
                    continue
                outcome = local.receive(snapshot, time)
                if outcome.published:
                    known.observe(local.view, outcome.applied)
                record(time, "snapshot_published" if outcome.published else "snapshot_rejected",
                       f"index={snapshot.event_index}, sequence={snapshot.sequence}: {outcome.reason or 'accepted'}")
                # A failed response waits for the existing timeout, avoiding a
                # zero-latency request loop at one instant.
            elif kind == "timeout":
                if local.view.book.is_valid or payload != timeout_generation:
                    continue
                outcome = local.check_timeout(time)
                record(time, "recovery_timeout", outcome.reason)
                request_pending = False
                request_token += 1  # Late responses from old requests are obsolete.
                timeout_generation = None
                # Retry only while bounded; exhausted requests remain invalid.
                if requests < config.recovery.max_requests and config.recovery.snapshot_read_latency_ns is not None:
                    request_recovery(time)
            elif kind == "arrival":
                index = payload
                order = orders[index]
                if not market.state.is_valid:
                    execution_known = False
                    failure = "market reference unavailable at order arrival; execution unknown"
                    orders[index] = replace(order, status="unknown")
                    terminal_market = market.state
                    record(time, "execution_unknown", order.child_id)
                    break
                else:
                    actual.execute(order.child_id)
                child = next(c for c in actual.state.children if c.child_id == order.child_id)
                response_time = time + config.response_latency_ns
                orders[index] = replace(order, status=child.status, outcome=child,
                                        response_time_ns=response_time,
                                        execution_sequence=market.state.sequence)
                enqueue(response_time, 4, "report", (index, child, market.state.sequence))
                record(time, "order_executed", order.child_id)
            elif kind == "report":
                index, child, sequence = payload
                applied = known.receive(child, sequence)
                orders[index] = replace(orders[index], received_time_ns=time)
                record(time, "report_received" if applied else "duplicate_report", orders[index].child_id)
            elif kind == "decision":
                target, label = payload
                reason = local_condition(time)
                if time + config.order_latency_ns > config.deadline_ns:
                    reason = "arrival_after_deadline"
                view = PolicyView(time, target, parent, known.state.filled_qty,
                                  known.state.reserved_qty, known.state.available_qty,
                                  local.view.book.bids, local.view.book.asks,
                                  known.state.bids, known.state.asks, config.limit_price,
                                  reason, "catch_up" in label)
                decision = decide(view, liquidity_rule)
                decisions.append(DecisionRecord(decision, view, local.view))
                if decision.selected_qty:
                    child_id = f"child-{len(orders) + 1}"
                    known.submit(child_id, decision.selected_qty, config.limit_price)
                    actual.submit(child_id, decision.selected_qty, config.limit_price)
                    arrival = time + config.order_latency_ns
                    orders.append(OrderRecord(child_id, time, arrival))
                    enqueue(arrival, 3, "arrival", len(orders) - 1)
                    record(time, "decision_submitted", child_id)
                else:
                    record(time, "decision_skipped", decision.reason)
            elif kind == "deadline":
                terminal_market = market.state
                closed = True
                record(time, "deadline")
            else:
                raise RuntimeError(f"unknown causal event: {kind}")
        metrics = _metrics(actual.state, terminal_market, terminal_market.is_valid) if execution_known else None
    except DecimalException as exc:
        raise ValueError("causal execution arithmetic exceeds supported exact precision") from exc
    contract = "causal-replay-v1"
    status = "complete" if execution_known else "execution_unknown"
    result = CausalResult(contract, parent, config, strategy, liquidity_rule,
                        tuple(orders), tuple(decisions), tuple(trace), actual.state,
                        known.state, terminal_market, metrics, execution_known,
                        status, failure)
    return result
