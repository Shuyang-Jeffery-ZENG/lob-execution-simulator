"""Independent business expectations for separate market and observation clocks.

All displayed helper times are milliseconds on one synthetic aligned clock.
These are mechanism checks, not estimates of real-market execution quality.
"""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from lob_lab.book import LevelUpdate
from lob_lab.execution import ChildOrder, ParentOrder
from lob_lab.policy import LiquidityRule
from lob_lab.timed_execution import StrategySpec
from lob_sim.engine import run_causal_execution
from lob_sim.models import CausalConfig, FeedDelivery, MarketMessage, RecoveryConfig


D = Decimal
MS = 1_000_000


def snapshot(time=0, index=0, sequence=100, *, bid="99", ask="101", size="10", bids=None, asks=None):
    return MarketMessage(
        state_time_ns=time * MS, event_index=index, sequence=sequence, kind="snapshot",
        bids=tuple((D(p), D(q)) for p, q in bids) if bids is not None else (() if bid is None else ((D(bid), D(size)),)),
        asks=tuple((D(p), D(q)) for p, q in asks) if asks is not None else (() if ask is None else ((D(ask), D(size)),)),
    )


def update(time, index, sequence, *changes):
    return MarketMessage(
        state_time_ns=time * MS, event_index=index, sequence=sequence,
        updates=tuple(LevelUpdate(side, D(price), D(quantity)) for side, price, quantity in changes),
    )


def delivery(index, time):
    return FeedDelivery(index, time * MS)


def run(messages, deliveries=None, *, quantity="4", side="BUY", fee="0", reference="100",
        start=0, deadline=20, order_delay=0, response_delay=0, model="persistent_debit",
        limit=None, kind="immediate", slices=1, catch_up=False, rule=None,
        receive_age=None, state_age=None, recovery=None):
    if deliveries is None:
        deliveries = [FeedDelivery(m.event_index, m.state_time_ns) for m in messages]
    config = CausalConfig(
        start_time_ns=start * MS, deadline_ns=deadline * MS,
        market_start_ns=0, market_end_ns=max(200, deadline) * MS,
        order_latency_ns=order_delay * MS, response_latency_ns=response_delay * MS,
        liquidity_model=model, limit_price=None if limit is None else D(limit),
        max_receive_age_ns=None if receive_age is None else receive_age * MS,
        max_state_age_ns=None if state_age is None else state_age * MS,
        recovery=RecoveryConfig() if recovery is None else recovery,
    )
    return run_causal_execution(
        messages, deliveries, ParentOrder("task", side, D(quantity), D(reference), D(fee)),
        config, StrategySpec(kind, slices, catch_up), liquidity_rule=rule,
    )


@pytest.mark.parametrize("bid,ask,notional,cost", [("109", "111", "444", "44"), ("89", "91", "364", "-36")])
def test_market_change_at_three_affects_arrival_at_five_but_not_view_until_eight(bid, ask, notional, cost):
    result = run(
        [snapshot(), snapshot(3, 1, 101, bid=bid, ask=ask)],
        [delivery(0, 0), delivery(1, 8)], order_delay=5, deadline=10,
    )
    decision = result.decisions[0]
    assert decision.view.raw_asks == ((D("101"), D("10")),)
    assert decision.local.provenance == (0,)
    assert result.orders[0].arrival_time_ns == 5 * MS
    assert result.orders[0].outcome.fills == ((D(ask), D("4")),)
    assert result.actual_state.notional == D(notional)
    assert result.metrics.covered_cost == result.metrics.is_cash == D(cost)
    at_five = [r for r in result.trace if r.time_ns == 5 * MS and r.actual.filled_qty == D("4")]
    assert at_five
    assert all(r.local.book.asks == ((D("101"), D("10")),) for r in at_five)


def test_local_sequence_gap_does_not_cancel_an_already_submitted_order():
    result = run(
        [snapshot(), update(9, 1, 101), update(10, 2, 102)],
        [delivery(0, 0), delivery(2, 10)], start=9, order_delay=3, quantity="5",
    )
    assert result.execution_known
    assert result.orders[0].outcome.fills == ((D("101"), D("5")),)
    assert result.actual_state.notional == D("505")
    at_arrival = [r for r in result.trace if r.time_ns == 12 * MS]
    assert at_arrival and all(not r.local.book.is_valid and r.market.is_valid for r in at_arrival)
    assert result.known_state.reserved_qty == D("0")


def test_local_gap_still_blocks_new_decisions_while_market_keeps_evolving():
    result = run(
        [snapshot(), update(4, 1, 101), update(5, 2, 102)],
        [delivery(0, 0), delivery(2, 5)], quantity="8", order_delay=2,
        deadline=12, kind="cumulative_twap", slices=2,
    )
    assert len(result.orders) == 1
    assert result.actual_state.filled_qty == D("4")
    assert result.decisions[1].decision.selected_qty == D("0")
    assert not result.decisions[1].local.book.is_valid
    assert result.terminal_market.is_valid


def test_delayed_partial_report_preserves_unknown_remainder_at_next_decision():
    result = run([snapshot(size="4")], quantity="12", deadline=12,
                 order_delay=5, response_delay=3, kind="cumulative_twap", slices=2)
    first, second = result.orders
    assert first.outcome.requested_qty == second.outcome.requested_qty == D("6")
    assert first.outcome.filled_qty == D("4") and first.outcome.unfilled_qty == D("2")
    assert first.response_time_ns == 8 * MS and second.response_time_ns == 14 * MS
    assert first.received_time_ns == 8 * MS and second.received_time_ns == 14 * MS
    view = result.decisions[1].view
    assert view.time_ns == 6 * MS
    assert (view.filled_qty, view.reserved_qty, view.available_qty) == (D("0"), D("6"), D("6"))
    assert view.asks == ((D("101"), D("4")),)  # Market-side exhaustion is not yet known.
    assert result.decisions[1].decision.selected_qty == D("6")
    assert (result.actual_state.filled_qty, result.actual_state.notional) == (D("4"), D("404"))
    assert result.known_state.filled_qty == D("4") and result.known_state.reserved_qty == D("0")
    assert max(r.time_ns for r in result.trace) == 14 * MS


def test_private_report_can_legally_reveal_a_fill_before_the_new_public_quote():
    result = run(
        [snapshot(), snapshot(3, 1, 101, bid="109", ask="111")],
        [delivery(0, 0), delivery(1, 8)], quantity="8", deadline=12,
        order_delay=5, kind="cumulative_twap", slices=2,
    )
    view = result.decisions[1].view
    assert view.time_ns == 6 * MS and view.filled_qty == D("4")
    assert view.raw_asks == ((D("101"), D("10")),)
    assert result.orders[0].outcome.fills == ((D("111"), D("4")),)


def test_fill_before_deadline_is_settled_when_report_arrives_after_deadline():
    result = run([snapshot()], start=8, deadline=10, order_delay=1, response_delay=3)
    assert result.orders[0].arrival_time_ns == 9 * MS
    assert result.orders[0].response_time_ns == 12 * MS
    assert result.metrics.filled_qty == D("4") and result.metrics.is_cash == D("4")
    assert result.known_state.filled_qty == D("4") and result.known_state.reserved_qty == D("0")
    assert all(d.view.time_ns <= 10 * MS for d in result.decisions)
    assert max(r.time_ns for r in result.trace) == 12 * MS


def test_known_late_arrival_is_not_sent_and_does_not_reserve_parent_quantity():
    result = run([snapshot()], start=8, deadline=10, order_delay=3)
    assert result.orders == ()
    assert result.actual_state.filled_qty == result.known_state.reserved_qty == D("0")
    assert result.known_state.available_qty == D("4")
    assert result.decisions[0].decision.selected_qty == D("0")


def test_order_arriving_exactly_at_deadline_executes_before_cutoff():
    result = run([snapshot()], start=8, deadline=10, order_delay=2, response_delay=3)
    assert result.orders[0].arrival_time_ns == 10 * MS
    assert result.orders[0].response_time_ns == 13 * MS
    assert result.actual_state.filled_qty == result.known_state.filled_qty == D("4")


def test_source_gap_at_arrival_stops_with_unknown_order_not_fabricated_zero_fill():
    result = run(
        [snapshot(), update(7, 1, 102), snapshot(10, 2, 103)],
        quantity="12", deadline=18, order_delay=2, kind="cumulative_twap", slices=3,
    )
    assert not result.execution_known and result.metrics is None
    assert result.status == "execution_unknown" and result.orders[1].status == "unknown"
    assert len(result.orders) == 2 and result.orders[0].outcome.filled_qty == D("4")
    assert result.orders[1].outcome is None and result.orders[1].response_time_ns is None
    assert result.known_state.filled_qty == D("4")
    assert result.known_state.reserved_qty == D("4")
    assert len(result.decisions) == 2 and max(r.time_ns for r in result.trace) == 8 * MS
    assert result.reason
    assert any(not r.execution_known for r in result.trace)


def test_unknown_execution_does_not_pretend_an_earlier_scheduled_report_was_received():
    result = run([snapshot(), update(7, 1, 102)], quantity="12", deadline=18,
                 order_delay=2, response_delay=10, kind="cumulative_twap", slices=3)
    first, second = result.orders
    assert not result.execution_known and result.metrics is None
    assert first.outcome.filled_qty == D("4") and first.execution_sequence == 100
    assert first.response_time_ns == 12 * MS and first.received_time_ns is None
    assert second.outcome is None and second.response_time_ns is None and second.received_time_ns is None
    assert result.actual_state.filled_qty == D("4")  # Known prefix, not a claim about the unknown child's total.
    assert result.known_state.filled_qty == D("0") and result.known_state.reserved_qty == D("8")
    assert max(r.time_ns for r in result.trace) == 8 * MS


def test_source_recovery_before_arrival_allows_execution_even_while_local_is_invalid():
    result = run(
        [snapshot(), update(4, 1, 102), snapshot(6, 2, 103, bid="109", ask="111")],
        [delivery(0, 0), delivery(1, 4), delivery(2, 10)], order_delay=7,
    )
    assert result.execution_known and result.orders[0].outcome.fills == ((D("111"), D("4")),)
    assert any(r.time_ns == 7 * MS and r.market.is_valid and not r.local.book.is_valid for r in result.trace)


def test_terminal_market_values_known_partial_fills_despite_local_invalidity():
    result = run(
        [snapshot(size="6"), snapshot(8, 1, 101, bid="102", ask="104"), update(9, 2, 102)],
        [delivery(0, 0), delivery(2, 9)], quantity="10", deadline=10,
    )
    assert result.execution_known and result.metrics.evaluation_available
    assert result.metrics.filled_qty == D("6") and result.metrics.remaining_qty == D("4")
    assert result.metrics.covered_cost == D("6") and result.metrics.terminal_mid == D("103")
    assert result.metrics.opportunity_cost == D("12") and result.metrics.is_cash == D("18")
    assert result.metrics.is_bps == D("180") and result.metrics.completion_ratio == D("0.6")
    assert not result.trace[-1].local.book.is_valid


@pytest.mark.parametrize("quantity,expected_cash", [("10", None), ("4", D("4"))])
def test_missing_market_endpoint_only_prevents_valuation_when_known_remainder_exists(quantity, expected_cash):
    result = run([snapshot(size="6"), update(9, 1, 102)], quantity=quantity, deadline=10)
    assert result.execution_known and result.metrics is not None
    assert result.metrics.is_cash == expected_cash
    assert result.metrics.evaluation_available is (expected_cash is not None)
    assert result.metrics.terminal_mid is None


def test_later_snapshot_cannot_repair_deadline_mark_while_waiting_for_private_report():
    result = run(
        [snapshot(size="6"), update(9, 1, 102), snapshot(12, 2, 103, bid="102", ask="104")],
        quantity="10", deadline=10, response_delay=15,
    )
    assert result.execution_known and result.known_state.filled_qty == D("6")
    assert result.metrics.terminal_mid is None and result.metrics.is_cash is None
    assert not result.terminal_market.is_valid
    assert all(r.time_ns <= 10 * MS or r.time_ns == 15 * MS for r in result.trace)


@pytest.mark.parametrize("side,limit,filled,notional,fees,cost,bps", [
    ("BUY", None, "4", "406", "0.0812", "6.0812", "152.03"),
    ("SELL", None, "4", "394", "0.0788", "6.0788", "151.97"),
    ("BUY", "101", "2", "202", "0.0404", "2.0404", "51.01"),
    ("SELL", "99", "2", "198", "0.0396", "2.0396", "50.99"),
])
def test_buy_sell_fees_protection_and_partial_accounting(side, limit, filled, notional, fees, cost, bps):
    market = snapshot(bids=(("99", "2"), ("98", "5")), asks=(("101", "2"), ("102", "5")))
    result = run([market], side=side, fee="2", limit=limit)
    assert result.actual_state.filled_qty == D(filled)
    assert result.actual_state.notional == D(notional) and result.metrics.fees == D(fees)
    assert result.metrics.is_cash == D(cost) and result.metrics.is_bps == D(bps)
    assert result.known_state.reserved_qty == D("0")
    assert result.known_state.available_qty == D("4") - D(filled)


@pytest.mark.parametrize("model,before,after", [("refresh_touched", "12", "8"), ("persistent_debit", "8", "4")])
def test_external_depth_rules_preserve_12_or_8_before_second_fill(model, before, after):
    result = run([snapshot(), update(5, 1, 101, ("ASK", "101", "12"))], quantity="8",
                 deadline=10, kind="cumulative_twap", slices=2, model=model)
    assert result.decisions[1].view.asks == ((D("101"), D(before)),)
    assert result.actual_state.asks == ((D("101"), D(after)),)
    assert result.known_state.asks == result.actual_state.asks
    assert result.actual_state.notional == D("808") and result.metrics.filled_qty == D("8")
    assert result.terminal_market.asks == ((D("101"), D("12")),)


@pytest.mark.parametrize("model,at_first_report,final", [("refresh_touched", "12", "8"), ("persistent_debit", "8", "4")])
def test_late_report_does_not_resurrect_consumption_already_covered_by_external_refresh(model, at_first_report, final):
    result = run([snapshot(), update(3, 1, 101, ("ASK", "101", "12"))], quantity="8",
                 deadline=10, response_delay=7, kind="cumulative_twap", slices=2, model=model)
    at_seven = [r for r in result.trace if r.time_ns == 7 * MS and r.known.filled_qty == D("4")]
    assert at_seven and at_seven[-1].known.asks == ((D("101"), D(at_first_report)),)
    assert result.actual_state.asks == result.known_state.asks == ((D("101"), D(final)),)
    assert result.known_state.reserved_qty == D("0")


def test_snapshot_service_restores_observation_without_regenerating_market_liquidity():
    recovery = RecoveryConfig(snapshot_read_latency_ns=0, snapshot_response_latency_ns=MS, max_requests=1)
    result = run(
        [snapshot(), update(2, 1, 101), update(3, 2, 102)],
        [delivery(0, 0), delivery(2, 3)], quantity="8", deadline=10,
        kind="cumulative_twap", slices=2, model="refresh_touched", recovery=recovery,
    )
    assert len(result.orders) == 2 and result.decisions[1].local.book.is_valid
    assert result.decisions[1].local.book.sequence == 102
    assert result.actual_state.asks == ((D("101"), D("2")),)
    assert result.actual_state.notional == D("808")


def test_zero_delay_snapshot_service_bootstraps_before_initial_decision_without_push_feed():
    recovery = RecoveryConfig(snapshot_read_latency_ns=0, snapshot_response_latency_ns=0, max_requests=1)
    result = run([snapshot()], [], deadline=10, recovery=recovery)
    assert result.decisions[0].view.time_ns == 0 and result.decisions[0].local.book.is_valid
    assert result.decisions[0].local.provenance == (2**63,)
    assert result.orders[0].outcome.fills == ((D("101"), D("4")),)


def test_future_input_index_cannot_change_identity_of_an_earlier_service_snapshot():
    recovery = RecoveryConfig(snapshot_read_latency_ns=0, snapshot_response_latency_ns=0, max_requests=1)
    future = snapshot(15, 42, 101, bid="109", ask="111")
    modified = replace(future, event_index=2**62)
    results = [run([snapshot(index=7), message], [], deadline=10, recovery=recovery)
               for message in (future, modified)]
    assert results[0].decisions == results[1].decisions
    assert results[0].decisions[0].local.provenance == (2**63,)


def test_same_timestamp_market_feed_arrival_response_then_decision():
    result = run([snapshot(), snapshot(5, 1, 101, bid="109", ask="111")], quantity="8",
                 deadline=10, order_delay=5, kind="cumulative_twap", slices=2)
    second = result.decisions[1].view
    assert second.time_ns == 5 * MS
    assert second.raw_asks == ((D("111"), D("10")),)
    assert second.filled_qty == D("4") and second.reserved_qty == D("0")
    assert second.asks == ((D("111"), D("6")),)
    assert [o.outcome.fills for o in result.orders] == [((D("111"), D("4")),)] * 2
    assert result.actual_state.notional == D("888")


def test_hidden_market_changes_cannot_leak_before_either_feed_or_report_arrives():
    common = snapshot()
    future_a = snapshot(4, 1, 101, bid="109", ask="111")
    future_b = replace(future_a, asks=((D("121"), D("1")),))
    results = [run([common, future], [delivery(0, 0), delivery(1, 8)], quantity="8",
                   deadline=12, order_delay=5, response_delay=5,
                   kind="cumulative_twap", slices=2) for future in (future_a, future_b)]
    # Market fills differ at 5ms; neither public update (8) nor private report
    # (10) is available at the two decisions (0, 6).
    assert results[0].decisions == results[1].decisions
    assert [d.decision.selected_qty for d in results[0].decisions] == [D("4"), D("4")]
    assert results[0].orders[0].outcome.filled_qty == D("4")
    assert results[1].orders[0].outcome.filled_qty == D("1")
    assert results[0].orders[0].outcome.notional != results[1].orders[0].outcome.notional


def test_receive_age_and_state_age_are_independent_gates():
    message = snapshot()
    received_only = run([message], [delivery(0, 100)], start=100, deadline=101, receive_age=5)
    state_checked = run([message], [delivery(0, 100)], start=100, deadline=101, receive_age=5, state_age=20)
    assert received_only.actual_state.filled_qty == D("4")
    assert state_checked.orders == () and state_checked.actual_state.filled_qty == D("0")
    view = state_checked.decisions[0].local
    assert view.book.is_valid and view.last_receive_time_ns == 100 * MS and view.state_time_ns == 0


def test_cost_gate_uses_local_estimate_and_output_views_are_immutable():
    result = run([snapshot()], kind="cost_gated_twap", rule=LiquidityRule(D("99")))
    assert result.orders == ()
    assert result.decisions[0].decision.reason == "cost_above_cap"
    with pytest.raises(FrozenInstanceError):
        result.decisions[0].view.filled_qty = D("999")


def test_zero_target_keeps_known_zero_cash_without_inventing_a_ratio():
    result = run([snapshot()], quantity="0")
    assert result.execution_known and result.orders == ()
    assert result.metrics.is_cash == D("0") and result.metrics.is_bps is None
    assert result.metrics.completion_ratio is None


def test_repeated_run_is_deterministic_for_the_same_synthetic_inputs():
    messages = [snapshot(), snapshot(3, 1, 101, bid="109", ask="111")]
    feeds = [delivery(0, 0), delivery(1, 8)]
    assert run(messages, feeds, order_delay=5, response_delay=3) == run(messages, feeds, order_delay=5, response_delay=3)


@pytest.mark.parametrize("messages,feeds", [
    ([snapshot(5)], [delivery(0, 4)]),
    ([snapshot()], [delivery(99, 0)]),
    ([snapshot(), snapshot(1, 0, 101)], [delivery(0, 0)]),
])
def test_invalid_delivery_provenance_or_duplicate_source_indices_rejected(messages, feeds):
    with pytest.raises((ValueError, TypeError)):
        run(messages, feeds)


def test_terminal_report_transition_is_idempotent_and_rejects_conflicting_duplicates():
    # The public runner generates each terminal report once; exercise the
    # contract's duplicate-delivery claim directly at the receiving boundary.
    from lob_sim.engine import _KnownOrders

    parent = ParentOrder("receipt", "BUY", D("10"), D("100"))
    known = _KnownOrders(parent, "persistent_debit")
    known.submit("c1", D("6"), None)
    report = ChildOrder(
        "c1", D("6"), None, status="partial", filled_qty=D("4"), unfilled_qty=D("2"),
        fills=((D("101"), D("4")),), notional=D("404"), covered_cost=D("4"),
    )
    assert known.receive(report, 100) is True
    accepted = known.state
    assert (accepted.filled_qty, accepted.reserved_qty, accepted.available_qty) == (D("4"), D("0"), D("6"))
    assert accepted.notional == D("404") and accepted.covered_cost == D("4")
    assert known.receive(report, 100) is False
    assert known.state is accepted

    changed_fill = replace(report, filled_qty=D("3"), unfilled_qty=D("3"),
                           fills=((D("101"), D("3")),), notional=D("303"), covered_cost=D("3"))
    for conflicting_report, sequence in ((changed_fill, 100), (report, 101)):
        with pytest.raises(ValueError, match="conflicting"):
            known.receive(conflicting_report, sequence)
        assert known.state is accepted
