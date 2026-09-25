"""Independent synthetic accounting examples; no timing or market-impact model."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, Inexact, ROUND_UP, Rounded, localcontext

import pytest

from lob_lab.execution import ParentOrder, StaticExecution


D = Decimal
BIDS = ((D("99.99"), D("30")), (D("99.97"), D("40")), (D("99.90"), D("50")))
ASKS = ((D("100.01"), D("30")), (D("100.03"), D("40")), (D("100.10"), D("50")))


def parent(side="BUY", quantity="80", reference="100", fee="2"):
    return ParentOrder("parent", side, D(quantity), D(reference), D(fee))


def execution(side="BUY", quantity="80", *, bids=BIDS, asks=ASKS, reference="100", fee="2"):
    return StaticExecution(parent(side, quantity, reference, fee), bids, asks)


@pytest.mark.parametrize(
    "side,first_notional,second_notional,first_fee,second_fee,first_cost,second_cost,total,fees,cost,average,bps,residual",
    [
        ("BUY", "4000.6", "4001.9", "0.80012", "0.80038", "1.40012", "2.70038", "8002.5", "1.6005", "4.1005", "100.03125", "5.125625", "100.10"),
        ("SELL", "3999.4", "3998.1", "0.79988", "0.79962", "1.39988", "2.69962", "7997.5", "1.5995", "4.0995", "99.96875", "5.124375", "99.90"),
    ],
)
def test_two_children_share_depth_and_match_one_large_execution_with_exact_fees(
    side, first_notional, second_notional, first_fee, second_fee, first_cost, second_cost,
    total, fees, cost, average, bps, residual,
):
    # Each 40-unit child sweeps a different part of the original 120-unit book.
    split = execution(side)
    assert split.state.version == 0 and split.state.children == ()
    submitted = split.submit("first", D("40"))
    assert submitted == split.state and submitted.version == 1
    assert submitted.remaining_qty == D("80")
    assert submitted.reserved_qty == submitted.available_qty == D("40")
    assert submitted.filled_qty == submitted.notional == submitted.fees == submitted.covered_cost == D("0")
    assert submitted.children[0].status == "reserved"
    assert submitted.children[0].unfilled_qty is None
    assert submitted.bids == BIDS and submitted.asks == ASKS
    second_reserved = split.submit("second", D("40"))
    assert second_reserved.remaining_qty == second_reserved.reserved_qty == D("80")
    assert second_reserved.available_qty == D("0")
    after_first = split.execute("first")
    assert after_first.filled_qty == after_first.remaining_qty == after_first.reserved_qty == D("40")
    assert after_first.available_qty == D("0")
    assert after_first.full_cost_bps is None
    final = split.execute("second")
    assert final.version == 4
    assert final.filled_qty == D("80")
    assert final.remaining_qty == final.reserved_qty == final.available_qty == D("0")
    assert final.notional == D(total) and final.fees == D(fees) and final.covered_cost == D(cost)
    assert final.average_price == D(average) and final.full_cost_bps == D(bps)
    for child, notional, fee, cash in zip(
        final.children, (first_notional, second_notional), (first_fee, second_fee), (first_cost, second_cost),
    ):
        assert child.status == "filled" and child.requested_qty == child.filled_qty == D("40")
        assert child.unfilled_qty == D("0")
        assert child.notional == D(notional) and child.fee_amount == D(fee) and child.covered_cost == D(cash)
    relevant = ASKS if side == "BUY" else BIDS
    assert final.children[0].fills == ((relevant[0][0], D("30")), (relevant[1][0], D("10")))
    assert final.children[1].fills == ((relevant[1][0], D("30")), (relevant[2][0], D("10")))
    assert (final.asks if side == "BUY" else final.bids) == ((D(residual), D("40")),)
    assert (final.bids if side == "BUY" else final.asks) == (BIDS if side == "BUY" else ASKS)

    single = execution(side)
    single.submit("all", D("80"))
    one = single.execute("all")
    for field in ("filled_qty", "remaining_qty", "reserved_qty", "available_qty", "notional", "fees", "covered_cost", "average_price", "full_cost_bps", "bids", "asks"):
        assert getattr(one, field) == getattr(final, field)


def test_reserving_parent_quantity_does_not_reserve_prices_or_change_quote():
    ledger = execution()
    original = ledger.quote(D("40"))
    ledger.submit("first-submitted", D("40"))
    ledger.submit("second-submitted", D("40"))
    assert ledger.quote(D("40")) == original
    result = ledger.execute("second-submitted")
    assert [c.child_id for c in result.children] == ["first-submitted", "second-submitted"]
    assert result.children[0].status == "reserved"
    assert result.children[1].notional == D("4000.6")
    assert result.reserved_qty == result.remaining_qty == D("40")
    assert ledger.quote(D("40")).fill.notional == D("4001.9")
    final = ledger.execute("first-submitted")
    assert final.children[0].notional == D("4001.9")
    assert final.covered_cost == D("4.1005")


@pytest.mark.parametrize(
    "side,limit,first_notional,first_fee,first_cost,total,fees,cost,residual",
    [
        ("BUY", "100.03", "7001.5", "1.4003", "2.9003", "10004.5", "2.0009", "6.5009", "100.10"),
        ("SELL", "99.97", "6998.5", "1.3997", "2.8997", "9995.5", "1.9991", "6.4991", "99.90"),
    ],
)
def test_inclusive_limit_allows_partial_then_releases_unfilled_quantity_for_retry(
    side, limit, first_notional, first_fee, first_cost, total, fees, cost, residual,
):
    ledger = execution(side, "100")
    ledger.submit("limited", D("100"), limit_price=D(limit))
    partial = ledger.execute("limited")
    child = partial.children[0]
    assert child.status == "partial" and child.limit_price == D(limit)
    assert child.requested_qty == D("100") and child.filled_qty == D("70") and child.unfilled_qty == D("30")
    assert child.notional == D(first_notional) and child.fee_amount == D(first_fee) and child.covered_cost == D(first_cost)
    assert len(child.fills) == 2 and child.fills[-1][0] == D(limit)
    assert partial.remaining_qty == partial.available_qty == D("30") and partial.reserved_qty == D("0")
    assert partial.full_cost_bps is None
    assert (partial.asks if side == "BUY" else partial.bids) == ((D(residual), D("50")),)
    ledger.submit("retry", D("30"))
    complete = ledger.execute("retry")
    assert complete.filled_qty == D("100") and complete.remaining_qty == D("0")
    assert complete.notional == D(total) and complete.fees == D(fees) and complete.covered_cost == D(cost)
    assert complete.full_cost_bps == D(cost)  # q * reference = 10,000.
    assert complete.children[0] == child


@pytest.mark.parametrize("side,limit", [("BUY", "100"), ("SELL", "100")])
def test_nonmarketable_limit_is_terminal_unfilled_and_preserves_book(side, limit):
    ledger = execution(side)
    ledger.submit("blocked", D("80"), limit_price=D(limit))
    state = ledger.execute("blocked")
    child = state.children[0]
    assert child.status == "unfilled" and child.filled_qty == D("0") and child.unfilled_qty == D("80")
    assert child.fills == ()
    assert child.notional == child.fee_amount == child.covered_cost == D("0")
    assert state.remaining_qty == state.available_qty == D("80") and state.reserved_qty == D("0")
    assert state.filled_qty == state.notional == state.fees == state.covered_cost == D("0")
    assert state.average_price is state.full_cost_bps is None
    assert state.bids == BIDS and state.asks == ASKS


def test_exhausted_depth_keeps_parent_remainder_and_retry_cannot_reuse_consumed_levels():
    ledger = execution(quantity="150")
    ledger.submit("consume", D("150"))
    partial = ledger.execute("consume")
    assert partial.filled_qty == D("120") and partial.remaining_qty == partial.available_qty == D("30")
    assert partial.notional == D("12006.5") and partial.fees == D("2.4013") and partial.covered_cost == D("8.9013")
    assert partial.asks == () and partial.bids == BIDS and partial.full_cost_bps is None
    ledger.submit("retry", D("30"))
    result = ledger.execute("retry")
    assert result.children[-1].status == "unfilled" and result.children[-1].unfilled_qty == D("30")
    assert result.children[-1].fills == ()
    for field in ("filled_qty", "remaining_qty", "available_qty", "notional", "fees", "covered_cost", "average_price"):
        assert getattr(result, field) == getattr(partial, field)
    assert result.full_cost_bps is None


def test_cancel_releases_only_that_reservation_and_retains_terminal_child():
    ledger = execution()
    ledger.submit("cancel-me", D("30"))
    ledger.submit("keep-me", D("40"))
    cancelled = ledger.cancel("cancel-me")
    child = cancelled.children[0]
    assert cancelled.version == 3 and child.status == "cancelled"
    assert child.filled_qty == D("0") and child.unfilled_qty == D("30") and child.fills == ()
    assert child.notional == child.fee_amount == child.covered_cost == D("0")
    assert cancelled.remaining_qty == D("80") and cancelled.reserved_qty == cancelled.available_qty == D("40")
    assert cancelled.bids == BIDS and cancelled.asks == ASKS
    assert cancelled.notional == cancelled.fees == cancelled.covered_cost == D("0")
    ledger.submit("replacement", D("40"))
    ledger.execute("keep-me")
    result = ledger.execute("replacement")
    assert result.filled_qty == D("80") and result.covered_cost == D("4.1005")
    assert result.children[0] == child


def test_read_only_quote_uses_residual_depth_and_limit_without_parent_capacity_constraint():
    ledger = execution(quantity="40")
    before = ledger.state
    quote = ledger.quote(D("80"), limit_price=D("100.03"))
    assert quote.fill.requested_qty == D("80") and quote.fill.filled_qty == D("70")
    assert quote.fill.unfilled_qty == D("10") and quote.full_cost_bps is None
    assert quote.covered_cost == D("2.9003")
    assert ledger.state == before
    ledger.submit("finish", D("40"))
    ledger.execute("finish")
    complete = ledger.state
    assert complete.remaining_qty == D("0")
    assert ledger.quote(D("80")).fill.notional == D("8005.9")
    assert ledger.quote(D("0")).fill.filled_qty == D("0")
    assert ledger.quote(D("0")).full_cost_bps is None
    assert ledger.state == complete


def test_zero_parent_has_no_full_cost_and_cannot_submit_but_can_quote_book():
    ledger = execution(quantity="0")
    state = ledger.state
    assert state.filled_qty == state.remaining_qty == state.reserved_qty == state.available_qty == D("0")
    assert state.notional == state.fees == state.covered_cost == D("0")
    assert state.average_price is state.full_cost_bps is None
    assert ledger.quote(D("40")).fill.notional == D("4000.6")
    with pytest.raises(ValueError):
        ledger.submit("not-allowed", D("1"))
    assert ledger.state == state


def test_empty_book_has_known_zero_fills_and_can_release_and_resubmit_quantity():
    ledger = execution(bids=(), asks=())
    ledger.submit("empty", D("20"))
    result = ledger.execute("empty")
    assert result.children[0].status == "unfilled"
    assert result.children[0].unfilled_qty == D("20")
    assert result.filled_qty == D("0") and result.remaining_qty == result.available_qty == D("80")
    assert result.average_price is result.full_cost_bps is None
    assert ledger.quote(D("1")).fill.unfilled_qty == D("1")
    assert ledger.submit("again", D("80")).reserved_qty == D("80")


def test_input_and_state_snapshots_are_deeply_immutable_and_books_are_private():
    bids, asks = [list(level) for level in BIDS], [list(level) for level in ASKS]
    original = deepcopy((bids, asks))
    ledger = StaticExecution(parent(), bids, asks)
    other = StaticExecution(parent(), bids, asks)
    initial = ledger.state
    assert (bids, asks) == original
    bids[0][1] = D("999")
    asks.clear()
    assert initial.bids == BIDS and initial.asks == ASKS
    ledger.submit("one", D("40"))
    reserved = ledger.state
    ledger.execute("one")
    assert initial.children == () and initial.version == 0
    assert reserved.children[0].status == "reserved" and reserved.reserved_qty == D("40")
    assert other.state == initial
    assert ledger.state.asks != initial.asks
    for value, field, replacement in [(initial, "filled_qty", D("9")), (initial.parent, "quantity", D("9")), (ledger.state.children[0], "status", "reserved")]:
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, replacement)


@pytest.mark.parametrize(
    "changes",
    [
        {"parent_id": " "}, {"parent_id": 1}, {"side": "HOLD"},
        {"quantity": D("-1")}, {"quantity": 1.0},
        {"reference_price": D("0")}, {"fee_bps": D("NaN")}, {"fee_bps": D("-1")},
    ],
)
def test_invalid_parent_contract_is_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        StaticExecution(replace(parent(), **changes), BIDS, ASKS)


@pytest.mark.parametrize(
    "bids,asks",
    [
        (BIDS, tuple(reversed(ASKS))),
        (tuple(reversed(BIDS)), ASKS),
        (BIDS, ((D("100"), D("1")), (D("100"), D("2")))),
        (BIDS, ((D("99.99"), D("1")),)),
        (BIDS, ((D("99.98"), D("1")),)),
        (BIDS, ((D("101"), D("0")),)),
        (((D("99"), D("-1")),), ASKS),
        (BIDS, ((101.0, D("1")),)),
        (BIDS, None),
    ],
)
def test_constructor_validates_both_raw_book_sides_even_for_zero_parent(bids, asks):
    with pytest.raises((ValueError, TypeError)):
        StaticExecution(parent(quantity="0"), bids, asks)


@pytest.mark.parametrize(
    "name,quantity,limit",
    [
        ("bad", D("0"), None), ("bad", D("-1"), None), ("bad", D("NaN"), None),
        ("bad", 1.0, None), ("", D("1"), None), ("  ", D("1"), None),
        ("bad", D("1"), D("0")), ("bad", D("1"), D("Infinity")),
        ("bad", D("1"), "100"),
    ],
)
def test_bad_submission_is_atomic_and_does_not_claim_identifier(name, quantity, limit):
    ledger = execution()
    before = ledger.state
    with pytest.raises((ValueError, TypeError)):
        ledger.submit(name, quantity, limit_price=limit)
    assert ledger.state == before
    assert ledger.submit("bad", D("1")).children[0].status == "reserved"


def test_parent_availability_includes_pending_children_and_failed_attempt_can_be_reused():
    ledger = execution()
    ledger.submit("pending", D("50"))
    reserved = ledger.state
    with pytest.raises(ValueError):
        ledger.submit("too-large", D("31"))
    assert ledger.state == reserved
    ledger.cancel("pending")
    assert ledger.submit("too-large", D("80")).reserved_qty == D("80")


@pytest.mark.parametrize("terminal", ["filled", "partial", "unfilled", "cancelled"])
def test_terminal_children_cannot_execute_cancel_or_reuse_identifier(terminal):
    ledger = execution()
    limit = D("100.01") if terminal == "partial" else D("100") if terminal == "unfilled" else None
    ledger.submit("one", D("40"), limit_price=limit)
    if terminal == "cancelled":
        ledger.cancel("one")
    else:
        ledger.execute("one")
    assert ledger.state.children[0].status == terminal
    before = ledger.state
    for action in (lambda: ledger.execute("one"), lambda: ledger.cancel("one"), lambda: ledger.submit("one", D("1"))):
        with pytest.raises(ValueError):
            action()
        assert ledger.state == before


def test_unknown_child_and_duplicate_pending_identifier_fail_without_mutation():
    ledger = execution()
    ledger.submit("pending", D("40"))
    before = ledger.state
    for action in (lambda: ledger.execute("missing"), lambda: ledger.cancel("missing"), lambda: ledger.submit("pending", D("1"))):
        with pytest.raises(ValueError):
            action()
        assert ledger.state == before


@pytest.mark.parametrize("quantity,limit", [(D("-1"), None), (1.0, None), (D("1"), D("0"))])
def test_bad_quote_is_atomic(quantity, limit):
    ledger = execution()
    ledger.submit("pending", D("40"))
    before = ledger.state
    with pytest.raises((ValueError, TypeError)):
        ledger.quote(quantity, limit_price=limit)
    assert ledger.state == before


def test_exact_cash_is_accumulated_before_rounded_average_or_bps():
    asks = ((D("100.0000000000000000000000000001"), D("1")), (D("100.0000000000000000000000000002"), D("1")))
    ledger = execution(quantity="2", fee="0", bids=(), asks=asks)
    ledger.submit("first", D("1"))
    first = ledger.execute("first")
    assert first.covered_cost == D("1e-28")
    assert first.average_price == D("100")
    ledger.submit("second", D("1"))
    final = ledger.execute("second")
    assert final.notional == D("200.0000000000000000000000000003")
    assert final.covered_cost == D("3e-28") and final.fees == D("0")
    assert final.average_price == D("100") and final.full_cost_bps == D("1.5e-26")


def test_costs_can_be_favorable_and_signed_child_costs_cancel_without_losing_amounts():
    ledger = execution(quantity="2", fee="0", bids=(), asks=((D("99"), D("1")), (D("101"), D("1"))))
    ledger.submit("favorable", D("1"))
    assert ledger.execute("favorable").covered_cost == D("-1")
    ledger.submit("adverse", D("1"))
    final = ledger.execute("adverse")
    assert final.children[0].covered_cost == D("-1") and final.children[1].covered_cost == D("1")
    assert final.notional == D("200") and final.covered_cost == final.full_cost_bps == D("0")


def test_book_consumption_reservations_fees_and_ratios_ignore_ambient_decimal_context():
    with localcontext() as context:
        context.prec = 3
        context.rounding = ROUND_UP
        context.Emax = 2
        context.Emin = -2
        context.traps[Inexact] = context.traps[Rounded] = True
        context.clear_flags()
        before = (context.prec, context.rounding, context.Emin, context.Emax, dict(context.traps), dict(context.flags))
        ledger = execution()
        quote = ledger.quote(D("80"))
        ledger.submit("first", D("40"))
        ledger.submit("second", D("40"))
        ledger.execute("first")
        final = ledger.execute("second")
        assert (context.prec, context.rounding, context.Emin, context.Emax, dict(context.traps), dict(context.flags)) == before
    assert quote.fill.notional == final.notional == D("8002.5")
    assert final.fees == D("1.6005") and final.covered_cost == D("4.1005")
    assert final.average_price == D("100.03125") and final.full_cost_bps == D("5.125625")
    assert final.remaining_qty == final.reserved_qty == final.available_qty == D("0")


def test_unsupported_reservation_precision_fails_atomically_instead_of_rounding_quantity():
    ledger = execution()
    before = ledger.state
    with pytest.raises(ValueError):
        ledger.submit("tiny", D("1e-4000"))
    assert ledger.state == before
    assert ledger.submit("tiny", D("1")).reserved_qty == D("1")


def test_execution_accounting_failure_keeps_depth_and_reservation_and_still_allows_cancel():
    # Initial values and quote fit their limits; fee multiplication introduces
    # enough decimal places that the resulting execution ledger cannot fit.
    ledger = execution(quantity="1", reference="1", fee="1e-1300", bids=(), asks=((D("1e-1300"), D("1")),))
    assert ledger.quote(D("1")).fee_amount == D("1e-2604")
    ledger.submit("precision-limit", D("1"))
    before = ledger.state
    with pytest.raises(ValueError, match="precision"):
        ledger.execute("precision-limit")
    assert ledger.state == before
    assert ledger.state.asks == ((D("1e-1300"), D("1")),)
    cancelled = ledger.cancel("precision-limit")
    assert cancelled.children[0].status == "cancelled"
    assert cancelled.available_qty == D("1") and cancelled.reserved_qty == D("0")
