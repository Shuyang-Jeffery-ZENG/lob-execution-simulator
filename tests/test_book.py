"""Independent state trajectories for the synthetic absolute-size L2 contract."""

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from lob_lab.book import BookError, BookNotReadyError, BookState, LevelUpdate, OrderBook


D = Decimal
BIDS = ((D("99.99"), D("4")), (D("99.97"), D("6")))
ASKS = (
    (D("100.01"), D("2")),
    (D("100.03"), D("3")),
    (D("100.08"), D("5")),
)


def ready_book():
    book = OrderBook()
    book.apply_snapshot(sequence=100, bids=BIDS, asks=ASKS)
    return book


def assert_last_accepted_data_preserved(before, after):
    """A rejected message changes validity/recovery metadata, never accepted content."""
    assert after.bids == before.bids
    assert after.asks == before.asks
    assert after.sequence == before.sequence
    assert after.version == before.version
    assert after.is_valid is False
    assert isinstance(after.invalid_reason, str) and after.invalid_reason


def test_new_book_requires_a_snapshot_before_quotes_or_updates():
    book = OrderBook()
    initial = book.state
    assert isinstance(initial, BookState)
    assert initial.bids == initial.asks == ()
    assert initial.sequence is None
    assert initial.version == 0
    assert initial.is_valid is False
    assert initial.required_snapshot_sequence == 0
    assert issubclass(BookNotReadyError, BookError)
    with pytest.raises(BookNotReadyError):
        book.estimate_fill("BUY", D("1"))
    with pytest.raises(BookError):
        book.apply_update(sequence=5, updates=[])
    assert_last_accepted_data_preserved(initial, book.state)
    assert book.state.required_snapshot_sequence == 5
    with pytest.raises(BookError):
        book.apply_snapshot(sequence=4, bids=BIDS, asks=ASKS)
    recovered = book.apply_snapshot(sequence=5, bids=BIDS, asks=ASKS)
    assert recovered.is_valid is True
    assert recovered.sequence == 5
    assert recovered.version == 1
    assert recovered.required_snapshot_sequence == 6


def test_snapshot_sorts_both_sides_and_publishes_one_accepted_version():
    book = OrderBook()
    state = book.apply_snapshot(
        sequence=100, bids=tuple(reversed(BIDS)), asks=(ASKS[2], ASKS[0], ASKS[1])
    )
    assert state == book.state
    assert state.bids == BIDS
    assert state.asks == ASKS
    assert state.sequence == 100
    assert state.version == 1
    assert state.is_valid is True
    assert state.invalid_reason is None
    assert state.required_snapshot_sequence == 101
    assert book.estimate_fill("BUY", D("4")).average_price == D("100.02")
    assert book.estimate_fill("SELL", D("5")).average_price == D("99.986")


def test_absolute_replacement_and_zero_deletion_change_costs_as_expected():
    book = ready_book()
    replaced = book.apply_update(
        sequence=101, updates=[LevelUpdate("ASK", D("100.01"), D("3"))]
    )
    # An absolute 3 replaces the old 2; treating it as +3 would leave five at best ask.
    assert replaced.asks == ((D("100.01"), D("3")), ASKS[1], ASKS[2])
    fill = book.estimate_fill("BUY", D("4"))
    assert fill.notional == D("400.06")
    assert fill.average_price == D("100.015")
    assert fill.levels_swept == 2

    deleted = book.apply_update(
        sequence=102, updates=[LevelUpdate("ASK", D("100.01"), D("0"))]
    )
    assert deleted.asks == ASKS[1:]
    assert deleted.bids == BIDS
    fill = book.estimate_fill("BUY", D("4"))
    assert fill.notional == D("400.17")
    assert fill.average_price == D("100.0425")
    assert deleted.version == 3
    assert deleted.sequence == 102
    assert deleted.required_snapshot_sequence == 103


def test_bid_replacement_insertion_and_deletion_have_sell_side_meaning():
    book = ready_book()
    state = book.apply_update(
        sequence=101,
        updates=[
            LevelUpdate("BID", D("99.99"), D("1")),
            LevelUpdate("BID", D("99.98"), D("2")),
            LevelUpdate("BID", D("99.97"), D("0")),
        ],
    )
    assert state.bids == ((D("99.99"), D("1")), (D("99.98"), D("2")))
    fill = book.estimate_fill("SELL", D("4"))
    assert fill.filled_qty == D("3")
    assert fill.unfilled_qty == D("1")
    assert fill.notional == D("299.95")
    assert fill.average_price == D("99.98333333333333333333333333")
    assert fill.depth_sufficient is False


def test_deleting_absent_level_and_empty_batch_still_advance_accepted_sequence():
    book = ready_book()
    removed = book.apply_update(
        sequence=101, updates=[LevelUpdate("ASK", D("101"), D("0"))]
    )
    assert removed.bids == BIDS and removed.asks == ASKS
    assert removed.sequence == 101 and removed.version == 2
    empty = book.apply_update(sequence=102, updates=[])
    assert empty.bids == BIDS and empty.asks == ASKS
    assert empty.sequence == 102 and empty.version == 3
    assert empty.required_snapshot_sequence == 103


def test_new_snapshot_replaces_all_old_levels_instead_of_merging():
    book = ready_book()
    state = book.apply_snapshot(
        sequence=120, bids=[(D("90"), D("1"))], asks=[(D("91"), D("2"))]
    )
    assert state.bids == ((D("90"), D("1")),)
    assert state.asks == ((D("91"), D("2")),)
    assert state.sequence == 120
    assert state.version == 2
    assert state.required_snapshot_sequence == 121
    assert book.estimate_fill("BUY", D("3")).unfilled_qty == D("1")


def test_atomic_batch_may_cross_temporarily_if_its_final_book_is_valid():
    book = OrderBook()
    before = book.apply_snapshot(
        sequence=0, bids=[(D("99"), D("2"))], asks=[(D("101"), D("2"))]
    )
    after = book.apply_update(
        sequence=1,
        updates=[
            LevelUpdate("BID", D("102"), D("3")),
            LevelUpdate("ASK", D("101"), D("0")),
            LevelUpdate("ASK", D("103"), D("2")),
        ],
    )
    assert after.bids == ((D("102"), D("3")), (D("99"), D("2")))
    assert after.asks == ((D("103"), D("2")),)
    assert after.version == before.version + 1
    assert after.is_valid is True
    assert before.bids == ((D("99"), D("2")),)
    assert before.asks == ((D("101"), D("2")),)


@pytest.mark.parametrize("new_bid", ["100.01", "100.02"])
def test_final_locked_or_crossed_batch_rolls_back_every_level(new_bid):
    book = ready_book()
    before = book.state
    with pytest.raises(BookError):
        book.apply_update(
            sequence=101,
            updates=[
                LevelUpdate("ASK", D("100.08"), D("7")),
                LevelUpdate("BID", D(new_bid), D("1")),
            ],
        )
    assert_last_accepted_data_preserved(before, book.state)
    with pytest.raises(BookNotReadyError):
        book.estimate_fill("BUY", D("1"))


@pytest.mark.parametrize("side", ["BID", "ASK"])
def test_duplicate_side_price_updates_are_rejected_not_last_write_wins(side):
    book = ready_book()
    before = book.state
    price = D("99.99") if side == "BID" else D("100.01")
    with pytest.raises(BookError):
        book.apply_update(
            sequence=101,
            updates=[LevelUpdate(side, price, D("0")), LevelUpdate(side, price, D("8"))],
        )
    assert_last_accepted_data_preserved(before, book.state)


def test_same_price_on_different_sides_is_not_an_update_duplicate():
    # A bid deletion and ask insertion may share a price without sharing a key.
    book = OrderBook()
    book.apply_snapshot(sequence=0, bids=[], asks=[(D("101"), D("2"))])
    state = book.apply_update(
        sequence=1,
        updates=[LevelUpdate("BID", D("100"), D("0")), LevelUpdate("ASK", D("100"), D("1"))],
    )
    assert state.asks == ((D("100"), D("1")), (D("101"), D("2")))
    assert state.bids == ()


@pytest.mark.parametrize(
    "bad_update",
    [
        LevelUpdate("BUY", D("100"), D("1")),
        LevelUpdate("ask", D("101"), D("1")),
        LevelUpdate("ASK", "101", D("1")),
        LevelUpdate("ASK", D("101"), 1),
        LevelUpdate("ASK", D("0"), D("1")),
        LevelUpdate("ASK", D("101"), D("-1")),
        LevelUpdate("ASK", D("NaN"), D("1")),
        LevelUpdate("ASK", D("101"), D("sNaN")),
        LevelUpdate("ASK", D("Infinity"), D("1")),
        LevelUpdate("ASK", D("101"), D("Infinity")),
        None,
        ("ASK", D("101"), D("1")),
    ],
)
def test_bad_update_after_a_valid_change_is_fully_rolled_back_and_invalidates(bad_update):
    book = ready_book()
    before = book.state
    with pytest.raises(BookError):
        book.apply_update(
            sequence=101,
            updates=[LevelUpdate("BID", D("99.99"), D("9")), bad_update],
        )
    assert_last_accepted_data_preserved(before, book.state)
    assert book.state.required_snapshot_sequence == 101


@pytest.mark.parametrize(
    "bids,asks",
    [
        ([(D("99"), D("0"))], ASKS),
        (BIDS, [(D("101"), D("-1"))]),
        ([(D("NaN"), D("1"))], ASKS),
        (BIDS, [(D("101"), D("Infinity"))]),
        ([(99, D("1"))], ASKS),
        (BIDS, [(D("101"), 1.0)]),
        (BIDS, [(D("101"),)]),
        (BIDS + ((D("99.990"), D("1")),), ASKS),
        (BIDS, ASKS + ((D("100.010"), D("1")),)),
        ([(D("100.01"), D("1"))], ASKS),
        ([(D("102"), D("1"))], ASKS),
    ],
)
def test_bad_snapshot_cannot_partially_replace_the_accepted_book(bids, asks):
    book = ready_book()
    before = book.state
    with pytest.raises(BookError):
        book.apply_snapshot(sequence=110, bids=bids, asks=asks)
    assert_last_accepted_data_preserved(before, book.state)
    assert book.state.required_snapshot_sequence == 110


@pytest.mark.parametrize("sequence", [-1, True, False, 101.0, "101", None, D("101")])
@pytest.mark.parametrize("kind", ["snapshot", "update"])
def test_invalid_sequence_types_do_not_advance_recovery_floor(sequence, kind):
    book = ready_book()
    before = book.state
    with pytest.raises(BookError):
        if kind == "snapshot":
            book.apply_snapshot(sequence=sequence, bids=BIDS, asks=ASKS)
        else:
            book.apply_update(sequence=sequence, updates=[])
    assert_last_accepted_data_preserved(before, book.state)
    assert book.state.required_snapshot_sequence == 101


@pytest.mark.parametrize("sequence", [99, 100, 102])
def test_reverse_duplicate_and_gap_updates_all_require_a_fresh_snapshot(sequence):
    book = ready_book()
    before = book.state
    with pytest.raises(BookError):
        book.apply_update(sequence=sequence, updates=[])
    assert_last_accepted_data_preserved(before, book.state)
    assert book.state.required_snapshot_sequence == max(101, sequence)
    with pytest.raises(BookError):
        book.apply_update(sequence=101, updates=[])
    assert_last_accepted_data_preserved(before, book.state)


def test_recovery_snapshot_cannot_precede_a_gap_or_later_observed_invalid_update():
    book = ready_book()
    before = book.state
    with pytest.raises(BookError):
        book.apply_update(sequence=105, updates=[])
    assert book.state.required_snapshot_sequence == 105
    with pytest.raises(BookError):
        book.apply_snapshot(sequence=102, bids=BIDS, asks=ASKS)
    assert book.state.required_snapshot_sequence == 105
    with pytest.raises(BookError):
        book.apply_update(sequence=110, updates=[])
    assert book.state.required_snapshot_sequence == 110
    with pytest.raises(BookError):
        book.apply_snapshot(sequence=109, bids=BIDS, asks=ASKS)
    assert_last_accepted_data_preserved(before, book.state)
    recovered = book.apply_snapshot(sequence=110, bids=BIDS, asks=ASKS)
    assert recovered.is_valid is True
    assert recovered.invalid_reason is None
    assert recovered.sequence == 110
    assert recovered.version == 2
    assert recovered.required_snapshot_sequence == 111
    assert book.apply_update(sequence=111, updates=[]).version == 3


@pytest.mark.parametrize("sequence", [99, 100])
def test_stale_snapshot_is_rejected_even_if_its_prices_are_valid(sequence):
    book = ready_book()
    before = book.state
    with pytest.raises(BookError):
        book.apply_snapshot(sequence=sequence, bids=[], asks=[])
    assert_last_accepted_data_preserved(before, book.state)
    assert book.state.required_snapshot_sequence == 101
    assert book.apply_snapshot(sequence=101, bids=[], asks=[]).is_valid is True


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_valid_empty_book_is_distinct_from_an_invalid_book(side):
    book = OrderBook()
    state = book.apply_snapshot(sequence=0, bids=[], asks=[])
    assert state.is_valid is True
    assert state.invalid_reason is None
    quote = book.estimate_fill(side, D("2"))
    assert quote.filled_qty == D("0")
    assert quote.unfilled_qty == D("2")
    assert quote.average_price is None
    assert quote.depth_sufficient is False


def test_deleting_the_last_level_keeps_a_valid_empty_side():
    book = OrderBook()
    book.apply_snapshot(sequence=0, bids=[], asks=[(D("101"), D("2"))])
    state = book.apply_update(
        sequence=1, updates=[LevelUpdate("ASK", D("101"), D("0"))]
    )
    assert state.asks == ()
    assert state.is_valid is True
    assert book.estimate_fill("BUY", D("2")).filled_qty == D("0")


@pytest.mark.parametrize(
    "side,quantity,exception",
    [("HOLD", D("1"), ValueError), ("BUY", D("-1"), ValueError), ("SELL", 1.0, TypeError)],
)
def test_bad_query_is_not_a_bad_market_message(side, quantity, exception):
    book = ready_book()
    before = book.state
    with pytest.raises(exception):
        book.estimate_fill(side, quantity)
    assert book.state == before
    assert book.estimate_fill("BUY", D("4")).average_price == D("100.02")


def test_input_containers_old_states_and_queries_share_no_mutable_book_storage():
    bids, asks = list(BIDS), list(ASKS)
    book = OrderBook()
    before = book.apply_snapshot(sequence=100, bids=bids, asks=asks)
    bids.clear()
    asks[0] = (D("105"), D("999"))
    assert book.state.bids == BIDS and book.state.asks == ASKS
    changes = [LevelUpdate("ASK", D("100.01"), D("3"))]
    after = book.apply_update(sequence=101, updates=changes)
    changes.clear()
    assert before.asks == ASKS and before.version == 1
    assert after.asks[0] == (D("100.01"), D("3"))
    assert after.version == 2
    first = book.estimate_fill("BUY", D("4"))
    book.estimate_fill("SELL", D("100"))
    assert book.estimate_fill("BUY", D("4")) == first
    assert book.state == after
    with pytest.raises(AttributeError):
        book.state = before
    with pytest.raises(FrozenInstanceError):
        after.is_valid = False
    with pytest.raises(TypeError):
        after.asks[0] = (D("110"), D("1"))
    update = LevelUpdate("ASK", D("101"), D("2"))
    with pytest.raises(FrozenInstanceError):
        update.quantity = D("3")


def test_rejected_initial_snapshot_preserves_an_empty_history_and_sets_recovery_floor():
    book = OrderBook()
    before = book.state
    with pytest.raises(BookError):
        book.apply_snapshot(sequence=7, bids=BIDS, asks=[(D("100"), D("0"))])
    assert_last_accepted_data_preserved(before, book.state)
    assert book.state.required_snapshot_sequence == 7
    with pytest.raises(BookNotReadyError):
        book.estimate_fill("BUY", D("0"))
    recovered = book.apply_snapshot(sequence=7, bids=BIDS, asks=ASKS)
    assert recovered.version == 1
    assert recovered.required_snapshot_sequence == 8


@pytest.mark.parametrize("container", [None, {}, "not-levels", 1])
@pytest.mark.parametrize("kind", ["snapshot", "update"])
def test_malformed_message_container_invalidates_without_corrupting_depth(container, kind):
    book = ready_book()
    before = book.state
    with pytest.raises(BookError):
        if kind == "snapshot":
            book.apply_snapshot(sequence=101, bids=BIDS, asks=container)
        else:
            book.apply_update(sequence=101, updates=container)
    assert_last_accepted_data_preserved(before, book.state)
    assert book.state.required_snapshot_sequence == 101


def test_sorting_a_snapshot_does_not_reorder_the_callers_input_arrays():
    bids, asks = list(reversed(BIDS)), [ASKS[2], ASKS[0], ASKS[1]]
    original_bids, original_asks = bids.copy(), asks.copy()
    book = OrderBook()
    book.apply_snapshot(sequence=0, bids=bids, asks=asks)
    assert bids == original_bids
    assert asks == original_asks
    assert book.state.bids == BIDS
    assert book.state.asks == ASKS


def test_restarted_source_sequence_requires_a_new_book_instance():
    old_session = ready_book()
    with pytest.raises(BookError):
        old_session.apply_snapshot(sequence=0, bids=BIDS, asks=ASKS)
    assert old_session.state.sequence == 100
    new_session = OrderBook()
    state = new_session.apply_snapshot(sequence=0, bids=BIDS, asks=ASKS)
    assert state.is_valid is True
    assert state.sequence == 0
    assert state.version == 1
    assert state.required_snapshot_sequence == 1
    assert old_session.state.is_valid is False
