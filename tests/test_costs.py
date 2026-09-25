"""Independent examples and contract checks for a read-only visible-book query."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from decimal import MAX_EMAX, MIN_EMIN, Decimal, Inexact, ROUND_UP, Rounded, localcontext
from fractions import Fraction

import pytest

from lob_lab.costs import FillEstimate, estimate_fill


D = Decimal
BIDS = ((D("99.99"), D("4")), (D("99.97"), D("6")))
ASKS = (
    (D("100.01"), D("2")),
    (D("100.03"), D("3")),
    (D("100.08"), D("5")),
)


@pytest.mark.parametrize(
    "side,quantity,filled,remaining,notional,average,worst,levels,ratio,sufficient",
    [
        ("BUY", "4", "4", "0", "400.08", "100.02", "100.03", 2, "1", True),
        (
            "BUY", "7", "7", "0", "700.27",
            "100.0385714285714285714285714", "100.08", 3, "1", True,
        ),
        (
            "BUY", "12", "10", "2", "1000.51", "100.051", "100.08", 3,
            "0.8333333333333333333333333333", False,
        ),
        ("SELL", "5", "5", "0", "499.93", "99.986", "99.97", 2, "1", True),
        (
            "SELL", "12", "10", "2", "999.78", "99.978", "99.97", 2,
            "0.8333333333333333333333333333", False,
        ),
    ],
)
def test_independent_business_examples(
    side, quantity, filled, remaining, notional, average, worst, levels, ratio, sufficient
):
    # Values come from the separately calculated examples, not a second sweep loop.
    expected = FillEstimate(
        requested_qty=D(quantity),
        filled_qty=D(filled),
        unfilled_qty=D(remaining),
        notional=D(notional),
        average_price=D(average),
        worst_price=D(worst),
        levels_swept=levels,
        fill_ratio=D(ratio),
        depth_sufficient=sufficient,
    )
    result = estimate_fill(side, D(quantity), BIDS, ASKS)
    assert result == expected
    assert result.filled_qty + result.unfilled_qty == result.requested_qty


@pytest.mark.parametrize(
    "side,quantity,notional,average,worst,levels",
    [
        ("BUY", "1", "100.01", "100.01", "100.01", 1),
        ("BUY", "2", "200.02", "100.01", "100.01", 1),
        ("BUY", "5", "500.11", "100.022", "100.03", 2),
        ("SELL", "4", "399.96", "99.99", "99.99", 1),
        ("SELL", "10", "999.78", "99.978", "99.97", 2),
    ],
)
def test_single_level_and_exact_level_boundaries(
    side, quantity, notional, average, worst, levels
):
    result = estimate_fill(side, D(quantity), BIDS, ASKS)
    assert result.filled_qty == D(quantity)
    assert result.unfilled_qty == D("0")
    assert result.notional == D(notional)
    assert result.average_price == D(average)
    assert result.worst_price == D(worst)
    assert result.levels_swept == levels
    assert result.fill_ratio == D("1")
    assert result.depth_sufficient is True


def test_fractional_quantities_use_the_actual_amount_at_each_level():
    asks = ((D("100.125"), D("0.125")), (D("100.25"), D("0.375")))
    result = estimate_fill("BUY", D("0.3"), (), asks)
    assert result.filled_qty == D("0.3")
    assert result.unfilled_qty == D("0")
    assert result.notional == D("30.059375")
    assert result.average_price == D("100.1979166666666666666666667")
    assert result.worst_price == D("100.25")
    assert result.levels_swept == 2


@pytest.mark.parametrize("side", ["BUY", "SELL"])
@pytest.mark.parametrize("empty_both", [False, True])
def test_empty_contra_side_preserves_the_whole_requested_quantity(side, empty_both):
    bids = () if side == "SELL" or empty_both else BIDS
    asks = () if side == "BUY" or empty_both else ASKS
    result = estimate_fill(side, D("3"), bids, asks)
    assert result.requested_qty == D("3")
    assert result.filled_qty == result.notional == D("0")
    assert result.unfilled_qty == D("3")
    assert result.average_price is None
    assert result.worst_price is None
    assert result.levels_swept == 0
    assert result.fill_ratio == D("0")
    assert result.depth_sufficient is False


@pytest.mark.parametrize("side", ["BUY", "SELL"])
@pytest.mark.parametrize("bids,asks", [(BIDS, ASKS), ((), ())])
def test_zero_quantity_is_a_defined_noop(side, bids, asks):
    result = estimate_fill(side, D("0"), bids, asks)
    assert result.requested_qty == result.filled_qty == result.unfilled_qty == D("0")
    assert result.notional == D("0")
    assert result.average_price is None
    assert result.worst_price is None
    assert result.levels_swept == 0
    assert result.fill_ratio is None
    assert result.depth_sufficient is True


@pytest.mark.parametrize("quantity", [1, 1.0, "1", None, True, Fraction(1, 1)])
def test_requested_quantity_requires_decimal(quantity):
    with pytest.raises(TypeError):
        estimate_fill("BUY", quantity, BIDS, ASKS)


@pytest.mark.parametrize("value", [1, 1.0, "1", None, True])
@pytest.mark.parametrize("component", ["price", "size"])
@pytest.mark.parametrize("book_side", ["bids", "asks"])
def test_each_book_numeric_field_requires_decimal(value, component, book_side):
    bids, asks = list(BIDS), list(ASKS)
    book = bids if book_side == "bids" else asks
    price, size = book[-1]
    book[-1] = (value, size) if component == "price" else (price, value)
    # Small BUY does not consume these deep prices or any bids: both sides still validate.
    with pytest.raises(TypeError):
        estimate_fill("BUY", D("0.1"), bids, asks)


@pytest.mark.parametrize("quantity", ["-1", "NaN", "sNaN", "Infinity", "-Infinity"])
def test_invalid_decimal_quantity_is_rejected(quantity):
    with pytest.raises(ValueError):
        estimate_fill("BUY", D(quantity), BIDS, ASKS)


@pytest.mark.parametrize("side", ["buy", "sell", "HOLD", "", None, 1])
@pytest.mark.parametrize("quantity", [D("0"), D("1")])
def test_invalid_side_is_rejected_even_for_zero_quantity(side, quantity):
    with pytest.raises(ValueError):
        estimate_fill(side, quantity, BIDS, ASKS)


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "sNaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("component", ["price", "size"])
@pytest.mark.parametrize("book_side", ["bids", "asks"])
def test_invalid_book_values_are_rejected_before_zero_quantity_return(
    value, component, book_side
):
    bids, asks = list(BIDS), list(ASKS)
    book = bids if book_side == "bids" else asks
    price, size = book[-1]
    book[-1] = (D(value), size) if component == "price" else (price, D(value))
    with pytest.raises(ValueError):
        estimate_fill("BUY", D("0"), bids, asks)


@pytest.mark.parametrize(
    "bids,asks",
    [
        (tuple(reversed(BIDS)), ASKS),
        (BIDS, tuple(reversed(ASKS))),
        (BIDS + ((D("99.97"), D("1")),), ASKS),
        (BIDS, ASKS + ((D("100.08"), D("1")),)),
    ],
    ids=["bids-ascending", "asks-descending", "duplicate-bid", "duplicate-ask"],
)
@pytest.mark.parametrize("quantity", [D("0"), D("0.1")])
def test_unsorted_and_duplicate_levels_are_not_silently_repaired(bids, asks, quantity):
    with pytest.raises(ValueError):
        estimate_fill("BUY", quantity, bids, asks)


@pytest.mark.parametrize("bid_price", [D("100.01"), D("100.02")])
@pytest.mark.parametrize("quantity", [D("0"), D("1")])
def test_locked_and_crossed_synthetic_books_are_rejected(bid_price, quantity):
    with pytest.raises(ValueError):
        estimate_fill("BUY", quantity, ((bid_price, D("4")),), ASKS)


def test_queries_are_repeatable_and_do_not_consume_or_reorder_input():
    bids, asks = list(BIDS), list(ASKS)
    before = deepcopy((bids, asks))
    first = estimate_fill("BUY", D("4"), bids, asks)
    estimate_fill("SELL", D("12"), bids, asks)
    estimate_fill("BUY", D("100"), bids, asks)
    assert estimate_fill("BUY", D("4"), bids, asks) == first
    assert (bids, asks) == before


def test_estimate_is_an_immutable_value():
    result = estimate_fill("BUY", D("4"), BIDS, ASKS)
    assert isinstance(result, FillEstimate)
    with pytest.raises(FrozenInstanceError):
        result.filled_qty = D("0")


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_sweeping_more_quantity_cannot_improve_static_average_price(side):
    results = [estimate_fill(side, D(q), BIDS, ASKS) for q in ("1", "2", "4", "5", "7", "10", "12")]
    for previous, result in zip(results, results[1:]):
        assert result.filled_qty >= previous.filled_qty
        assert result.notional >= previous.notional
        assert result.levels_swept >= previous.levels_swept
        if side == "BUY":
            assert result.average_price >= previous.average_price
        else:
            assert result.average_price <= previous.average_price
    assert results[-1].filled_qty == D("10")
    assert results[-1].unfilled_qty == D("2")
    assert results[-1].notional == results[-2].notional


def test_high_precision_amount_and_quantity_are_not_rounded_to_display_precision():
    price = D("100.0000000000000000000000000001")
    size = D("3.0000000000000000000000000001")
    result = estimate_fill("BUY", size, (), ((price, size),))
    assert result.filled_qty == size
    assert result.unfilled_qty == D("0")
    assert result.notional == D(
        "300.00000000000000000000000001030000000000000000000000000001"
    )


def test_external_decimal_context_does_not_change_results_or_receive_flags():
    # Callers may run unrelated low-precision work or enable traps. Query semantics stay fixed.
    with localcontext() as context:
        context.prec = 4
        context.rounding = ROUND_UP
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        context.clear_flags()
        context_before = (context.prec, context.rounding, dict(context.traps), dict(context.flags))
        full = estimate_fill("BUY", D("7"), BIDS, ASKS)
        partial = estimate_fill("BUY", D("12"), BIDS, ASKS)
        assert (context.prec, context.rounding, dict(context.traps), dict(context.flags)) == context_before
    assert full.notional == D("700.27")
    assert full.average_price == D("100.0385714285714285714285714")
    assert partial.fill_ratio == D("0.8333333333333333333333333333")


@pytest.mark.parametrize(
    "price,expected",
    [
        ("1.0000000000000000000000000005", "1.000000000000000000000000000"),
        ("1.0000000000000000000000000015", "1.000000000000000000000000002"),
    ],
)
def test_average_rounds_ties_to_even_at_28_significant_digits(price, expected):
    result = estimate_fill("BUY", D("1"), (), ((D(price), D("1")),))
    assert result.average_price == D(expected)
    # The exact cash amount is separate from the rounded reporting average.
    assert result.notional == D(price)


def test_extreme_scale_spread_is_rejected_instead_of_silently_rounding_money():
    with pytest.raises(ValueError, match="precision"):
        estimate_fill("BUY", D("1"), (), ((D("1e-5000"), D("1")),))


@pytest.mark.parametrize("exponent", [MAX_EMAX, MIN_EMIN], ids=["overflow", "underflow"])
def test_unrepresentable_exact_amount_is_reported_as_a_value_error(exponent):
    # Equal exponents have a narrow scale spread but their product exceeds Decimal's range.
    value = D(f"1e{exponent}")
    with pytest.raises(ValueError):
        estimate_fill("BUY", value, (), ((value, value),))
