"""Independent monetary examples for covered cost; partial orders are not full IS."""

from dataclasses import FrozenInstanceError
from decimal import MAX_EMAX, Decimal, Inexact, ROUND_UP, Rounded, localcontext

import pytest

from lob_lab.valuation import CostEstimate, estimate_cost


D = Decimal
BIDS = ((D("99.99"), D("4")), (D("99.97"), D("6")))
ASKS = ((D("100.01"), D("2")), (D("100.03"), D("3")), (D("100.08"), D("5")))


@pytest.mark.parametrize(
    "side,quantity,fee,fee_amount,cash,bps,complete",
    [
        ("BUY", "4", "0", "0", "0.08", "2", True),
        ("BUY", "4", "2", "0.080016", "0.160016", "4.0004", True),
        ("BUY", "4", "10", "0.40008", "0.48008", "12.002", True),
        ("SELL", "5", "10", "0.49993", "0.56993", "11.3986", True),
        ("BUY", "12", "10", "1.00051", "1.51051", "15.1051", False),
        ("SELL", "12", "10", "0.99978", "1.21978", "12.1978", False),
    ],
)
def test_hard_coded_buy_sell_and_partial_costs(side, quantity, fee, fee_amount, cash, bps, complete):
    result = estimate_cost(side, D(quantity), BIDS, ASKS, reference_price=D("100"), fee_bps=D(fee))
    assert isinstance(result, CostEstimate)
    assert result.reference_price == D("100")
    assert result.fee_bps == D(fee)
    assert result.fee_amount == D(fee_amount)
    assert result.covered_cost == D(cash)
    assert result.covered_cost_bps == D(bps)
    assert result.full_cost_bps == (D(bps) if complete else None)
    assert result.fill.requested_qty == D(quantity)
    assert result.fill.depth_sufficient is complete
    if not complete:
        assert result.fill.filled_qty == D("10")
        assert result.fill.unfilled_qty == D("2")


@pytest.mark.parametrize(
    "side,bids,asks",
    [
        ("BUY", ((D("98"), D("2")),), ((D("99"), D("2")),)),
        ("SELL", ((D("101"), D("2")),), ((D("102"), D("2")),)),
    ],
)
def test_favorable_price_is_negative_cost_for_both_directions(side, bids, asks):
    result = estimate_cost(side, D("1"), bids, asks, reference_price=D("100"))
    assert result.covered_cost == D("-1")
    assert result.covered_cost_bps == D("-100")
    assert result.full_cost_bps == D("-100")
    assert result.fee_bps == result.fee_amount == D("0")


@pytest.mark.parametrize(
    "side,price,fee,cash,bps",
    [("BUY", "99", "0.099", "-0.901", "-90.1"), ("SELL", "101", "0.101", "-0.899", "-89.9")],
)
def test_fee_remains_positive_even_when_price_is_favorable(side, price, fee, cash, bps):
    levels = ((D(price), D("1")),)
    result = estimate_cost(
        side, D("1"), levels if side == "SELL" else (), levels if side == "BUY" else (),
        reference_price=D("100"), fee_bps=D("10"),
    )
    assert result.fee_amount == D(fee)
    assert result.covered_cost == D(cash)
    assert result.full_cost_bps == D(bps)


@pytest.mark.parametrize("side", ["BUY", "SELL"])
@pytest.mark.parametrize("quantity,bids,asks", [(D("0"), BIDS, ASKS), (D("3"), (), ())])
def test_no_coverage_has_zero_cash_but_no_percentage_cost(side, quantity, bids, asks):
    result = estimate_cost(side, quantity, bids, asks, reference_price=D("100"), fee_bps=D("10"))
    assert result.fill.filled_qty == D("0")
    assert result.fee_amount == result.covered_cost == D("0")
    assert result.covered_cost_bps is None
    assert result.full_cost_bps is None


@pytest.mark.parametrize("field", ["reference_price", "fee_bps"])
@pytest.mark.parametrize("value", [100, 1.0, "100", None, True])
def test_reference_and_fee_types_are_strict_even_for_zero_demand(field, value):
    values = {"reference_price": D("100"), "fee_bps": D("0")}
    values[field] = value
    with pytest.raises(TypeError):
        estimate_cost("BUY", D("0"), (), (), **values)


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "sNaN", "Infinity", "-Infinity"])
def test_reference_price_must_be_positive_finite(value):
    with pytest.raises(ValueError):
        estimate_cost("BUY", D("0"), (), (), reference_price=D(value))


@pytest.mark.parametrize("value", ["-1", "NaN", "sNaN", "Infinity", "-Infinity"])
def test_fee_must_be_nonnegative_finite(value):
    with pytest.raises(ValueError):
        estimate_cost("BUY", D("0"), (), (), reference_price=D("100"), fee_bps=D(value))


@pytest.mark.parametrize(
    "side,quantity,bids,asks,exception",
    [
        ("HOLD", D("1"), BIDS, ASKS, ValueError),
        ("BUY", 1.0, BIDS, ASKS, TypeError),
        ("SELL", D("-1"), BIDS, ASKS, ValueError),
        ("BUY", D("1"), BIDS, tuple(reversed(ASKS)), ValueError),
    ],
)
def test_cost_layer_preserves_fill_validation(side, quantity, bids, asks, exception):
    with pytest.raises(exception):
        estimate_cost(side, quantity, bids, asks, reference_price=D("100"))


def test_fee_and_cash_are_exact_beyond_average_reporting_precision():
    price = D("100.0000000000000000000000000001")
    result = estimate_cost(
        "BUY", D("1"), (), ((price, D("1")),),
        reference_price=D("100"), fee_bps=D("0.0000000000000000000000000001"),
    )
    assert result.fee_amount == D("1.000000000000000000000000000001E-30")
    assert result.covered_cost == D("1.01000000000000000000000000000001E-28")


def test_valuation_is_independent_of_external_decimal_context_and_its_flags():
    with localcontext() as context:
        context.prec = 4
        context.rounding = ROUND_UP
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        context.clear_flags()
        before = (context.prec, context.rounding, dict(context.traps), dict(context.flags))
        result = estimate_cost(
            "BUY", D("1"), (), ((D("1"), D("1")),), reference_price=D("3")
        )
        assert (context.prec, context.rounding, dict(context.traps), dict(context.flags)) == before
    assert result.covered_cost == D("-2")
    assert result.covered_cost_bps == D("-6666.666666666666666666666667")


def test_unsupported_reference_scale_is_rejected_instead_of_losing_cash_precision():
    with pytest.raises(ValueError):
        estimate_cost("BUY", D("1"), BIDS, ASKS, reference_price=D(f"1e{MAX_EMAX}"))


def test_result_is_frozen_and_inputs_are_not_consumed():
    bids, asks = list(BIDS), list(ASKS)
    original = (bids.copy(), asks.copy())
    first = estimate_cost("BUY", D("4"), bids, asks, reference_price=D("100"))
    second = estimate_cost("BUY", D("4"), bids, asks, reference_price=D("100"))
    assert first == second
    assert (bids, asks) == original
    with pytest.raises(FrozenInstanceError):
        first.covered_cost = D("0")
