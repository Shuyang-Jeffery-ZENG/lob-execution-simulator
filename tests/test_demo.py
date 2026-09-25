"""Independent arithmetic and CLI checks for the installed synthetic examples."""
import json
import subprocess
import sys

import pytest


def invoke(cwd, *args):
    return subprocess.run(
        [sys.executable, "-I", "-m", "lob_sim.demo", *map(str, args)],
        cwd=cwd, capture_output=True, text=True, timeout=30,
    )


@pytest.fixture(scope="module")
def report():
    from lob_sim.demo import build_report
    return build_report()


def test_delayed_report_keeps_reservation_until_receipt_and_settles_after_deadline(report):
    runs = {row["case"]: row["result"] for row in report["cases"]}
    assert set(runs) == {"delayed-report", "gap-recovery"}
    result = runs["delayed-report"]
    assert result["execution_known"] is True and result["status"] == "complete"
    assert result["reason"] is None
    first, second = result["orders"]
    assert [order["decision_time_ns"] for order in (first, second)] == [0, 6_000_000]
    assert [order["arrival_time_ns"] for order in (first, second)] == [5_000_000, 11_000_000]
    assert [order["response_time_ns"] for order in (first, second)] == [8_000_000, 14_000_000]
    assert [order["received_time_ns"] for order in (first, second)] == [8_000_000, 14_000_000]
    assert [order["outcome"]["requested_qty"] for order in (first, second)] == ["6", "6"]
    assert first["outcome"]["fills"] == [["101", "4"]]
    assert first["outcome"]["unfilled_qty"] == "2"
    assert second["outcome"]["fills"] == []
    assert second["outcome"]["unfilled_qty"] == "6"
    view = result["decisions"][1]["view"]
    assert view["time_ns"] == 6_000_000
    assert (view["filled_qty"], view["reserved_qty"], view["available_qty"]) == ("0", "6", "6")
    assert view["asks"] == [["101", "4"]]
    for state in (result["actual_state"], result["known_state"]):
        assert (state["filled_qty"], state["remaining_qty"], state["reserved_qty"]) == ("4", "8", "0")
        assert state["notional"] == "404" and state["fees"] == "0"
        assert state["full_cost_bps"] is None
    metrics = result["metrics"]
    assert metrics["covered_cost"] == metrics["is_cash"] == "4"
    assert metrics["terminal_mid"] == "100" and metrics["opportunity_cost"] == "0"
    assert metrics["completion_ratio"] == "0.3333333333333333333333333333"
    assert metrics["is_bps"] == "33.33333333333333333333333333"
    assert metrics["evaluation_available"] is True
    assert max(row["time_ns"] for row in result["trace"]) == 14_000_000


def test_recovery_publishes_contiguous_state_before_catchup_execution(report):
    result = next(row["result"] for row in report["cases"] if row["case"] == "gap-recovery")
    publications = [row for row in result["trace"] if row["event_type"] == "snapshot_published"]
    assert len(publications) == 1
    publication = publications[0]
    assert publication["time_ns"] == 16_000_000
    assert publication["local"]["book"]["is_valid"] is True
    assert publication["local"]["book"]["sequence"] == 106
    assert publication["local"]["book"]["asks"] == [["101", "4"]]
    assert publication["local"]["buffered_sequences"] == []
    assert [decision["view"]["time_ns"] for decision in result["decisions"]] == [10_000_000, 20_000_000]
    assert [decision["decision"]["selected_qty"] for decision in result["decisions"]] == ["0", "4"]
    assert result["decisions"][0]["local"]["book"]["is_valid"] is False
    assert len(result["orders"]) == 1
    order = result["orders"][0]
    assert order["arrival_time_ns"] == order["received_time_ns"] == 20_000_000
    assert order["outcome"]["fills"] == [["101", "4"]]
    assert result["actual_state"]["notional"] == "404"
    assert result["known_state"]["remaining_qty"] == result["known_state"]["reserved_qty"] == "0"
    metrics = result["metrics"]
    assert metrics["filled_qty"] == "4" and metrics["fees"] == "0"
    assert metrics["covered_cost"] == metrics["is_cash"] == "4"
    assert metrics["opportunity_cost"] == "0"
    assert metrics["is_bps"] == "100" and metrics["completion_ratio"] == "1"


@pytest.mark.parametrize("case", ["delayed-report", "gap-recovery"])
def test_cli_selects_one_case_from_an_independent_directory(tmp_path, report, case):
    run = invoke(tmp_path, "--case", case)
    assert run.returncode == 0, run.stderr
    selected = json.loads(run.stdout)
    assert selected == {**report, "cases": [row for row in report["cases"] if row["case"] == case]}
    assert run.stderr == "" and list(tmp_path.iterdir()) == []


def test_default_cli_is_repeatable_and_matches_both_examples(tmp_path, report):
    first = invoke(tmp_path)
    second = invoke(tmp_path)
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert first.stdout == second.stdout
    assert json.loads(first.stdout) == report
    assert report["contract"] == "causal-replay-v1" and report["time_unit"] == "ns"


def test_cli_writes_a_new_file_and_preserves_existing_content(tmp_path, report):
    path = tmp_path / "result.json"
    run = invoke(tmp_path, "--output", path)
    assert run.returncode == 0, run.stderr
    assert run.stdout == "" and json.loads(path.read_text(encoding="utf-8")) == report
    original = path.read_bytes()
    repeated = invoke(tmp_path, "--output", path)
    assert repeated.returncode == 2
    assert "cannot create output" in repeated.stderr
    assert path.read_bytes() == original


def test_cli_rejects_unknown_case_without_creating_output(tmp_path):
    path = tmp_path / "result.json"
    run = invoke(tmp_path, "--case", "unknown", "--output", path)
    assert run.returncode == 2
    assert "invalid choice" in run.stderr
    assert run.stdout == "" and not path.exists()
