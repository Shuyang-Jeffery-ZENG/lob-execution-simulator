"""Run two synthetic execution examples: python -m lob_sim.demo --case NAME."""
import argparse
from dataclasses import asdict, is_dataclass, replace
from decimal import Decimal
import json
from pathlib import Path

from lob_lab.book import LevelUpdate
from lob_lab.execution import ParentOrder
from lob_lab.timed_execution import StrategySpec
from .engine import run_causal_execution
from .models import CausalConfig, FeedDelivery, MarketMessage, RecoveryConfig

D = Decimal
MS = 1_000_000
CASE_NAMES = ("delayed-report", "gap-recovery")


def snapshot(time, index, sequence, bid="99", ask="101", quantity="10"):
    return MarketMessage(time * MS, index, sequence, "snapshot",
                         ((D(bid), D("20")),), ((D(ask), D(quantity)),))


def update(time, index, sequence, *changes):
    return MarketMessage(time * MS, index, sequence, updates=tuple(
        LevelUpdate(side, D(price), D(quantity)) for side, price, quantity in changes))


def cases():
    """Inputs are prespecified mechanism examples, not market observations."""
    parent = ParentOrder("demo", "BUY", D("4"), D("100"))
    strategy = StrategySpec("immediate", 1, False)
    base = CausalConfig(0, 12 * MS, 0, 20 * MS, order_latency_ns=5 * MS)
    result = []

    def add(name, messages, times, *, p=parent, c=base, s=strategy):
        deliveries = tuple(FeedDelivery(index, time * MS) for index, time in times)
        result.append((name, tuple(messages), deliveries, p, c, s))

    add("delayed-report", [snapshot(0, 0, 100, quantity="4")], [(0, 0)],
        p=replace(parent, quantity=D("12")), c=replace(base, response_latency_ns=3*MS),
        s=StrategySpec("cumulative_twap", 2, False))
    recovery_messages = [snapshot(8, 0, 100)] + [
        update(time, sequence-100, sequence, ("ASK", "101", str(110-sequence)))
        for time, sequence in ((9, 101), (10, 102), (12, 103), (13, 104), (14, 105), (15, 106))]
    recovery_config = replace(base, market_start_ns=8*MS, start_time_ns=10*MS,
        deadline_ns=20*MS, order_latency_ns=0,
        recovery=RecoveryConfig(timeout_ns=10*MS, snapshot_read_latency_ns=2*MS,
                                snapshot_response_latency_ns=4*MS))
    add("gap-recovery", recovery_messages,
        [(0, 8), (2, 10), (4, 13), (5, 14), (6, 15)],
        c=recovery_config, s=StrategySpec("immediate", 1, True))
    return tuple(result)


def serializable(value):
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def build_report(case="all"):
    if case not in (*CASE_NAMES, "all"):
        raise ValueError(f"unknown case: {case}")
    records = []
    for name, messages, deliveries, parent, config, strategy in cases():
        if case != "all" and name != case:
            continue
        run = run_causal_execution(messages, deliveries, parent, config, strategy)
        records.append({"case": name, "inputs": {"messages": messages, "deliveries": deliveries,
                       "parent": parent, "config": config, "strategy": strategy}, "result": run})
    return serializable({"contract": "causal-replay-v1", "time_unit": "ns",
                         "evidence": "synthetic correctness examples; no market performance inference",
                         "cases": records})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*CASE_NAMES, "all"), default="all",
                        help="synthetic example to run (default: all)")
    parser.add_argument("--output", type=Path, help="new JSON file (existing files are never overwritten)")
    args = parser.parse_args(argv)
    payload = json.dumps(build_report(args.case), ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        except OSError as exc:
            parser.exit(2, f"cannot create output: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
