# LOB execution simulator

A synthetic limit-order-book execution simulator for a single parent order. It
separates market state, received market data, and received order reports, with
fixed latencies and explicit cost and completion accounting.

The examples show active immediate-or-cancel (IOC) execution, quantity reservations,
partial fills, and recovery from missing feed updates. BUY and SELL orders, price
protection, fees, and deterministic event ordering share the same execution path.

## Local reproduction

These commands are for the non-production evaluation permitted by the
[Portfolio Evaluation License](LICENSE).

Use Python 3.12 or newer with `venv` and `pip`. From a fresh checkout, these POSIX
shell commands build and install a wheel, then run the examples outside the source
directory. The runtime uses only the Python standard library; installation fetches
the declared build and test tools from the package index.

Local candidate validation used CPython 3.12.14 and 3.14.5. The minimum required
version is Python 3.12; these local checks do not establish support for every
later Python version.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements-dev.txt
.venv/bin/python -m pip wheel --no-cache-dir --no-deps --wheel-dir dist .
.venv/bin/python -m pip install --no-cache-dir --no-deps dist/lob_execution_simulator-*.whl

SIM_ROOT=$(pwd -P)
SIM_PY="$SIM_ROOT/.venv/bin/python"
SIM_RUN_DIR=$(mktemp -d)
cd "$SIM_RUN_DIR"
"$SIM_PY" -I -m lob_sim.demo --case delayed-report --output delayed-report.json
"$SIM_PY" -I -m lob_sim.demo --case gap-recovery --output gap-recovery.json
```

Start with an empty `dist` directory so the wheel pattern selects exactly one
file. Omit `--output` to print JSON. Existing output files are never overwritten.
The installed `lob-execution-demo` command provides the same options. Neither
example downloads data or uses network services.

## Two synthetic examples

These are prescribed mechanism examples, not observations of a market. Both use a
reference price of 100, a zero fee rate, and the `persistent_debit` depth rule.

| Example | Mechanism | Expected result |
|---|---|---|
| `delayed-report` | Two cumulative TWAP decisions with 5 ms order latency and 3 ms report latency | Buy 4 of 12 units at 101; 8 remain; implementation shortfall is 4 cash units |
| `gap-recovery` | A snapshot joins buffered updates after a sequence gap; a deadline catch-up order follows | Recover through sequence 106; buy all 4 units at 101; implementation shortfall is 4 cash units, or 100 bps |

The delayed-report example makes the information boundary visible:

| Time | Event |
|---|---|
| 0 ms | Submit 6 units and reserve that parent quantity |
| 5 ms | The order fills 4 units in the market |
| 6 ms | The next decision still knows 0 fills and sees 6 units reserved |
| 8 ms | The first terminal report releases its reservation and records the 4 fills |
| 11 ms | The second child order arrives; executable ask depth is exhausted |
| 14 ms | Its zero-fill report is received after the 12 ms deadline |

JSON contains the synthetic inputs, decisions and their visible state, orders,
event trace, actual and known execution states, and metrics. Decimal values are
strings; times are integer nanoseconds. Missing values stay `null`. A completed
simulation does not imply a fully completed parent order.

## Python interface and design

The recommended entry point is:

```python
from lob_sim.engine import run_causal_execution

result = run_causal_execution(
    messages, deliveries, parent, config, strategy, liquidity_rule=None
)
```

`MarketMessage`, `FeedDelivery`, and `CausalConfig` are in `lob_sim.models`;
`ParentOrder` is in `lob_lab.execution`; `StrategySpec` is in
`lob_lab.timed_execution`. [The demo](src/lob_sim/demo.py) constructs complete
inputs. [Design notes](docs/design.md) explain state separation, event order,
recovery, cost formulas, and the two depth assumptions.

Implementation shortfall includes the cost of filled quantity and, when a valid
terminal midpoint is available, the opportunity cost of unfilled quantity. Always
read it alongside completion. An unknown execution or an unavailable terminal
valuation is preserved rather than converted to a successful zero-cost result.

## Scope and limitations

The model assumes one instrument, one session, complete depth, absolute quantities,
continuous integer sequence numbers, and aligned synthetic clocks. It has no live
exchange adapter, passive queue model, cancellation race, multi-parent scheduler,
or calibrated market-impact model. Public depth is externally prescribed; own
fills do not change future feed messages. Recovery restores local market data,
not a crashed process or an unknown execution history.

The two depth rules are modeling assumptions, not measured bounds on real fills.
Decimal arithmetic has supported precision limits, and ratios use explicit
rounding. Performance and production suitability are not claimed.

## Tests

After installation, while still in the temporary run directory:

```sh
"$SIM_PY" -I -m pytest --import-mode=importlib -q "$SIM_ROOT/tests" \
  --basetemp="$SIM_RUN_DIR/pytest" -o "cache_dir=$SIM_RUN_DIR/pytest-cache"
```

Tests cover independent quantity and cost expectations, information isolation,
report timing, deadline boundaries, price protection, numeric validation, recovery,
and installed-package behavior. They use synthetic inputs and no external data.

## Copyright and development

Copyright (c) 2026 Shuyang Zeng. Copyright is retained.
Source code is provided for viewing and non-production technical evaluation.
See [LICENSE](LICENSE) for permitted use and restrictions, and
[third-party notices](THIRD_PARTY_NOTICES.md) for applicable third-party terms.
AI coding tools assisted implementation, documentation, and review.
