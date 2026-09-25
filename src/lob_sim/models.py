"""Versioned synthetic contracts. Times share one known nanosecond origin."""
from dataclasses import dataclass
from decimal import Decimal

from lob_lab.book import BookState, LevelUpdate
from lob_lab.costs import Level
from lob_lab.execution import ChildOrder, ExecutionState, ParentOrder
from lob_lab.policy import LiquidityRule, PolicyDecision, PolicyView
from lob_lab.timed_execution import ExecutionMetrics, StrategySpec


@dataclass(frozen=True, slots=True)
class MarketMessage:
    state_time_ns: int
    event_index: int
    sequence: int
    kind: str = "update"
    bids: tuple[Level, ...] = ()
    asks: tuple[Level, ...] = ()
    updates: tuple[LevelUpdate, ...] = ()
    session_id: str = "synthetic-1"
    full_depth: bool = True


@dataclass(frozen=True, slots=True)
class FeedDelivery:
    event_index: int
    receive_time_ns: int


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    max_buffer: int = 10000
    timeout_ns: int = 1_000_000_000
    max_requests: int = 3
    # None disables the synthetic on-demand snapshot service (push only).
    snapshot_read_latency_ns: int | None = None
    snapshot_response_latency_ns: int = 0


@dataclass(frozen=True, slots=True)
class CausalConfig:
    start_time_ns: int
    deadline_ns: int
    market_start_ns: int
    market_end_ns: int
    order_latency_ns: int = 0
    response_latency_ns: int = 0
    liquidity_model: str = "persistent_debit"
    limit_price: Decimal | None = None
    max_receive_age_ns: int | None = None
    max_state_age_ns: int | None = None
    session_id: str = "synthetic-1"
    recovery: RecoveryConfig = RecoveryConfig()


@dataclass(frozen=True, slots=True)
class RecoveryView:
    book: BookState
    last_receive_time_ns: int | None
    state_time_ns: int | None
    high_water_sequence: int
    buffered_sequences: tuple[int, ...]
    recovering_since_ns: int | None
    publication: int
    reason: str | None
    # Origin indices of messages actually used in the currently published book.
    provenance: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RecoveryOutcome:
    published: bool
    reason: str | None
    applied: tuple[MarketMessage, ...] = ()


@dataclass(frozen=True, slots=True)
class OrderRecord:
    child_id: str
    decision_time_ns: int
    arrival_time_ns: int
    response_time_ns: int | None = None
    status: str = "pending"
    outcome: ChildOrder | None = None  # None on unknown execution, never fake zero.
    execution_sequence: int | None = None
    received_time_ns: int | None = None  # Actual delivery; response_time is scheduled.


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    decision: PolicyDecision
    view: PolicyView
    local: RecoveryView


@dataclass(frozen=True, slots=True)
class CausalTrace:
    time_ns: int
    event_type: str
    detail: str | None
    market: BookState
    local: RecoveryView
    known: ExecutionState
    actual: ExecutionState
    execution_known: bool


@dataclass(frozen=True, slots=True)
class CausalResult:
    contract_version: str
    parent: ParentOrder
    config: CausalConfig
    strategy: StrategySpec
    liquidity_rule: LiquidityRule | None
    orders: tuple[OrderRecord, ...]
    decisions: tuple[DecisionRecord, ...]
    trace: tuple[CausalTrace, ...]
    actual_state: ExecutionState
    known_state: ExecutionState
    terminal_market: BookState
    metrics: ExecutionMetrics | None
    execution_known: bool
    status: str
    reason: str | None
