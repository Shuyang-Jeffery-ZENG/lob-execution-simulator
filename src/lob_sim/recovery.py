"""Local, atomic reconstruction of the normalized single-session feed.

The incremental buffer is bounded; one isolated snapshot anchor may also be
retained. Audit fingerprints (and the current book's provenance) retain message
identity across publications, so duplicate delivery never refreshes a clock.
This is not an exchange-specific snapshot or retransmission client.
"""

from dataclasses import replace
from hashlib import sha256

from lob_lab.book import BookError, BookState, OrderBook

from .models import MarketMessage, RecoveryConfig, RecoveryOutcome, RecoveryView


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _number_key(value):
    """Context-independent numeric identity, including equivalent decimals."""
    sign, digits, exponent = value.as_tuple()
    if not any(digits):
        return (0, (0,), 0)
    digits = list(digits)
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    return (sign, tuple(digits), exponent)


def _depth_key(message):
    return tuple(tuple((_number_key(p), _number_key(q)) for p, q in side)
                 for side in (message.bids, message.asks))


def _fingerprint(message, *, depth_only=False):
    key = _depth_key(message)
    if not depth_only:
        key = (message.kind, message.sequence, message.state_time_ns,
               message.session_id, message.full_depth, key,
               tuple((u.side, _number_key(u.price), _number_key(u.quantity))
                     for u in message.updates))
    return sha256(repr(key).encode("utf-8")).digest()


class LocalRecovery:
    """Publish only a complete candidate reaching the observed high water.

    A missing update may arrive late: a contiguous chain from the last accepted
    book can then repair a gap without a snapshot. Content faults, overflow and
    timeout instead require a new full snapshot. Old published states never
    change; publication versions and observed sequence high water never fall.
    """

    def __init__(self, session_id: str, config: RecoveryConfig):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a nonempty string")
        if not isinstance(config, RecoveryConfig):
            raise TypeError("config must be RecoveryConfig")
        for field in ("max_buffer", "timeout_ns", "max_requests",
                      "snapshot_response_latency_ns"):
            _integer(getattr(config, field), field)
        if config.max_buffer == 0 or config.timeout_ns == 0:
            raise ValueError("max_buffer and timeout_ns must be positive")
        if config.snapshot_read_latency_ns is not None:
            _integer(config.snapshot_read_latency_ns, "snapshot_read_latency_ns")
        self._session = session_id
        self._config = config
        self._book = BookState()
        self._accepted = None
        self._state_time = None
        self._last_receive = None
        self._last_clock = None
        self._high_water = 0
        self._publication = 0
        self._provenance = ()
        self._since = None
        self._reason = "snapshot required"
        self._updates = {}
        self._anchor = None
        self._needs_snapshot = True
        # Compact identity history is separate from buffered message payloads.
        self._seen_events = {}
        self._seen_updates = {}
        self._seen_snapshots = {}

    @property
    def view(self) -> RecoveryView:
        return RecoveryView(
            self._book, self._last_receive, self._state_time, self._high_water,
            tuple(sorted(self._updates)), self._since, self._publication,
            self._reason, self._provenance,
        )

    def _clock(self, time_ns):
        _integer(time_ns, "time_ns")
        if self._last_clock is not None and time_ns < self._last_clock:
            raise ValueError("local receive/check times must not move backwards")

    def _invalidate(self, reason, time_ns, *, reset=False):
        if reset:
            self._updates.clear()
            self._anchor = None
            self._needs_snapshot = True
        if self._since is None:
            self._since = time_ns
        self._reason = reason
        self._book = replace(
            self._book, is_valid=False, invalid_reason=reason,
            required_snapshot_sequence=self._high_water,
        )
        return RecoveryOutcome(False, reason)

    def _validate_content(self, message):
        if message.session_id != self._session:
            raise BookError("message belongs to another source session")
        if type(message.full_depth) is not bool or not message.full_depth:
            raise BookError("full-depth message required")
        if message.kind not in ("snapshot", "update"):
            raise BookError("unknown message kind")
        if any(not isinstance(getattr(message, field), tuple)
               for field in ("bids", "asks", "updates")):
            raise BookError("message containers must be immutable tuples")
        if any(not isinstance(level, tuple) for level in message.bids + message.asks):
            raise BookError("snapshot levels must be immutable tuples")
        candidate = OrderBook()
        if message.kind == "snapshot":
            if message.updates:
                raise BookError("snapshot cannot also contain updates")
            state = candidate.apply_snapshot(
                sequence=message.sequence, bids=message.bids, asks=message.asks)
            return replace(message, bids=state.bids, asks=state.asks)
        if message.bids or message.asks:
            raise BookError("update cannot also contain snapshot levels")
        # Validate the batch independently; the complete candidate is checked
        # again against its actual predecessor before anything is published.
        candidate.apply_snapshot(sequence=0, bids=(), asks=())
        candidate.apply_update(sequence=1, updates=message.updates)
        return replace(message, updates=tuple(sorted(
            message.updates, key=lambda update: (update.side, update.price))))

    def receive(self, message: MarketMessage, receive_time_ns: int) -> RecoveryOutcome:
        self._clock(receive_time_ns)
        if not isinstance(message, MarketMessage):
            raise TypeError("message must be MarketMessage")
        for field in ("state_time_ns", "event_index", "sequence"):
            _integer(getattr(message, field), field)
        if receive_time_ns < message.state_time_ns:
            raise ValueError("message cannot arrive before its state exists")
        self._last_clock = receive_time_ns
        # Foreign sessions must not poison this session's recovery target.
        if message.session_id == self._session:
            self._high_water = max(self._high_water, message.sequence)
        try:
            message = self._validate_content(message)
        except BookError as exc:
            return self._invalidate(str(exc), receive_time_ns, reset=True)

        fingerprint = _fingerprint(message)
        previous = self._seen_events.get(message.event_index)
        if previous is not None:
            if previous != fingerprint:
                return self._invalidate("conflicting duplicate event index",
                                        receive_time_ns, reset=True)
            return RecoveryOutcome(False, "duplicate delivery")
        self._seen_events[message.event_index] = fingerprint

        if message.kind == "update":
            previous = self._seen_updates.get(message.sequence)
            if previous is not None:
                if previous != fingerprint:
                    return self._invalidate("conflicting duplicate update sequence",
                                            receive_time_ns, reset=True)
                return RecoveryOutcome(False, "duplicate update")
            self._seen_updates[message.sequence] = fingerprint
            if self._accepted is not None and message.sequence <= self._accepted.sequence:
                # A later full snapshot already covers this old delta. It is
                # diagnosed but never applied and never refreshes either age.
                return RecoveryOutcome(False, "update covered by published state")
            self._updates[message.sequence] = message
        else:
            depth_fingerprint = _fingerprint(message, depth_only=True)
            previous = self._seen_snapshots.get(message.sequence)
            if previous is not None and previous != depth_fingerprint:
                return self._invalidate("conflicting snapshot for the same sequence",
                                        receive_time_ns, reset=True)
            self._seen_snapshots[message.sequence] = depth_fingerprint
            if self._accepted is not None:
                if message.sequence < self._accepted.sequence:
                    return self._invalidate("snapshot older than published sequence",
                                            receive_time_ns, reset=True)
                if message.state_time_ns < self._state_time:
                    return self._invalidate("snapshot state time moved backwards",
                                            receive_time_ns, reset=True)
                if message.sequence == self._accepted.sequence:
                    if (message.bids, message.asks) != (self._accepted.bids, self._accepted.asks):
                        return self._invalidate("snapshot conflicts with published state",
                                                receive_time_ns, reset=True)
            if self._anchor is None or message.sequence >= self._anchor.sequence:
                self._anchor = message
                self._updates = {seq: update for seq, update in self._updates.items()
                                 if seq > message.sequence}

        outcome = self._try_publish(receive_time_ns)
        if outcome.published:
            return outcome
        if len(self._updates) > self._config.max_buffer:
            self._since = receive_time_ns
            return self._invalidate("recovery buffer limit exceeded",
                                    receive_time_ns, reset=True)
        return outcome

    def _try_publish(self, receive_time_ns):
        anchor = self._anchor
        if anchor is not None:
            sequence, bids, asks = anchor.sequence, anchor.bids, anchor.asks
            state_time = anchor.state_time_ns
            applied = [anchor]
            provenance = ()
        elif self._accepted is not None and not self._needs_snapshot:
            sequence = self._accepted.sequence
            bids, asks = self._accepted.bids, self._accepted.asks
            state_time = self._state_time
            applied = []
            provenance = self._provenance
        else:
            return self._invalidate("snapshot required", receive_time_ns)

        expected = sequence + 1
        deltas = []
        for seq, message in sorted(self._updates.items()):
            if seq <= sequence:
                continue
            if seq != expected:
                return self._invalidate(f"waiting for sequence {expected}", receive_time_ns)
            deltas.append(message)
            expected += 1
        if expected - 1 < self._high_water:
            return self._invalidate(f"waiting for sequence {expected}", receive_time_ns)

        candidate = OrderBook()
        try:
            candidate.apply_snapshot(sequence=sequence, bids=bids, asks=asks)
            for message in deltas:
                if message.state_time_ns < state_time:
                    raise BookError("update state time moved backwards")
                candidate.apply_update(sequence=message.sequence, updates=message.updates)
                state_time = message.state_time_ns
                applied.append(message)
        except BookError as exc:
            return self._invalidate(str(exc), receive_time_ns, reset=True)
        if not applied:
            return RecoveryOutcome(False, "no new state")

        self._publication += 1
        self._book = replace(candidate.state, version=self._publication,
                             required_snapshot_sequence=candidate.state.sequence)
        self._accepted = self._book
        self._last_receive = receive_time_ns
        self._state_time = state_time
        self._provenance = provenance + tuple(message.event_index for message in applied)
        self._updates.clear()
        self._anchor = None
        self._needs_snapshot = False
        self._since = None
        self._reason = None
        return RecoveryOutcome(True, None, tuple(applied))

    def check_timeout(self, time_ns: int) -> RecoveryOutcome:
        self._clock(time_ns)
        self._last_clock = time_ns
        if self._book.is_valid:
            return RecoveryOutcome(False, None)
        if self._since is None:
            # The caller initializes this at its bootstrap time. A constructor
            # cannot assume that the session starts at the clock origin.
            self._since = time_ns
            return RecoveryOutcome(False, None)
        if time_ns - self._since < self._config.timeout_ns:
            return RecoveryOutcome(False, None)
        self._since = time_ns
        return self._invalidate("recovery timeout", time_ns, reset=True)
