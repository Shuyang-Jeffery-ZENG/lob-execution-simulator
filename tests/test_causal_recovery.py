"""Independent sequence, timing and atomicity expectations for local recovery."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal as D

import pytest

from lob_lab.book import LevelUpdate
from lob_sim.models import MarketMessage, RecoveryConfig
from lob_sim.recovery import LocalRecovery


def snapshot(seq=100, time=0, index=0, size="10", **kwargs):
    return MarketMessage(time, index, seq, "snapshot",
                         ((D("99"), D("10")),), ((D("101"), D(size)),),
                         **kwargs)


def update(seq, time, index, size="5", **kwargs):
    return MarketMessage(time, index, seq, updates=(
        LevelUpdate("ASK", D("101"), D(size)),), **kwargs)


def ready(*, config=RecoveryConfig(), session="synthetic-1", time=0):
    local = LocalRecovery(session, config)
    local.receive(snapshot(time=time, session_id=session), time)
    return local


def test_initial_view_is_immutable_and_bootstrap_sets_timeout_origin():
    local = LocalRecovery("synthetic-1", RecoveryConfig(timeout_ns=10))
    before = local.view
    assert not before.book.is_valid and before.publication == 0
    assert before.last_receive_time_ns is before.state_time_ns is None
    assert before.recovering_since_ns is None and before.provenance == ()
    with pytest.raises(FrozenInstanceError):
        before.publication = 4
    assert local.check_timeout(50).reason is None
    assert local.view.recovering_since_ns == 50
    assert local.check_timeout(59).reason is None
    assert local.check_timeout(60).reason == "recovery timeout"
    assert local.view.recovering_since_ns == 60
    assert local.check_timeout(60).reason is None
    assert local.check_timeout(70).reason == "recovery timeout"
    assert before.recovering_since_ns is None


@pytest.mark.parametrize("changes", [
    {"max_buffer": 0}, {"max_buffer": True}, {"timeout_ns": 0},
    {"timeout_ns": -1}, {"max_requests": -1},
    {"snapshot_read_latency_ns": -1}, {"snapshot_response_latency_ns": True},
])
def test_invalid_recovery_configuration_fails_before_observation(changes):
    with pytest.raises(ValueError):
        LocalRecovery("synthetic-1", replace(RecoveryConfig(), **changes))


def test_publications_keep_global_versions_and_current_origin_chain():
    local = ready()
    old = local.view
    message = update(101, 2, 1, "8")
    outcome = local.receive(message, 5)
    assert outcome.published and outcome.applied == (message,)
    view = local.view
    assert view.book.asks == ((D("101"), D("8")),)
    assert (view.publication, view.book.version, view.book.sequence) == (2, 2, 101)
    assert (view.last_receive_time_ns, view.state_time_ns) == (5, 2)
    assert view.provenance == (0, 1)
    assert old.book.asks == ((D("101"), D("10")),)
    assert old.publication == 1 and old.provenance == (0,)
    local.receive(snapshot(120, 6, 2, "11"), 7)
    assert local.view.publication == 3 and local.view.provenance == (2,)


def test_snapshot_103_bridges_cached_104_105_106_atomically():
    local = ready()
    preserved = local.view
    messages = [update(102, 9, 1, "9"), update(104, 13, 2, "7"),
                update(105, 14, 3, "6"), update(106, 15, 4, "5")]
    for message in messages:
        assert not local.receive(message, message.state_time_ns).published
        assert local.view.book.sequence == 100
        assert local.view.book.asks == preserved.book.asks
        assert local.view.publication == 1
    assert local.view.high_water_sequence == 106
    assert local.view.buffered_sequences == (102, 104, 105, 106)
    anchor = snapshot(103, 12, 5, "8")
    result = local.receive(anchor, 16)
    assert result.published
    assert [message.sequence for message in result.applied] == [103, 104, 105, 106]
    view = local.view
    assert view.book.is_valid and view.book.sequence == 106
    assert view.book.asks == ((D("101"), D("5")),)
    assert view.publication == view.book.version == 2
    assert (view.last_receive_time_ns, view.state_time_ns) == (16, 15)
    assert view.provenance == (5, 2, 3, 4)
    assert view.buffered_sequences == () and view.recovering_since_ns is None
    assert preserved.book.is_valid and preserved.book.sequence == 100


def test_missing_105_keeps_candidate_private_until_the_gap_is_filled():
    local = ready()
    for message in [update(102, 9, 1), update(104, 13, 2, "7"),
                    update(106, 15, 4, "5"), snapshot(103, 12, 5, "8")]:
        local.receive(message, max(message.state_time_ns, 16 if message.kind == "snapshot" else 0))
    view = local.view
    assert not view.book.is_valid and view.book.sequence == 100
    assert view.book.asks == ((D("101"), D("10")),)
    assert view.buffered_sequences == (104, 106)
    assert view.reason == "waiting for sequence 105"
    result = local.receive(update(105, 14, 3, "6"), 17)
    assert result.published and local.view.book.sequence == 106
    assert local.view.state_time_ns == 15 and local.view.last_receive_time_ns == 17
    assert local.view.provenance == (5, 2, 3, 4)


def test_late_missing_update_can_replay_from_last_accepted_base():
    local = ready()
    later = update(102, 2, 2, "6")
    earlier = update(101, 1, 1, "8")
    local.receive(later, 3)
    result = local.receive(earlier, 4)
    assert result.published and result.applied == (earlier, later)
    assert local.view.book.asks == ((D("101"), D("6")),)
    assert local.view.provenance == (0, 1, 2)
    assert (local.view.state_time_ns, local.view.last_receive_time_ns) == (2, 4)
    assert local.view.publication == 2


def test_duplicate_delivery_and_duplicate_update_do_not_refresh_either_age():
    local = ready()
    message = update(101, 1, 1, "8")
    local.receive(message, 2)
    before = local.view
    assert local.receive(message, 10).reason == "duplicate delivery"
    equivalent = replace(message, event_index=2, updates=(
        LevelUpdate("ASK", D("101.00"), D("8.000")),))
    assert local.receive(equivalent, 11).reason == "duplicate update"
    assert local.view == before


def test_new_same_sequence_snapshot_can_revalidate_but_delivery_duplicate_cannot():
    local = ready()
    message = snapshot(time=10, index=2**63)
    assert local.receive(message, 12).published
    view = local.view
    assert (view.book.sequence, view.publication) == (100, 2)
    assert (view.state_time_ns, view.last_receive_time_ns) == (10, 12)
    assert view.provenance == (2**63,)
    assert not local.receive(message, 20).published
    assert local.view == view


def test_unicode_session_identity_is_supported_without_loss_of_provenance():
    local = ready(session="合成会话一")
    message = update(101, 1, 2**63, session_id="合成会话一")
    assert local.receive(message, 2).published
    assert local.view.provenance == (0, 2**63)


@pytest.mark.parametrize("different_index", [False, True])
def test_conflicting_duplicate_invalidates_without_partial_publication(different_index):
    local = ready()
    message = update(101, 1, 1, "8")
    local.receive(message, 2)
    before = local.view
    conflict = update(101, 1, 2 if different_index else 1, "7")
    result = local.receive(conflict, 3)
    assert "conflicting duplicate" in result.reason
    assert not local.view.book.is_valid
    assert local.view.book.asks == before.book.asks
    assert local.view.publication == before.publication
    assert local.view.last_receive_time_ns == 2
    assert local.receive(snapshot(102, 4, 3, "9"), 5).published


def test_same_sequence_snapshot_with_changed_depth_is_a_conflict():
    local = ready()
    result = local.receive(snapshot(100, 1, 1, "9"), 2)
    assert "conflicting snapshot" in result.reason
    assert not local.view.book.is_valid
    assert local.view.book.asks == ((D("101"), D("10")),)
    assert local.receive(snapshot(101, 3, 2, "9"), 4).published


def test_old_covered_update_is_diagnosed_without_refreshing_the_snapshot():
    local = ready()
    before = local.view
    result = local.receive(update(99, 0, 1), 3)
    assert result.reason == "update covered by published state"
    assert local.view == before


def test_older_snapshot_cannot_roll_back_a_published_sequence():
    local = ready()
    result = local.receive(snapshot(99, 1, 1), 2)
    assert result.reason == "snapshot older than published sequence"
    assert not local.view.book.is_valid and local.view.book.sequence == 100
    assert local.view.publication == 1 and local.view.high_water_sequence == 100


@pytest.mark.parametrize("sequence", [100, 101])
def test_snapshot_state_time_cannot_regress_even_with_a_higher_sequence(sequence):
    local = ready(time=10)
    result = local.receive(snapshot(sequence, 9, 1), 12)
    assert result.reason == "snapshot state time moved backwards"
    assert not local.view.book.is_valid and local.view.state_time_ns == 10
    assert local.view.publication == 1


def test_increment_state_time_cannot_regress_while_repairing_a_gap():
    local = ready(time=10)
    result = local.receive(update(101, 9, 1), 12)
    assert result.reason == "update state time moved backwards"
    assert not local.view.book.is_valid and local.view.book.sequence == 100


@pytest.mark.parametrize("size", ["0", "-1", "NaN", "Infinity"])
def test_malformed_snapshot_preserves_last_accepted_data_and_recovery_target(size):
    local = ready()
    result = local.receive(snapshot(105, 1, 1, size), 2)
    assert not result.published and result.reason
    assert local.view.high_water_sequence == 105
    assert local.view.book.sequence == 100 and local.view.publication == 1
    assert local.view.book.asks == ((D("101"), D("10")),)
    assert local.receive(snapshot(105, 3, 2, "9"), 4).published


def test_batch_with_a_late_bad_level_never_publishes_the_earlier_valid_change():
    local = ready()
    message = MarketMessage(1, 1, 101, updates=(
        LevelUpdate("ASK", D("101"), D("8")),
        LevelUpdate("BID", D("98"), D("-1")),))
    assert not local.receive(message, 2).published
    assert local.view.book.asks == ((D("101"), D("10")),)
    assert local.view.publication == 1 and local.view.buffered_sequences == ()


def test_valid_batch_that_crosses_the_actual_predecessor_is_not_published():
    local = ready()
    message = MarketMessage(1, 1, 101, updates=(LevelUpdate("BID", D("102"), D("2")),))
    result = local.receive(message, 2)
    assert "strictly below" in result.reason
    assert local.view.book.bids == ((D("99"), D("10")),)
    assert local.view.book.sequence == 100 and local.view.publication == 1


def test_foreign_session_is_invalid_but_cannot_raise_this_sessions_high_water():
    local = ready()
    result = local.receive(snapshot(999999, 1, 1, session_id="other"), 2)
    assert "another source session" in result.reason
    assert local.view.high_water_sequence == 100
    assert not local.view.book.is_valid
    assert local.receive(snapshot(100, 3, 2), 4).published


def test_incomplete_snapshot_cannot_claim_recovery():
    local = ready()
    result = local.receive(snapshot(101, 1, 1, full_depth=False), 2)
    assert result.reason == "full-depth message required"
    assert not local.view.book.is_valid and local.view.high_water_sequence == 101
    assert local.receive(snapshot(101, 3, 2), 4).published


def test_overflow_clears_payloads_preserves_high_water_and_requires_snapshot():
    local = ready(config=RecoveryConfig(max_buffer=1))
    local.receive(update(102, 1, 1), 2)
    assert local.view.buffered_sequences == (102,)
    result = local.receive(update(104, 3, 2), 4)
    assert result.reason == "recovery buffer limit exceeded"
    assert local.view.buffered_sequences == ()
    assert local.view.high_water_sequence == 104 and local.view.recovering_since_ns == 4
    assert local.view.book.sequence == 100 and local.view.last_receive_time_ns == 0
    assert not local.receive(update(101, 1, 3), 5).published
    assert local.receive(snapshot(104, 6, 4), 7).published


def test_timeout_discards_candidate_and_buffer_but_not_known_high_water():
    local = ready(config=RecoveryConfig(timeout_ns=10))
    local.receive(update(102, 1, 1), 5)
    assert local.check_timeout(14).reason is None
    assert local.check_timeout(15).reason == "recovery timeout"
    assert local.view.recovering_since_ns == 15
    assert local.view.high_water_sequence == 102 and local.view.buffered_sequences == ()
    assert not local.receive(update(101, 1, 2), 16).published
    assert local.view.book.sequence == 100
    assert local.receive(snapshot(102, 17, 3), 18).published
    assert local.check_timeout(100).reason is None


def test_timeout_discards_an_isolated_snapshot_anchor_too():
    local = ready(config=RecoveryConfig(timeout_ns=10))
    local.receive(update(103, 3, 3, "5"), 3)
    local.receive(snapshot(101, 1, 1, "8"), 4)
    assert local.view.reason == "waiting for sequence 102"
    assert local.check_timeout(13).reason == "recovery timeout"
    # Filling the old gap cannot reuse the expired snapshot candidate.
    assert not local.receive(update(102, 2, 2, "6"), 14).published
    assert local.view.book.sequence == 100 and local.view.high_water_sequence == 103
    assert local.receive(snapshot(103, 15, 4, "5"), 16).published


def test_duplicate_during_gap_does_not_postpone_the_recovery_timeout():
    local = ready(config=RecoveryConfig(timeout_ns=10))
    message = update(102, 1, 1)
    local.receive(message, 5)
    before = local.view
    assert local.receive(message, 14).reason == "duplicate delivery"
    assert local.view == before
    assert local.check_timeout(15).reason == "recovery timeout"


def test_failed_candidate_does_not_publish_an_earlier_valid_cached_update():
    local = ready()
    local.receive(update(104, 4, 4, "6"), 5)
    local.receive(MarketMessage(5, 5, 105, updates=(
        LevelUpdate("BID", D("102"), D("2")),)), 6)
    result = local.receive(snapshot(103, 3, 3, "8"), 7)
    assert not result.published and "strictly below" in result.reason
    assert local.view.book.sequence == 100 and local.view.publication == 1
    assert local.view.book.asks == ((D("101"), D("10")),)
    assert local.view.high_water_sequence == 105 and local.view.buffered_sequences == ()


def test_missing_initial_snapshot_cannot_be_replaced_by_incremental_messages():
    local = LocalRecovery("synthetic-1", RecoveryConfig())
    result = local.receive(update(2, 2, 2), 3)
    assert not result.published and local.view.book.sequence is None
    assert local.view.high_water_sequence == 2
    result = local.receive(snapshot(1, 1, 1), 4)
    assert result.published and local.view.book.sequence == 2
    assert local.view.provenance == (1, 2)


@pytest.mark.parametrize("time,kind", [(9, "past"), (10, "future_source")])
def test_clock_metadata_errors_fail_without_mutating_the_view(time, kind):
    local = ready(time=10)
    before = local.view
    message = update(101, 0 if kind == "past" else 11, 1)
    with pytest.raises(ValueError):
        local.receive(message, time)
    assert local.view == before


def test_empty_valid_snapshot_and_deletion_update_remain_valid_states():
    local = ready()
    assert local.receive(update(101, 1, 1, "0"), 2).published
    assert local.view.book.is_valid and local.view.book.asks == ()
    assert local.receive(MarketMessage(3, 2, 102, "snapshot"), 4).published
    assert local.view.book.is_valid and local.view.book.bids == local.view.book.asks == ()
