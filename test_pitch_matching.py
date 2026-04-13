"""
Regression tests for geometric pitch cross-check system.

Tests the core algorithms in pitch_from_position.py:
1. Position-to-MIDI mapping for all clefs
2. Per-bucket matching (replaces old greedy cursor)
3. Clef change handling
4. Selective override rules

Run: python -m pytest test_pitch_matching.py -v
"""

import pytest
from pitch_from_position import (
    position_to_midi,
    _diatonic_pitch,
    _effective_clef_at,
    _match_notes_to_entries,
    _apply_selective_override,
    _find_bucket,
    _detect_measure_offset,
    recompute_pitches_with_confidence,
)


# ===== Position-to-MIDI mapping =====

class TestPositionToMidi:
    """Verify position → MIDI for all clefs. These are the foundation
    of the geometric pitch system — if these are wrong, everything is."""

    def test_treble_bottom_line(self):
        midi, name = position_to_midi(1, "treble")
        assert midi == 64 and name == "E4"

    def test_treble_middle_line(self):
        midi, name = position_to_midi(5, "treble")
        assert midi == 71 and name == "B4"

    def test_treble_top_line(self):
        midi, name = position_to_midi(9, "treble")
        assert midi == 77 and name == "F5"

    def test_treble_middle_c(self):
        midi, name = position_to_midi(-1, "treble")
        assert midi == 60 and name == "C4"

    def test_treble_ledger_above(self):
        midi, name = position_to_midi(11, "treble")
        assert midi == 81 and name == "A5"

    def test_bass_bottom_line(self):
        midi, name = position_to_midi(1, "bass")
        assert midi == 43 and name == "G2"

    def test_bass_middle_line(self):
        midi, name = position_to_midi(5, "bass")
        assert midi == 50 and name == "D3"

    def test_bass_top_line(self):
        midi, name = position_to_midi(9, "bass")
        assert midi == 57 and name == "A3"

    def test_bass_middle_c(self):
        midi, name = position_to_midi(11, "bass")
        assert midi == 60 and name == "C4"

    def test_bass_ledger_below(self):
        midi, name = position_to_midi(-1, "bass")
        assert midi == 40 and name == "E2"

    def test_alto_middle_line(self):
        midi, name = position_to_midi(5, "alto")
        assert midi == 60 and name == "C4"

    def test_key_signature_sharp(self):
        """G major: F# applied to all F notes."""
        midi, name = position_to_midi(2, "treble", fifths=1)  # pos 2 = F4
        assert midi == 66 and name == "F#4"

    def test_key_signature_flat(self):
        """F major: Bb applied to all B notes."""
        midi, name = position_to_midi(5, "treble", fifths=-1)  # pos 5 = B4
        assert midi == 70 and name == "Bb4"

    def test_position_zero_treble(self):
        """Position 0 = space below bottom line."""
        midi, name = position_to_midi(0, "treble")
        assert midi == 62 and name == "D4"

    def test_position_ten_treble(self):
        """Position 10 = space above top line (valid ledger line area)."""
        midi, name = position_to_midi(10, "treble")
        # G5 is space above F5 (top line)
        # Actually: pos 9=F5, pos 10=G5? Let me verify
        # ref_note_idx=1 (D), position=10, total_idx=11
        # octave=4+11//7=4+1=5, note_within=11%7=4=G
        # midi=(5+1)*12+7=79, name=G5
        assert midi == 79 and name == "G5"


# ===== Grand staff: no position remapping needed =====

class TestGrandStaffPositions:
    """Verify that positions are per-physical-staff (not merged 10-line grid).
    The old remap_grand_staff_position() was wrong — these tests prove
    no remapping is needed."""

    def test_treble_staff_position_3_is_G4_not_B2(self):
        """The old bug: position 3 on staff 1 was mapped to bass → B2 (47).
        Correct: position 3 on treble = G4 (67)."""
        midi, name = position_to_midi(3, "treble")
        assert midi == 67 and name == "G4"
        # The old wrong result was:
        wrong_midi, _ = position_to_midi(3, "bass")
        assert wrong_midi == 47  # B2 — 20 semitones wrong!
        assert midi - wrong_midi == 20

    def test_bass_staff_position_5_is_D3(self):
        midi, name = position_to_midi(5, "bass")
        assert midi == 50 and name == "D3"

    def test_both_staves_use_same_position_range(self):
        """Both staves have positions in ~(-6, 10) range. No 11-19 for treble."""
        for pos in range(-4, 11):
            treble_midi, _ = position_to_midi(pos, "treble")
            bass_midi, _ = position_to_midi(pos, "bass")
            assert treble_midi > bass_midi, f"pos {pos}: treble ({treble_midi}) should be higher than bass ({bass_midi})"


# ===== Clef change handling =====

class TestClefChanges:
    """Verify mid-piece clef changes (e.g., Drift Away staff 1 switches
    treble→bass at m9, back to treble at m13)."""

    DRIFT_AWAY_CLEF_CHANGES = [
        {"staff": 1, "measure": 1, "clef": "treble"},
        {"staff": 2, "measure": 1, "clef": "bass"},
        {"staff": 1, "measure": 9, "clef": "bass"},
        {"staff": 1, "measure": 13, "clef": "treble"},
        {"staff": 1, "measure": 17, "clef": "bass"},
    ]

    def test_staff1_m1_treble(self):
        clef = _effective_clef_at(self.DRIFT_AWAY_CLEF_CHANGES, 1, 1, "treble")
        assert clef == "treble"

    def test_staff1_m9_bass(self):
        clef = _effective_clef_at(self.DRIFT_AWAY_CLEF_CHANGES, 1, 9, "treble")
        assert clef == "bass"

    def test_staff1_m12_bass(self):
        clef = _effective_clef_at(self.DRIFT_AWAY_CLEF_CHANGES, 1, 12, "treble")
        assert clef == "bass"

    def test_staff1_m13_treble(self):
        clef = _effective_clef_at(self.DRIFT_AWAY_CLEF_CHANGES, 1, 13, "treble")
        assert clef == "treble"

    def test_staff1_m17_bass(self):
        clef = _effective_clef_at(self.DRIFT_AWAY_CLEF_CHANGES, 1, 17, "treble")
        assert clef == "bass"

    def test_staff2_always_bass(self):
        for m in [1, 9, 13, 17, 25]:
            clef = _effective_clef_at(self.DRIFT_AWAY_CLEF_CHANGES, 2, m, "bass")
            assert clef == "bass", f"Staff 2 m{m} should be bass"

    def test_empty_changes_uses_default(self):
        clef = _effective_clef_at([], 1, 5, "treble")
        assert clef == "treble"


# ===== Per-bucket matching =====

class TestBucketMatching:
    """Verify the per-bucket matching algorithm produces correct pairings
    and doesn't cascade errors like the old cursor approach."""

    def test_exact_match(self):
        """All notes agree — should pair perfectly."""
        parsed = [
            {"pitch": 64, "pitch_original_transformer": 64},  # E4
            {"pitch": 67, "pitch_original_transformer": 67},  # G4
        ]
        entries = [
            {"position": 1},  # E4 in treble
            {"position": 3},  # G4 in treble
        ]
        matches = _match_notes_to_entries(parsed, entries, "treble")
        assert len(matches) == 2

    def test_phantom_doesnt_steal(self):
        """A phantom notehead in the bucket shouldn't cascade."""
        parsed = [
            {"pitch": 64, "pitch_original_transformer": 64},  # E4
            {"pitch": 67, "pitch_original_transformer": 67},  # G4
        ]
        entries = [
            {"position": 1},  # E4 — matches first note
            {"position": -3},  # phantom (C3 in treble = 48) — no good match
            {"position": 3},  # G4 — matches second note
        ]
        matches = _match_notes_to_entries(parsed, entries, "treble")
        assert len(matches) == 2
        # Verify correct pairing
        for note, entry in matches:
            if note["pitch"] == 64:
                assert entry["position"] == 1
            elif note["pitch"] == 67:
                assert entry["position"] == 3

    def test_more_parsed_than_entries(self):
        """When parsed has more notes than segmentation (chord detection gap)."""
        parsed = [
            {"pitch": 64, "pitch_original_transformer": 64},
            {"pitch": 67, "pitch_original_transformer": 67},
            {"pitch": 71, "pitch_original_transformer": 71},  # extra
        ]
        entries = [
            {"position": 1},  # E4
            {"position": 3},  # G4
        ]
        matches = _match_notes_to_entries(parsed, entries, "treble")
        assert len(matches) == 2  # Only 2 can match

    def test_more_entries_than_parsed(self):
        """When segmentation has more entries (phantoms)."""
        parsed = [
            {"pitch": 67, "pitch_original_transformer": 67},  # G4
        ]
        entries = [
            {"position": -2},  # phantom
            {"position": 3},   # G4 — should match
            {"position": 8},   # phantom
        ]
        matches = _match_notes_to_entries(parsed, entries, "treble")
        assert len(matches) == 1
        assert matches[0][1]["position"] == 3

    def test_cost_threshold_rejects_bad_matches(self):
        """Entries with cost > 12 should be skipped."""
        parsed = [
            {"pitch": 64, "pitch_original_transformer": 64},  # E4
        ]
        entries = [
            {"position": 9},  # F5 in treble = 77, diff=13 → too far
        ]
        matches = _match_notes_to_entries(parsed, entries, "treble")
        assert len(matches) == 0


# ===== Selective override rules =====

class TestSelectiveOverride:

    def test_agreement_keeps_transformer(self):
        note = {"pitch": 64, "pitch_original_transformer": 64, "pitch_confidence": 0.9}
        entry = {"position": 1}  # E4 in treble = 64
        _apply_selective_override(note, entry, "treble")
        assert note["pitch"] == 64
        assert note["pitch_source"] == "agreement"

    def test_accidental_keeps_transformer(self):
        note = {"pitch": 66, "pitch_original_transformer": 66, "pitch_confidence": 0.9}  # F#4
        entry = {"position": 2}  # F4 in treble = 65, diff=1
        _apply_selective_override(note, entry, "treble")
        assert note["pitch"] == 66  # Keep F#4
        assert note["pitch_source"] == "transformer_accidental"

    def test_hallucination_overrides(self):
        note = {"pitch": 38, "pitch_original_transformer": 38, "pitch_confidence": 0.9}  # D2
        entry = {"position": 5}  # D3 in bass = 50, diff=12
        _apply_selective_override(note, entry, "bass")
        assert note["pitch"] == 50  # Override to D3
        assert "geometric" in note["pitch_source"]

    def test_low_confidence_overrides(self):
        note = {"pitch": 64, "pitch_original_transformer": 64, "pitch_confidence": 0.3}  # E4
        entry = {"position": 3}  # G4 in treble = 67, diff=3
        _apply_selective_override(note, entry, "treble")
        assert note["pitch"] == 67  # Override to G4
        assert "geometric" in note["pitch_source"]

    def test_high_confidence_small_diff_flags_only(self):
        note = {"pitch": 64, "pitch_original_transformer": 64, "pitch_confidence": 0.9}  # E4
        entry = {"position": 3}  # G4 in treble = 67, diff=3
        _apply_selective_override(note, entry, "treble")
        assert note["pitch"] == 64  # Keep transformer
        assert "flagged" in note["pitch_source"]


# ===== Bucket lookup with drift =====

class TestBucketLookup:

    def test_exact_match(self):
        buckets = {(1, 5): [{"position": 3}]}
        result = _find_bucket(buckets, 1, 5, measure_offset=0)
        assert result is not None
        assert len(result) == 1

    def test_drift_plus_1(self):
        buckets = {(1, 6): [{"position": 3}]}
        result = _find_bucket(buckets, 1, 5, measure_offset=0)
        assert result is not None

    def test_drift_minus_1(self):
        buckets = {(1, 4): [{"position": 3}]}
        result = _find_bucket(buckets, 1, 5, measure_offset=0)
        assert result is not None

    def test_no_match_returns_none(self):
        buckets = {(1, 10): [{"position": 3}]}
        result = _find_bucket(buckets, 1, 5, measure_offset=0)
        assert result is None

    def test_with_systematic_offset(self):
        """Systematic offset aligns bucket measures to parsed measures."""
        buckets = {(1, 7): [{"position": 3}]}
        result = _find_bucket(buckets, 1, 5, measure_offset=2)
        assert result is not None


class TestMeasureOffsetDetection:

    def test_no_offset_when_aligned(self):
        parsed = {(1, 1): [], (1, 2): [], (1, 3): []}
        buckets = {(1, 1): [], (1, 2): [], (1, 3): []}
        assert _detect_measure_offset(parsed, buckets) == 0

    def test_detects_plus_2_offset(self):
        parsed = {(1, 1): [], (1, 2): [], (1, 3): [], (1, 4): [], (1, 5): []}
        buckets = {(1, 3): [], (1, 4): [], (1, 5): [], (1, 6): [], (1, 7): []}
        assert _detect_measure_offset(parsed, buckets) == 2

    def test_no_offset_when_marginal(self):
        """Don't apply offset if it's not clearly better (1.5x threshold)."""
        parsed = {(1, 1): [], (1, 2): [], (1, 3): []}
        buckets = {(1, 1): [], (1, 2): [], (1, 4): []}  # 2/3 exact, offset +1 gets 2/3 too
        assert _detect_measure_offset(parsed, buckets) == 0


# ===== Double-pass overwrite protection =====

class TestDoublePassProtection:
    """Verify that running recompute_pitches_with_confidence twice doesn't
    corrupt the transformer baseline (expert review finding #14)."""

    def test_second_pass_uses_original_transformer(self):
        notes = [
            {"pitch": 64, "staff": 1, "measure": 1, "beat": 1.0},
        ]
        buckets = {(1, 1): [{"position": 3, "x": 100, "y": 200}]}  # G4 in treble
        clef_changes = [{"staff": 1, "measure": 1, "clef": "treble"}]

        # First pass
        recompute_pitches_with_confidence(notes, buckets, clef_changes)
        # Note should be flagged (diff=3, conf=1.0)
        assert notes[0]["pitch_original_transformer"] == 64

        # Second pass — should still compare against original 64, not current value
        recompute_pitches_with_confidence(notes, buckets, clef_changes)
        assert notes[0]["pitch_original_transformer"] == 64
        assert notes[0]["pitch_transformer"] == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
