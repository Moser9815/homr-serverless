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
    """Verify the per-bucket spatial matching algorithm.

    Spatial matching uses x-coordinate order (entries) vs beat order (parsed)
    to align notes — NO pitch information is used for matching. Pitch
    comparison only happens AFTER matching, in _apply_selective_override.

    Key properties:
    - Order-preserving: left-to-right in both lists
    - Phantom handling: DP skips extra entries (phantoms) at low cost
    - Chord handling: same-beat notes match same-x entries by y-order
    """

    def test_exact_match(self):
        """All notes in same order — should pair perfectly."""
        parsed = [
            {"pitch": 64, "pitch_original_transformer": 64, "beat": 1.0},  # E4
            {"pitch": 67, "pitch_original_transformer": 67, "beat": 2.0},  # G4
        ]
        entries = [
            {"position": 1, "x": 100, "y": 300},  # E4 in treble
            {"position": 3, "x": 200, "y": 280},  # G4 in treble
        ]
        matches = _match_notes_to_entries(parsed, entries, "treble")
        assert len(matches) == 2

    def test_phantom_skipped_by_dp(self):
        """A phantom entry between two real notes — DP should skip it.

        With spatial matching, the DP aligns by position order. When 2 parsed
        notes align with entries at positions 0 and 2 (skipping phantom at 1),
        the DP pays one gap penalty to skip the phantom and matches correctly.
        """
        parsed = [
            {"pitch": 64, "pitch_original_transformer": 64, "beat": 1.0},  # E4
            {"pitch": 67, "pitch_original_transformer": 67, "beat": 3.0},  # G4
        ]
        entries = [
            {"position": 1, "x": 100, "y": 300},   # beat 1 region
            {"position": -3, "x": 200, "y": 350},  # phantom at beat 2 region
            {"position": 3, "x": 300, "y": 280},   # beat 3 region
        ]
        matches = _match_notes_to_entries(parsed, entries, "treble")
        assert len(matches) == 2
        # First parsed note matches first entry, second matches third
        assert matches[0][1]["x"] == 100
        assert matches[1][1]["x"] == 300

    def test_more_parsed_than_entries(self):
        """When parsed has more notes than segmentation (chord detection gap)."""
        parsed = [
            {"pitch": 64, "pitch_original_transformer": 64, "beat": 1.0},
            {"pitch": 67, "pitch_original_transformer": 67, "beat": 2.0},
            {"pitch": 71, "pitch_original_transformer": 71, "beat": 3.0},  # extra
        ]
        entries = [
            {"position": 1, "x": 100, "y": 300},
            {"position": 3, "x": 200, "y": 280},
        ]
        matches = _match_notes_to_entries(parsed, entries, "treble")
        assert len(matches) == 2  # Only 2 can match

    def test_more_entries_than_parsed(self):
        """When segmentation has more entries (phantoms) — match by position."""
        parsed = [
            {"pitch": 67, "pitch_original_transformer": 67, "beat": 2.0},
        ]
        entries = [
            {"position": -2, "x": 100, "y": 350},  # phantom at start
            {"position": 3, "x": 200, "y": 280},   # real note in middle
            {"position": 8, "x": 300, "y": 250},   # phantom at end
        ]
        matches = _match_notes_to_entries(parsed, entries, "treble")
        assert len(matches) == 1
        # Single parsed note should match the middle entry (position index 1 of 3
        # is closest to normalized position 0.0 of 1 parsed note... actually
        # with DP the single parsed note matches the first entry that minimizes
        # total cost). The key property is that exactly 1 match occurs.
        assert matches[0][0]["pitch"] == 67

    def test_spatial_matching_ignores_pitch(self):
        """Even when pitch is wildly wrong, spatial order determines matching.

        This is the KEY test: the old pitch-proximity system would reject
        this match (diff=13). Spatial matching pairs them because they're
        at the same position in their respective orderings. The override
        logic in _apply_selective_override handles the pitch correction.
        """
        parsed = [
            {"pitch": 64, "pitch_original_transformer": 64, "beat": 1.0},  # E4
        ]
        entries = [
            {"position": 9, "x": 100, "y": 200},  # F5 in treble = 77, diff=13
        ]
        matches = _match_notes_to_entries(parsed, entries, "treble")
        # Spatial matching ALWAYS matches by position — override logic handles pitch
        assert len(matches) == 1

    def test_chord_matching_by_y_order(self):
        """Chord notes (same beat) should match same-x entries by y order.

        Higher pitch = lower y on page. Sorting parsed by -pitch and
        entries by y gives matching vertical order.
        """
        parsed = [
            {"pitch": 67, "pitch_original_transformer": 67, "beat": 1.0},  # G4 (higher)
            {"pitch": 64, "pitch_original_transformer": 64, "beat": 1.0},  # E4 (lower)
        ]
        entries = [
            {"position": 3, "x": 100, "y": 280},  # G4 — higher pitch, lower y
            {"position": 1, "x": 100, "y": 300},  # E4 — lower pitch, higher y
        ]
        matches = _match_notes_to_entries(parsed, entries, "treble")
        assert len(matches) == 2
        # Higher pitch parsed (G4=67) matches lower y entry (y=280, pos=3)
        # Lower pitch parsed (E4=64) matches higher y entry (y=300, pos=1)
        for note, entry in matches:
            if note["pitch"] == 67:
                assert entry["position"] == 3
            elif note["pitch"] == 64:
                assert entry["position"] == 1


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
        """Large diff (>= 7) with different pitch class → geometric override."""
        note = {"pitch": 64, "pitch_original_transformer": 64, "pitch_confidence": 0.9}  # E4
        entry = {"position": 8}  # G5 in treble... let me compute: pos 8 treble
        # pos 8 = ref(D,4) + 8 = total_idx=9, oct=4+9//7=5, note=9%7=2=E, E5=76
        # diff = |64-76| = 12 but same pitch class! That's octave fix not hallucination.
        # Need different pitch class. Use pos 9 = F5 = 77, diff=|64-77|=13, pc_diff=|4-5|=1
        # Still within 1! Let me pick something truly different.
        # pos 7 treble: total_idx=8, oct=4+8//7=5, note=8%7=1=D, D5=74
        # diff = |64-74| = 10. That's < 11, so won't trigger octave fix. diff >= 7 → hallucination.
        entry2 = {"position": 7}  # D5 in treble = 74, diff=10
        _apply_selective_override(note, entry2, "treble")
        assert note["pitch"] == 74  # Override to D5
        assert "geometric" in note["pitch_source"]

    def test_octave_error_same_pitch_class(self):
        """D2 vs D3 — exact octave, same pitch class → octave fix."""
        note = {"pitch": 38, "pitch_original_transformer": 38, "pitch_confidence": 0.9}  # D2
        entry = {"position": 5}  # D3 in bass = 50, diff=12
        _apply_selective_override(note, entry, "bass")
        assert note["pitch"] == 50  # Octave fix to D3
        assert "octave_fix" in note["pitch_source"]

    def test_octave_error_with_accidental(self):
        """Eb3 vs D2/E2 — transformer has accidental, geometric is diatonic.
        Diff within 1 semitone of octave → preserve Eb, fix octave.
        This is the Drift Away m16 beat 1.0 staff 1 case."""
        note = {"pitch": 51, "pitch_original_transformer": 51, "pitch_confidence": 0.9}  # Eb3
        entry = {"position": -2}  # D2 in bass = 38, diff=|51-38|=13
        # pitch_class_diff: |51%12 - 38%12| = |3-2| = 1 → within 1
        _apply_selective_override(note, entry, "bass")
        # new_midi = (38//12)*12 + (51%12) = 36 + 3 = 39 = Eb2
        assert note["pitch"] == 39
        assert "octave_fix" in note["pitch_source"]

    def test_octave_error_with_accidental_e(self):
        """Eb3 vs E2 — diff=11, pitch_class_diff=|3-4|=1 → octave fix."""
        note = {"pitch": 51, "pitch_original_transformer": 51, "pitch_confidence": 0.9}  # Eb3
        entry = {"position": -1}  # E2 in bass = 40, diff=|51-40|=11
        _apply_selective_override(note, entry, "bass")
        # new_midi = (40//12)*12 + (51%12) = 36 + 3 = 39 = Eb2
        assert note["pitch"] == 39
        assert "octave_fix" in note["pitch_source"]

    def test_no_octave_fix_when_pitch_class_differs(self):
        """Diff >= 11 but pitch classes differ by > 1 → regular hallucination."""
        note = {"pitch": 60, "pitch_original_transformer": 60, "pitch_confidence": 0.9}  # C4
        entry = {"position": 9}  # A3 in bass = 57, diff=3... too small.
        # Need diff >= 11 with pitch class diff > 1.
        # C4=60 vs G2=43 in bass (pos 1): diff=17, pc_diff=|0-7|=5 → hallucination
        entry2 = {"position": 1}  # G2 in bass = 43
        _apply_selective_override(note, entry2, "bass")
        assert note["pitch"] == 43  # Regular geometric override
        assert "geometric" in note["pitch_source"]
        assert "octave_fix" not in note["pitch_source"]

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


# ===== Measure beat validation (Step 5) =====

from parse_musicxml import validate_measure_beats


class TestMeasureBeatValidation:
    """Verify measure beat validation flags incorrect durations."""

    def test_correct_4_4_measure(self):
        """Four quarter notes in 4/4 — no flag."""
        notes = [
            {"measure": 1, "staff": 1, "voice": 1, "beat": 1.0, "duration_beats": 1.0},
            {"measure": 1, "staff": 1, "voice": 1, "beat": 2.0, "duration_beats": 1.0},
            {"measure": 1, "staff": 1, "voice": 1, "beat": 3.0, "duration_beats": 1.0},
            {"measure": 1, "staff": 1, "voice": 1, "beat": 4.0, "duration_beats": 1.0},
        ]
        flags = validate_measure_beats(notes, [], "4/4")
        assert len(flags) == 0

    def test_overfull_measure(self):
        """5 beats in 4/4 — overfull flag."""
        notes = [
            {"measure": 1, "staff": 1, "voice": 1, "beat": i, "duration_beats": 1.0}
            for i in [1.0, 2.0, 3.0, 4.0, 5.0]
        ]
        flags = validate_measure_beats(notes, [], "4/4")
        assert len(flags) == 1
        assert flags[0]["status"] == "overfull"
        assert flags[0]["actual"] == 5.0

    def test_underfull_measure(self):
        """2 beats in 4/4 — underfull flag."""
        notes = [
            {"measure": 1, "staff": 1, "voice": 1, "beat": 1.0, "duration_beats": 1.0},
            {"measure": 1, "staff": 1, "voice": 1, "beat": 2.0, "duration_beats": 1.0},
        ]
        flags = validate_measure_beats(notes, [], "4/4")
        assert len(flags) == 1
        assert flags[0]["status"] == "underfull"

    def test_chord_not_double_counted(self):
        """Two notes at same beat (chord) — only count duration once."""
        notes = [
            {"measure": 1, "staff": 1, "voice": 1, "beat": 1.0, "duration_beats": 2.0},  # C4
            {"measure": 1, "staff": 1, "voice": 1, "beat": 1.0, "duration_beats": 2.0},  # E4 (chord)
            {"measure": 1, "staff": 1, "voice": 1, "beat": 3.0, "duration_beats": 2.0},
        ]
        flags = validate_measure_beats(notes, [], "4/4")
        assert len(flags) == 0  # 2.0 + 2.0 = 4.0, correct

    def test_6_8_time(self):
        """6/8 expects 3.0 quarter-note beats."""
        notes = [
            {"measure": 1, "staff": 1, "voice": 1, "beat": 1.0, "duration_beats": 1.5},
            {"measure": 1, "staff": 1, "voice": 1, "beat": 2.5, "duration_beats": 1.5},
        ]
        flags = validate_measure_beats(notes, [], "6/8")
        assert len(flags) == 0  # 1.5 + 1.5 = 3.0

    def test_rests_included(self):
        """Rests contribute to the total."""
        notes = [
            {"measure": 1, "staff": 1, "voice": 1, "beat": 1.0, "duration_beats": 1.0},
        ]
        rests = [
            {"measure": 1, "staff": 1, "voice": 1, "beat": 2.0, "duration_beats": 3.0},
        ]
        flags = validate_measure_beats(notes, rests, "4/4")
        assert len(flags) == 0  # 1.0 + 3.0 = 4.0

    def test_multiple_voices_separate(self):
        """Different voices are validated independently."""
        notes = [
            {"measure": 1, "staff": 1, "voice": 1, "beat": 1.0, "duration_beats": 4.0},  # Voice 1: OK
            {"measure": 1, "staff": 1, "voice": 2, "beat": 1.0, "duration_beats": 2.0},  # Voice 2: underfull
        ]
        flags = validate_measure_beats(notes, [], "4/4")
        assert len(flags) == 1
        assert flags[0]["voice"] == 2
        assert flags[0]["status"] == "underfull"

    def test_tolerance(self):
        """Slight rounding errors within tolerance — no flag."""
        notes = [
            {"measure": 1, "staff": 1, "voice": 1, "beat": 1.0, "duration_beats": 1.0},
            {"measure": 1, "staff": 1, "voice": 1, "beat": 2.0, "duration_beats": 1.0},
            {"measure": 1, "staff": 1, "voice": 1, "beat": 3.0, "duration_beats": 1.0},
            {"measure": 1, "staff": 1, "voice": 1, "beat": 4.0, "duration_beats": 0.95},  # 3.95, within 0.1
        ]
        flags = validate_measure_beats(notes, [], "4/4")
        assert len(flags) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
