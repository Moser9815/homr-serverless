"""
Geometric pitch computation from staff position.

Maps note.position (staff line/space index from HOMR segmentation) + clef
to MIDI pitch. This is how Audiveris and oemer compute pitch — it's
deterministic and doesn't depend on the transformer.

HOMR position system (from find_position_in_unit_sizes):
  position 1 = bottom line
  position 2 = first space
  position 3 = second line
  ...
  position 5 = middle line
  ...
  position 9 = top line
  position 0 = first space below staff
  position -1 = first ledger line below
  position 10 = first space above
  position 11 = first ledger line above

This code is licensed under AGPL-3.0 to comply with HOMR's license.
"""

# Diatonic semitone offsets within one octave: C D E F G A B
DIATONIC_SEMITONES = [0, 2, 4, 5, 7, 9, 11]
NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B"]

# Reference: for each clef, which diatonic note is at position 0.
# Position 0 = first space below the bottom line.
CLEF_REF = {
    # Treble: bottom line (pos 1) = E4, so pos 0 = D4
    # D is index 1 in C D E F G A B, octave 4
    "treble": (1, 4),  # (note_index, octave)
    # Bass: bottom line (pos 1) = G2, so pos 0 = F2
    # F is index 3, octave 2
    "bass": (3, 2),
    # Alto (C clef line 3): middle line = C4, bottom line = F3,
    # so pos 0 (space below bottom) = E3. E is index 2, octave 3.
    "alto": (2, 3),
    # Tenor (C clef line 4): middle line = C4 only if we're talking
    # line 4 from bottom. Tenor has C4 on 4th line, so lines bottom-to-top:
    # D3 F3 A3 C4 E4 → bottom line (pos 1) = D3, pos 0 = C3.
    # C is index 0, octave 3.
    "tenor": (0, 3),
}

# Key signature: sharps/flats applied to specific notes
# Sharp order: F C G D A E B (note indices 3 0 4 1 5 2 6)
SHARP_ORDER = [3, 0, 4, 1, 5, 2, 6]
# Flat order: B E A D G C F (note indices 6 2 5 1 4 0 3)
FLAT_ORDER = [6, 2, 5, 1, 4, 0, 3]

FIFTHS_TO_KEY = {
    -7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
}


def position_to_midi(position: int, clef: str, fifths: int = 0) -> tuple[int, str]:
    """
    Convert staff position + clef to MIDI pitch and note name.

    Args:
        position: Staff position from HOMR segmentation (0-8 typical,
                  negative for ledger lines below, >8 for above)
        clef: "treble", "bass", or "alto"
        fifths: Key signature as number of sharps (positive) or flats (negative)

    Returns:
        (midi_number, pitch_name) e.g. (71, "B4") or (65, "F4")
    """
    ref_note_idx, ref_octave = CLEF_REF.get(clef, CLEF_REF["treble"])

    # Total diatonic steps from C0
    total_idx = ref_note_idx + position
    octave = ref_octave + total_idx // 7
    note_within = total_idx % 7

    # Handle negative positions (ledger lines below)
    if total_idx < 0:
        octave = ref_octave + (total_idx - 6) // 7
        note_within = total_idx % 7

    base_midi = (octave + 1) * 12 + DIATONIC_SEMITONES[note_within]
    note_name = NOTE_NAMES[note_within]

    # Apply key signature
    accidental = ""
    if fifths > 0:
        for i in range(min(fifths, 7)):
            if note_within == SHARP_ORDER[i]:
                base_midi += 1
                accidental = "#"
                break
    elif fifths < 0:
        for i in range(min(-fifths, 7)):
            if note_within == FLAT_ORDER[i]:
                base_midi -= 1
                accidental = "b"
                break

    pitch_name = f"{note_name}{accidental}{octave}"
    return base_midi, pitch_name


def recompute_pitches(
    notes: list[dict],
    note_positions: list[dict],
    clef: str,
    fifths: int = 0,
) -> list[dict]:
    """
    Recompute note pitches from geometric positions.

    Matches notes to note_positions by order within each staff,
    then computes pitch from position + clef.

    Args:
        notes: Parsed notes from MusicXML (with transformer pitches)
        note_positions: Pixel positions with staff_idx from segmentation
        clef: Determined clef ("treble", "bass", "alto")
        fifths: Key signature

    Returns:
        Updated notes list with corrected pitches
    """
    if not note_positions:
        return notes

    # note_positions have "position" field (staff line/space index)
    # Match by index order (both are ordered by staff, then left-to-right)
    for i, note in enumerate(notes):
        if i < len(note_positions) and "position" in note_positions[i]:
            pos = note_positions[i]["position"]
            midi, name = position_to_midi(pos, clef, fifths)
            note["pitch"] = midi
            note["pitch_name"] = name
            note["pitch_source"] = "geometric"

    return notes


def _effective_clef_at(clef_changes: list[dict], staff: int, measure: int, default: str) -> str:
    """Walk clef_changes and return the most recent clef <= (staff, measure)."""
    current = default
    for change in clef_changes:
        if change["staff"] == staff and change["measure"] <= measure:
            current = change["clef"]
    return current


def _diatonic_pitch(position: int, clef: str) -> tuple[int, int]:
    """Return (total_diatonic_idx, midi_without_accidental) for a position+clef.

    This gives the diatonic skeleton — accidentals from key sig or
    per-note `alter` are applied separately.
    """
    ref_note_idx, ref_octave = CLEF_REF.get(clef, CLEF_REF["treble"])
    total_idx = ref_note_idx + position
    if total_idx >= 0:
        octave = ref_octave + total_idx // 7
        note_within = total_idx % 7
    else:
        octave = ref_octave + (total_idx - 6) // 7
        note_within = total_idx % 7
    midi = (octave + 1) * 12 + DIATONIC_SEMITONES[note_within]
    return note_within, midi


def _score_confidence(t_midi: int, g_midi: int) -> tuple[float, int, str]:
    """Compare transformer and geometric MIDI values. Return (confidence,
    chosen_midi, reason). Lower confidence → geometric wins."""
    if t_midi == g_midi:
        return (1.0, t_midi, "exact")
    # Same pitch class (octave error)
    if t_midi % 12 == g_midi % 12:
        return (0.8, g_midi, "octave_off")
    diff = abs(t_midi - g_midi)
    if diff <= 2:
        return (0.5, g_midi, "step_error")
    if diff <= 5:
        return (0.3, g_midi, "medium_error")
    return (0.2, g_midi, "hallucination")


def recompute_pitches_with_confidence(
    notes: list[dict],
    homr_staff_buckets: dict,
    clef_changes: list[dict],
    fifths: int = 0,
    staff_clefs_default: dict | None = None,
) -> list[dict]:
    """Second-pass pitch resolution.

    For each parsed note, compute a geometric pitch from HOMR's
    per-note `position` + the effective clef for its (staff, measure).
    Compare against the transformer's pitch. Choose the final pitch
    based on a confidence score. Emit the confidence + both candidates
    alongside the chosen pitch.

    Args:
        notes: parsed notes (mutated in place; also returned)
        homr_staff_buckets: dict keyed (staff_number, measure_number)
            → list of HomrNote objects in x-order. The HomrNote objects
            must expose `.position` (int) and, optionally, `.center[1]`
            (y pixel) for chord sorting.
        clef_changes: ordered list of {staff, measure, clef} entries
            (from parse_musicxml metadata.clef_changes).
        fifths: key signature (ignored for now — we use per-note `alter`
            from transformer to preserve accidentals).
        staff_clefs_default: {staff: clef_name} initial clef fallback.

    Returns:
        The same notes list with these fields added per note:
            pitch_transformer (int) — original
            pitch_geometric (int | None) — may be None if no match
            pitch_confidence (float 0-1)
            pitch_source ("transformer" | "geometric" | "fallback")
            pitch (int) — final chosen value
    """
    if staff_clefs_default is None:
        staff_clefs_default = {}

    # --- Approach: global per-staff zip ---
    # Measure-level bucketing via barline positions is unreliable (HOMR's
    # barline detector counts system brackets, stems, etc. as barlines,
    # inflating the measure count). Instead, we sort both sides per-staff
    # by x (HOMR) / start_time (parsed) and pair them in order.
    #
    # This works because:
    # - HOMR's segmentation and the transformer see the SAME notes in the
    #   same left-to-right order.
    # - Per-staff global counts match closely (Drift Away: 133=133 for
    #   staff 1; 79 vs 75 for staff 2).
    # - When counts differ slightly (e.g. HOMR detects 4 extra noteheads),
    #   we truncate to the shorter list — a few unmatched notes at the end
    #   keep transformer pitch with lowered confidence.
    from collections import defaultdict

    # Flatten homr_buckets into per-staff sorted lists
    homr_by_staff: dict[int, list] = defaultdict(list)
    for (s, _m), bucket in homr_staff_buckets.items():
        homr_by_staff[s].extend(bucket)
    for s in homr_by_staff:
        homr_by_staff[s].sort(key=lambda h: (h["x"], h["y"]))

    # Group parsed notes by staff, preserving document order (which is
    # time-order within each staff for well-formed MusicXML).
    parsed_by_staff: dict[int, list] = defaultdict(list)
    for note in notes:
        s = note.get("staff") or 1
        parsed_by_staff[s].append(note)

    # Pre-init diagnostic fields
    for note in notes:
        note["pitch_transformer"] = int(note.get("pitch") or 0)
        note["pitch_geometric"] = None
        note["pitch_confidence"] = 1.0
        note["pitch_source"] = "transformer"

    for s in parsed_by_staff:
        parsed_list = parsed_by_staff[s]
        homr_list = homr_by_staff.get(s, [])
        n_parsed = len(parsed_list)
        n_homr = len(homr_list)
        print(f"[pitch_from_position] staff {s}: parsed={n_parsed} homr={n_homr}")

        if n_homr == 0:
            for note in parsed_list:
                note["pitch_confidence"] = 0.5
                note["pitch_source"] = "no_homr_data"
            continue

        # Nearest-x-match: for each parsed note, find the closest HOMR
        # note by x-coordinate that hasn't been claimed yet. This is
        # robust against phantom detections (handwritten notes, bass clef
        # dots misread as noteheads) — phantoms stay unclaimed and don't
        # shift subsequent pairings.
        #
        # We need parsed notes' x-positions. They don't have pixel x,
        # but they have start_time which is monotonically increasing
        # and correlates to x. HOMR notes are sorted by x. So both
        # sequences are in the same left-to-right order, and we can
        # do a greedy forward-only match: for each parsed note, scan
        # forward in the HOMR list from where we last matched, find
        # the nearest x within a tolerance. Skip HOMR notes that are
        # too far left (already passed).
        #
        # Since we don't have parsed x directly, use the HOMR ordering:
        # walk both lists with a HOMR cursor. For each parsed note,
        # advance the HOMR cursor to find the best match by allowing
        # small skips (up to 3 HOMR notes ahead) to jump over phantoms.
        homr_cursor = 0
        matched = 0
        for note in parsed_list:
            m = note.get("measure") or 1
            effective_clef = _effective_clef_at(
                clef_changes, s, m,
                default=staff_clefs_default.get(s, "treble"),
            )

            if homr_cursor >= n_homr:
                note["pitch_confidence"] = 0.5
                note["pitch_source"] = "no_homr_match"
                continue

            t_midi = int(note.get("pitch") or 0)

            # Check current HOMR note first.
            cur_pos = homr_list[homr_cursor].get("position")
            if cur_pos is not None:
                _, g_cur = _diatonic_pitch(int(cur_pos), effective_clef)
                diff_cur = abs(t_midi - g_cur)
                if t_midi % 12 == g_cur % 12:
                    diff_cur = min(diff_cur, 1)
            else:
                diff_cur = 999

            if diff_cur <= 12:
                # Current position is a reasonable match — take it.
                # Threshold 12 = one octave. Real notes typically agree
                # within an octave even when the transformer is wrong.
                # Phantoms (handwriting/clef dots) map to unrelated
                # positions → diff >> 12.
                _pair_and_score([note], [homr_list[homr_cursor]], effective_clef)
                homr_cursor += 1
            else:
                # Current position looks like a phantom (diff > 5).
                # Look ahead up to 3 to find the real note, skipping
                # the phantom(s).
                best_idx = homr_cursor
                best_diff = diff_cur
                lookahead = min(homr_cursor + 4, n_homr)
                for j in range(homr_cursor + 1, lookahead):
                    pos = homr_list[j].get("position")
                    if pos is None:
                        continue
                    _, g = _diatonic_pitch(int(pos), effective_clef)
                    diff = abs(t_midi - g)
                    if t_midi % 12 == g % 12:
                        diff = min(diff, 1)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = j
                _pair_and_score([note], [homr_list[best_idx]], effective_clef)
                skipped = best_idx - homr_cursor
                if skipped > 0:
                    print(f"[pitch_from_position] staff {s}: skipped {skipped} phantom(s) at cursor {homr_cursor}")
                homr_cursor = best_idx + 1
            matched += 1

        print(f"[pitch_from_position] staff {s}: matched {matched}/{n_parsed}")

    return notes


def _pair_and_score(parsed_notes: list, homr_notes: list, effective_clef: str) -> None:
    """Pair parsed notes with HOMR position entries (same length) and
    populate pitch_geometric / pitch_confidence / pitch fields. Preserves
    transformer accidentals when disagreement is exactly 1 semitone."""
    n = min(len(parsed_notes), len(homr_notes))
    for i in range(n):
        note = parsed_notes[i]
        homr = homr_notes[i]
        position = homr.get("position") if isinstance(homr, dict) else getattr(homr, "position", None)
        if position is None:
            note["pitch_confidence"] = 0.5
            note["pitch_source"] = "no_position"
            continue

        t_midi = int(note.get("pitch") or 0)
        _, geo_midi = _diatonic_pitch(int(position), effective_clef)
        note["pitch_geometric"] = geo_midi

        # Score confidence on RAW values (don't let accidental preservation
        # inflate the confidence to 1.0 — the reviewer caught this).
        confidence, _, reason = _score_confidence(t_midi, geo_midi)
        note["pitch_confidence"] = confidence

        if confidence == 1.0:
            # Agreement — keep transformer pitch unchanged.
            note["pitch_source"] = "agreement"
            continue

        # 1-semitone disagreement almost always means transformer has an
        # accidental that geometric (diatonic-only) lacks. Trust transformer
        # for pitch but keep the confidence score low so the uncertainty is
        # visible downstream.
        if abs(t_midi - geo_midi) == 1:
            note["pitch_source"] = f"transformer_accidental:{reason}"
            # pitch already == t_midi, no change
            continue

        # Larger disagreement: use geometric pitch.
        note["pitch"] = geo_midi
        note["pitch_source"] = f"geometric:{reason}"
        octave_new = (geo_midi // 12) - 1
        semitone = geo_midi % 12
        sharps = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        note["pitch_name"] = f"{sharps[semitone]}{octave_new}"


def determine_clef_from_positions(
    note_positions: list[dict],
) -> str:
    """
    Determine clef by computing pitches for both treble and bass,
    then checking which produces a more typical range.

    This breaks the circular dependency: note.position is from
    segmentation (correct), not from the transformer.
    """
    positions = [np["position"] for np in note_positions if "position" in np]
    if not positions:
        return "treble"

    # Compute median MIDI for both clefs
    treble_pitches = [position_to_midi(p, "treble")[0] for p in positions]
    bass_pitches = [position_to_midi(p, "bass")[0] for p in positions]

    treble_median = sorted(treble_pitches)[len(treble_pitches) // 2]
    bass_median = sorted(bass_pitches)[len(bass_pitches) // 2]

    # Typical centers:
    # Treble music: median ~68-74 (Ab4-D5)
    # Bass music: median ~46-52 (Bb2-E3)
    treble_dist = abs(treble_median - 71)  # B4 = center of treble staff
    bass_dist = abs(bass_median - 50)  # D3 = center of bass staff

    # Since treble and bass are always ~21 semitones apart for the same
    # positions, we need to look at the SPREAD of positions to break ties.
    # Notes that extend far above/below the staff (ledger lines) suggest
    # the wrong clef — correctly-cleffed music mostly stays within the staff.
    above_staff = sum(1 for p in positions if p > 9)
    below_staff = sum(1 for p in positions if p < 1)
    on_staff = sum(1 for p in positions if 1 <= p <= 9)

    # If most notes are on-staff, both clefs are plausible.
    # In that case, prefer the clef where the median is closest to center.
    if treble_dist < bass_dist:
        return "treble"
    elif bass_dist < treble_dist:
        return "bass"
    else:
        # Tie-break: treble is more common
        return "treble"
