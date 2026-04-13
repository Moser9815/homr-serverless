"""
Geometric pitch computation from staff position.

Maps note.position (staff line/space index from HOMR segmentation) + clef
to MIDI pitch. This is how Audiveris and oemer compute pitch — it's
deterministic and doesn't depend on the transformer.

HOMR position system (from find_position_in_unit_sizes):
  Positions are computed per-physical-staff BEFORE grand staff merging.
  Both treble and bass notes get positions relative to their own 5-line staff:
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

  Grand staff: positions are NOT 11-19 for treble. Both staves use the
  same 1-9 range independently. The staff_number (1=top, 2=bottom)
  determines which clef to use, NOT position remapping.

This code is licensed under AGPL-3.0 to comply with HOMR's license.
"""

# Diatonic semitone offsets within one octave: C D E F G A B
DIATONIC_SEMITONES = [0, 2, 4, 5, 7, 9, 11]
NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B"]

# Reference: for each clef, which diatonic note is at position 0.
# Position 0 = first space below the bottom line.
CLEF_REF = {
    # Treble: bottom line (pos 1) = E4, so pos 0 = D4
    "treble": (1, 4),  # (note_index_in_CDEFGAB, octave)
    # Bass: bottom line (pos 1) = G2, so pos 0 = F2
    "bass": (3, 2),
    # Alto (C clef line 3): middle line = C4, bottom line = F3,
    # so pos 0 = E3. E is index 2, octave 3.
    "alto": (2, 3),
    # Tenor (C clef line 4): bottom line = D3, pos 0 = C3.
    "tenor": (0, 3),
}

# Key signature: sharps/flats applied to specific notes
SHARP_ORDER = [3, 0, 4, 1, 5, 2, 6]  # F C G D A E B
FLAT_ORDER = [6, 2, 5, 1, 4, 0, 3]   # B E A D G C F

FIFTHS_TO_KEY = {
    -7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
}


def position_to_midi(position: int, clef: str, fifths: int = 0) -> tuple[int, str]:
    """
    Convert staff position + clef to MIDI pitch and note name.

    Args:
        position: Staff position (0-8 typical, negative for ledger lines below, >8 for above)
        clef: "treble", "bass", "alto", or "tenor"
        fifths: Key signature as number of sharps (positive) or flats (negative)

    Returns:
        (midi_number, pitch_name) e.g. (71, "B4") or (65, "F4")
    """
    ref_note_idx, ref_octave = CLEF_REF.get(clef, CLEF_REF["treble"])

    total_idx = ref_note_idx + position
    if total_idx >= 0:
        octave = ref_octave + total_idx // 7
        note_within = total_idx % 7
    else:
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


def _effective_clef_at(clef_changes: list[dict], staff: int, measure: int, default: str) -> str:
    """Return the most recent clef at or before (staff, measure)."""
    current = default
    for change in sorted(clef_changes, key=lambda c: c["measure"]):
        if change["staff"] == staff and change["measure"] <= measure:
            current = change["clef"]
    return current


def _diatonic_pitch(position: int, clef: str) -> tuple[int, int]:
    """Return (note_within_octave, midi_without_accidental) for a position+clef."""
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


def recompute_pitches_with_confidence(
    notes: list[dict],
    homr_staff_buckets: dict,
    clef_changes: list[dict],
    fifths: int = 0,
    staff_clefs_default: dict | None = None,
) -> list[dict]:
    """Second-pass pitch resolution with per-bucket matching and selective override.

    For each parsed note, find the best-matching segmentation entry in the
    same (staff, measure) bucket by pitch proximity. Then apply confidence-
    weighted selective override rules.

    Key design decisions (from 3-agent expert review):
    - Per-bucket matching prevents phantom-cascade errors (vs old cursor approach)
    - Positions are per-physical-staff (NOT merged 10-line grid) — no remapping needed
    - Clef determined from clef_changes + (staff, measure), not from position range
    - Dynamic measure fallback handles cumulative barline-count drift

    Args:
        notes: parsed notes (mutated in place; also returned)
        homr_staff_buckets: dict keyed (staff_number, measure_number)
            -> list of dicts with "position", "x", "y" in x-order.
        clef_changes: ordered list of {staff, measure, clef} entries.
        fifths: key signature (used for diatonic pitch computation).
        staff_clefs_default: {staff: clef_name} initial clef fallback.

    Returns:
        The same notes list with diagnostic fields added/updated per note.
    """
    if staff_clefs_default is None:
        staff_clefs_default = {}

    from collections import defaultdict

    # Pre-init diagnostic fields. Preserve original transformer pitch in a
    # field that is NEVER overwritten — critical for avoiding the double-pass
    # overwrite bug (expert review finding #14).
    for note in notes:
        if "pitch_original_transformer" not in note:
            note["pitch_original_transformer"] = int(note.get("pitch") or 0)
        note["pitch_transformer"] = note["pitch_original_transformer"]
        note["pitch_geometric"] = None
        if "pitch_confidence" not in note:
            note["pitch_confidence"] = 1.0
        note["pitch_source"] = "transformer"
        # Reset pitch to original transformer value before re-computing.
        # This ensures the second pass (Step 1) doesn't compare against
        # a value already modified by the first pass (Step 0b).
        note["pitch"] = note["pitch_original_transformer"]

    # Group parsed notes by (staff, measure)
    parsed_buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for note in notes:
        s = note.get("staff") or 1
        m = note.get("measure") or 1
        parsed_buckets[(s, m)].append(note)

    # Detect systematic measure offset between parsed measures and bucket
    # measures. The handler counts barlines per system to assign global_measure,
    # which can be offset from MusicXML measure numbers (e.g., pickup measures,
    # title-only first systems). If there's a consistent offset, apply it.
    measure_offset = _detect_measure_offset(parsed_buckets, homr_staff_buckets)
    if measure_offset != 0:
        print(f"[pitch_from_position] detected systematic measure offset: {measure_offset}")

    # Stats
    total_matched = 0
    total_overridden = 0
    total_flagged = 0
    total_no_bucket = 0

    for (staff, measure), parsed_list in parsed_buckets.items():
        # Look up effective clef for this (staff, measure)
        effective_clef = _effective_clef_at(
            clef_changes, staff, measure,
            default=staff_clefs_default.get(staff, "treble"),
        )

        # Find the matching homr bucket. Apply systematic offset first,
        # then try dynamic fallback for residual drift.
        bucket_entries = _find_bucket(
            homr_staff_buckets, staff, measure, measure_offset
        )

        if not bucket_entries:
            for note in parsed_list:
                note["pitch_source"] = "no_bucket"
            total_no_bucket += len(parsed_list)
            continue

        # Per-bucket matching: match parsed notes to bucket entries by
        # pitch proximity. Greedy assignment, smallest cost first.
        matches = _match_notes_to_entries(
            parsed_list, bucket_entries, effective_clef
        )

        for note, entry in matches:
            _apply_selective_override(note, entry, effective_clef)
            src = note.get("pitch_source", "")
            if "geometric" in src:
                total_overridden += 1
            elif "flagged" in src:
                total_flagged += 1
            total_matched += 1

        # Mark unmatched parsed notes
        matched_notes = {id(n) for n, _ in matches}
        for note in parsed_list:
            if id(note) not in matched_notes:
                note["pitch_source"] = "no_homr_match"

    # Log unmatched segmentation entries as quality signal (expert review suggestion)
    total_seg = sum(len(v) for v in homr_staff_buckets.values())
    total_parsed = len(notes)
    print(f"[pitch_from_position] matched={total_matched}/{total_parsed}, "
          f"overridden={total_overridden}, flagged={total_flagged}, "
          f"no_bucket={total_no_bucket}, "
          f"seg_entries={total_seg} (phantoms≈{total_seg - total_matched})")

    return notes


def _detect_measure_offset(
    parsed_buckets: dict,
    homr_staff_buckets: dict,
) -> int:
    """Detect systematic measure offset between parsed and homr buckets.

    The handler's barline-based measure counting can be systematically offset
    from MusicXML measure numbers (e.g., +2 for pieces with a title-only first
    system). Detect this by finding the offset that maximizes exact matches.

    Returns the offset to ADD to parsed measure numbers to get bucket measures.
    """
    if not parsed_buckets or not homr_staff_buckets:
        return 0

    parsed_measures = set()
    for (s, m) in parsed_buckets:
        parsed_measures.add((s, m))

    bucket_measures = set(homr_staff_buckets.keys())

    # Try offsets from -5 to +5 and count exact matches
    best_offset = 0
    best_count = 0
    for offset in range(-5, 6):
        count = sum(
            1 for (s, m) in parsed_measures
            if (s, m + offset) in bucket_measures
        )
        if count > best_count:
            best_count = count
            best_offset = offset

    # Only apply if it's clearly better than no offset
    no_offset_count = sum(
        1 for (s, m) in parsed_measures
        if (s, m) in bucket_measures
    )
    if best_offset != 0 and best_count > no_offset_count * 1.5:
        return best_offset

    return 0


def _find_bucket(
    homr_staff_buckets: dict,
    staff: int,
    measure: int,
    measure_offset: int = 0,
) -> list[dict] | None:
    """Find the best matching homr bucket for a (staff, measure).

    Applies systematic offset first, then tries nearby measures for
    residual drift from barline detection errors.
    """
    adjusted = measure + measure_offset

    # Exact match with offset
    key = (staff, adjusted)
    if key in homr_staff_buckets:
        return homr_staff_buckets[key]

    # Residual drift: search ±1, ±2
    for delta in [1, -1, 2, -2]:
        fallback_key = (staff, adjusted + delta)
        if fallback_key in homr_staff_buckets:
            return homr_staff_buckets[fallback_key]

    return None


def _match_notes_to_entries(
    parsed_list: list[dict],
    bucket_entries: list[dict],
    effective_clef: str,
) -> list[tuple[dict, dict]]:
    """Match parsed notes to segmentation entries by pitch proximity.

    Greedy assignment: compute cost matrix of |transformer_midi - geometric_midi|
    for all pairs, then assign in order of increasing cost. Each note and entry
    can only be used once. Skip any pair with cost > 12 semitones (phantom).

    Returns list of (parsed_note, bucket_entry) pairs.
    """
    # Build cost matrix
    costs = []
    for i, note in enumerate(parsed_list):
        t_midi = note.get("pitch_original_transformer") or note.get("pitch") or 0
        for j, entry in enumerate(bucket_entries):
            pos = entry.get("position")
            if pos is None:
                continue
            _, g_midi = _diatonic_pitch(int(pos), effective_clef)
            diff = abs(int(t_midi) - g_midi)
            # Reduce cost for same pitch class (octave errors are common)
            if int(t_midi) % 12 == g_midi % 12 and diff > 0:
                diff = min(diff, 2)  # Treat octave errors as small cost
            costs.append((diff, i, j))

    # Sort by cost (smallest first) and greedily assign
    costs.sort()
    used_notes = set()
    used_entries = set()
    matches = []

    for diff, i, j in costs:
        if diff > 12:
            break  # Remaining pairs are too far apart — likely phantoms
        if i in used_notes or j in used_entries:
            continue
        matches.append((parsed_list[i], bucket_entries[j]))
        used_notes.add(i)
        used_entries.add(j)

    return matches


def _apply_selective_override(note: dict, homr_entry: dict, effective_clef: str) -> None:
    """Apply confidence-weighted selective pitch override for a single note.

    Override rules:
      - Exact agreement (diff == 0): keep transformer, confidence boosted
      - 1-semitone diff: keep transformer (likely accidental)
      - Disagree >= 7 semitones: OVERRIDE with geometric
      - Disagree < 7 AND pitch_confidence < 0.5: OVERRIDE with geometric
      - Disagree < 7 AND pitch_confidence >= 0.5: FLAG but don't override
      - Preserve transformer's accidental knowledge (geometric is diatonic only)
    """
    position = homr_entry.get("position")
    if position is None:
        note["pitch_source"] = "no_position"
        return

    t_midi = int(note.get("pitch_original_transformer") or note.get("pitch") or 0)
    _, geo_midi = _diatonic_pitch(int(position), effective_clef)
    note["pitch_geometric"] = geo_midi

    diff = abs(t_midi - geo_midi)

    if diff == 0:
        note["pitch_source"] = "agreement"
        return

    if diff == 1:
        note["pitch_source"] = "transformer_accidental"
        return

    decoder_confidence = note.get("pitch_confidence", 0.9)

    if diff >= 7:
        _override_pitch(note, geo_midi, f"geometric:hallucination(diff={diff})")
    elif decoder_confidence < 0.5:
        _override_pitch(note, geo_midi,
                        f"geometric:low_conf(diff={diff},conf={decoder_confidence:.2f})")
    else:
        note["pitch_source"] = f"flagged:disagree(diff={diff},conf={decoder_confidence:.2f})"
        note["pitch_confidence"] = min(decoder_confidence, 0.7)


def _override_pitch(note: dict, geo_midi: int, reason: str) -> None:
    """Override a note's pitch with the geometric value."""
    note["pitch"] = geo_midi
    note["pitch_source"] = reason
    octave_new = (geo_midi // 12) - 1
    semitone = geo_midi % 12
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note["pitch_name"] = f"{names[semitone]}{octave_new}"
    note["pitch_confidence"] = 0.3


def determine_clef_from_positions(note_positions: list[dict]) -> str:
    """
    Determine clef by computing pitches for both treble and bass,
    then checking which produces a more typical range.
    """
    positions = [np["position"] for np in note_positions if "position" in np]
    if not positions:
        return "treble"

    treble_pitches = [position_to_midi(p, "treble")[0] for p in positions]
    bass_pitches = [position_to_midi(p, "bass")[0] for p in positions]

    treble_median = sorted(treble_pitches)[len(treble_pitches) // 2]
    bass_median = sorted(bass_pitches)[len(bass_pitches) // 2]

    treble_dist = abs(treble_median - 71)
    bass_dist = abs(bass_median - 50)

    if treble_dist < bass_dist:
        return "treble"
    elif bass_dist < treble_dist:
        return "bass"
    else:
        return "treble"
