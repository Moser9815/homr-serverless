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
    """Second-pass pitch resolution with spatial matching and selective override.

    For each parsed note, find the best-matching segmentation entry in the
    same (staff, measure) bucket by spatial position order (beat vs x-coordinate).
    Then apply confidence-weighted selective override rules.

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
        # spatial position order (beat-to-x proportionality). No pitch used.
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
    """Match parsed notes to segmentation entries by spatial position order.

    Within a (staff, measure) bucket, both lists should be in left-to-right
    order. We match by position order, NOT by pitch proximity, to break the
    circularity of using transformer pitch to match against the data that
    is supposed to correct transformer pitch.

    Approach:
    1. Sort parsed notes by (beat, -pitch) — left-to-right, high-to-low for chords
    2. Sort bucket entries by (x, y) — left-to-right, top-to-bottom for chords
    3. Use Needleman-Wunsch DP with zero match cost to align in order
    4. The DP handles length mismatches by inserting gaps (phantom/missed)

    The key insight: both lists are already in the SAME spatial order
    (left-to-right). We just need to handle length differences (phantoms
    in entries, missed notes in parsed) without using pitch for matching.

    Returns list of (parsed_note, bucket_entry) pairs.
    """
    if not parsed_list or not bucket_entries:
        return []

    # Filter out entries with no position data
    valid_entries = [e for e in bucket_entries if e.get("position") is not None]
    if not valid_entries:
        return []

    # Sort parsed notes by (beat, descending pitch) — chords: highest pitch first
    sorted_parsed = sorted(
        enumerate(parsed_list),
        key=lambda ip: (
            ip[1].get("beat", 0),
            -(ip[1].get("pitch_original_transformer") or ip[1].get("pitch") or 0),
        ),
    )

    # Sort entries by (x, y) — left to right, top to bottom within a chord
    sorted_entries = sorted(
        enumerate(valid_entries),
        key=lambda ie: (ie[1].get("x", 0), ie[1].get("y", 0)),
    )

    n_parsed = len(sorted_parsed)
    n_entries = len(sorted_entries)

    # --- Needleman-Wunsch DP alignment with spatial proportionality ---
    # Both lists are in the same left-to-right order. We use the proportional
    # position (beat within beat range vs x within x range) to distinguish
    # which entries should match which parsed notes. NO pitch is used.
    #
    # The cost of matching parsed[i] to entry[j] is based on how far apart
    # their normalized positions are. This lets the DP correctly skip
    # phantom entries that don't correspond to any parsed note.
    GAP_PENALTY = 2  # Cost of skipping (phantom or missed)

    # Extract beat and x ranges for normalization
    beats = [sorted_parsed[i][1].get("beat", 0) for i in range(n_parsed)]
    xs = [sorted_entries[j][1].get("x", 0) for j in range(n_entries)]

    min_beat = min(beats) if beats else 0
    max_beat = max(beats) if beats else 1
    beat_range = max_beat - min_beat if max_beat > min_beat else 1.0

    min_x = min(xs) if xs else 0
    max_x = max(xs) if xs else 1
    x_range = max_x - min_x if max_x > min_x else 1.0

    # Pre-compute match costs
    match_costs = [[0.0] * n_entries for _ in range(n_parsed)]
    for i in range(n_parsed):
        norm_beat = (beats[i] - min_beat) / beat_range
        for j in range(n_entries):
            norm_x = (xs[j] - min_x) / x_range
            # Cost = proportional distance, scaled to be comparable to gap penalty
            # Small proportional distance = good match, large = bad match
            cost = abs(norm_beat - norm_x) * GAP_PENALTY * 1.5
            match_costs[i][j] = min(cost, GAP_PENALTY - 0.01)  # Cap below gap penalty

    # DP table: dp[i][j] = min cost to align parsed[0:i] with entries[0:j]
    dp = [[0.0] * (n_entries + 1) for _ in range(n_parsed + 1)]
    for i in range(1, n_parsed + 1):
        dp[i][0] = i * GAP_PENALTY
    for j in range(1, n_entries + 1):
        dp[0][j] = j * GAP_PENALTY

    for i in range(1, n_parsed + 1):
        for j in range(1, n_entries + 1):
            dp[i][j] = min(
                dp[i-1][j-1] + match_costs[i-1][j-1],  # Match
                dp[i-1][j] + GAP_PENALTY,                # Skip parsed (missed)
                dp[i][j-1] + GAP_PENALTY,                # Skip entry (phantom)
            )

    # Traceback
    i, j = n_parsed, n_entries
    alignment = []
    while i > 0 or j > 0:
        if (i > 0 and j > 0 and
                abs(dp[i][j] - (dp[i-1][j-1] + match_costs[i-1][j-1])) < 1e-9):
            alignment.append((i-1, j-1))
            i -= 1
            j -= 1
        elif i > 0 and abs(dp[i][j] - (dp[i-1][j] + GAP_PENALTY)) < 1e-9:
            i -= 1  # Skip parsed
        else:
            j -= 1  # Skip entry (phantom)

    alignment.reverse()

    # Convert alignment back to original indices
    matches = []
    for pi, ej in alignment:
        _, parsed_note = sorted_parsed[pi]
        _, entry = sorted_entries[ej]
        matches.append((parsed_note, entry))

    return matches


def _apply_selective_override(note: dict, homr_entry: dict, effective_clef: str) -> None:
    """Apply confidence-weighted selective pitch override for a single note.

    Override rules (checked in order):
      - Exact agreement (diff == 0): keep transformer, confidence boosted
      - 1-semitone diff: keep transformer (likely accidental)
      - Octave error (diff >= 11, pitch class within 1): keep transformer's
        pitch class + geometric's octave (preserves accidentals)
      - Disagree >= 7 semitones: OVERRIDE with geometric
      - Disagree < 7 AND pitch_confidence < 0.5: OVERRIDE with geometric
      - Disagree < 7 AND pitch_confidence >= 0.5: FLAG but don't override
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

    # --- Step 4: Octave-aware override ---
    # When the transformer has the right pitch class (note name + accidental)
    # but wrong octave, keep the transformer's pitch class and use the
    # geometric pitch's octave. The geometric system is diatonic-only, so
    # the pitch classes may differ by up to 1 semitone (e.g., Eb vs D or E).
    # Check: diff >= 11 AND pitch class difference <= 1 semitone.
    if diff >= 11:
        pitch_class_diff = abs(t_midi % 12 - geo_midi % 12)
        pitch_class_diff = min(pitch_class_diff, 12 - pitch_class_diff)
        if pitch_class_diff <= 1:
            # Octave error: keep transformer's pitch class, use geometric's octave
            new_midi = (geo_midi // 12) * 12 + (t_midi % 12)
            _override_pitch(
                note, new_midi,
                f"octave_fix:trans_class+geo_octave(diff={diff},t={t_midi},g={geo_midi})"
            )
            return

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
