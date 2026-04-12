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

    # --- Step 1: sanity check measure alignment per staff ---
    from collections import defaultdict
    parsed_count: dict = defaultdict(lambda: defaultdict(int))  # [staff][measure] -> count
    for n in notes:
        s = n.get("staff") or 1
        m = n.get("measure") or 1
        parsed_count[s][m] += 1

    homr_count: dict = defaultdict(lambda: defaultdict(int))
    for (s, m), bucket in homr_staff_buckets.items():
        homr_count[s][m] = len(bucket)

    # A staff is trustworthy if every measure matches on count.
    staff_trusted: dict[int, bool] = {}
    mismatches_per_staff: dict[int, list] = defaultdict(list)
    for s in set(list(parsed_count.keys()) + list(homr_count.keys())):
        trusted = True
        all_measures = set(parsed_count[s].keys()) | set(homr_count[s].keys())
        for m in sorted(all_measures):
            if parsed_count[s][m] != homr_count[s][m]:
                trusted = False
                mismatches_per_staff[s].append(
                    f"m{m}: parsed={parsed_count[s][m]} homr={homr_count[s][m]}"
                )
        staff_trusted[s] = trusted
        if not trusted:
            print(f"[pitch_from_position] staff {s} measure alignment mismatch, "
                  f"degrading confidence: {mismatches_per_staff[s][:5]}")

    # --- Step 2: group parsed notes by (staff, measure) for chord-aware pairing ---
    # Within each group, we'll walk them in sub-groups keyed by `beat` to
    # identify chords. For each chord, pair parsed notes (sorted by MIDI
    # descending — highest pitch first) against the same number of HOMR
    # notes from the bucket (sorted by y ascending — lowest y first =
    # highest pitch on the staff).
    from collections import defaultdict
    parsed_groups: dict = defaultdict(list)  # {(s, m): [note_dict, ...]}
    for note in notes:
        s = note.get("staff") or 1
        m = note.get("measure") or 1
        parsed_groups[(s, m)].append(note)

    # Pre-init the diagnostic fields on every note so callers can rely on them
    for note in notes:
        note["pitch_transformer"] = int(note.get("pitch") or 0)
        note["pitch_geometric"] = None
        note["pitch_confidence"] = 1.0
        note["pitch_source"] = "transformer"

    for (s, m), group in parsed_groups.items():
        if not staff_trusted.get(s, False):
            for note in group:
                note["pitch_confidence"] = 0.5
                note["pitch_source"] = "untrusted_staff"
            continue

        bucket = homr_staff_buckets.get((s, m)) or []
        if not bucket:
            for note in group:
                note["pitch_confidence"] = 0.5
                note["pitch_source"] = "no_homr_match"
            continue

        effective_clef = _effective_clef_at(
            clef_changes, s, m,
            default=staff_clefs_default.get(s, "treble"),
        )

        # Split into beat-based chord sub-groups, preserving document order.
        chord_subs: list = []
        current_beat = None
        for note in group:
            beat = note.get("beat", 0)
            if current_beat is None or beat != current_beat:
                chord_subs.append([note])
                current_beat = beat
            else:
                chord_subs[-1].append(note)

        # HOMR bucket entries: sort by x first, then group consecutive same-x
        # entries as chord clusters (HOMR puts chord notes at identical x).
        homr_sorted = sorted(bucket, key=lambda h: (h["x"], h["y"]))
        homr_chords: list = []
        if homr_sorted:
            current_cluster = [homr_sorted[0]]
            for h in homr_sorted[1:]:
                if abs(h["x"] - current_cluster[-1]["x"]) <= 3:  # px tolerance
                    current_cluster.append(h)
                else:
                    homr_chords.append(current_cluster)
                    current_cluster = [h]
            homr_chords.append(current_cluster)

        # Pair parsed chord sub-groups with HOMR chord clusters by position.
        # If counts mismatch, fall back to note-by-note ordinal pairing for
        # the remainder of the group.
        chord_pairs_ok = len(chord_subs) == len(homr_chords)

        if chord_pairs_ok:
            for parsed_chord, homr_cluster in zip(chord_subs, homr_chords):
                # Sort parsed notes by MIDI descending (highest pitch first)
                # and HOMR by y ascending (highest pitch first on the staff)
                parsed_sorted = sorted(parsed_chord, key=lambda n: -int(n.get("pitch") or 0))
                homr_cluster_sorted = sorted(homr_cluster, key=lambda h: h["y"])
                _pair_and_score(parsed_sorted, homr_cluster_sorted, effective_clef)
        else:
            # Fallback: ordinal pairing across the whole (s, m) bucket.
            for i, note in enumerate(group):
                if i >= len(bucket):
                    note["pitch_confidence"] = 0.5
                    note["pitch_source"] = "no_homr_match"
                    continue
                _pair_and_score([note], [bucket[i]], effective_clef)

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
