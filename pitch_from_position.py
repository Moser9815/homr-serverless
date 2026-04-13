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


# recompute_pitches() DELETED — was a naive global-zip approach that broke
# on phantom noteheads. Use recompute_pitches_with_confidence() instead.
# (Deleted in Phase 1, Step 1.2 of the OMR Accuracy Improvements plan.)


def _effective_clef_at(clef_changes: list[dict], staff: int, measure: int, default: str) -> str:
    """Return the most recent clef at or before (staff, measure)."""
    current = default
    for change in sorted(clef_changes, key=lambda c: c["measure"]):
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


def remap_grand_staff_position(position: int, staff_number: int) -> tuple[int, str]:
    """Remap a merged grand-staff position to a per-physical-staff position.

    HOMR's 10-line merged grid assigns:
      positions 1-9  → bottom physical staff (staff_number 2 = bass)
      positions 11-19 → top physical staff (staff_number 1 = treble)
      position 10 → gap between staves
      positions < 1 or > 19 → ledger lines (pass through)

    Args:
        position: Raw position from HOMR segmentation (merged grid)
        staff_number: 1 = top staff, 2 = bottom staff

    Returns:
        (remapped_position, clef_hint) where remapped_position is
        relative to a single 5-line staff (1-9), and clef_hint is
        the expected clef for that physical staff.
    """
    if staff_number == 2:
        # Bottom staff — positions 1-9 are already relative to the
        # bottom physical staff. No remapping needed.
        return (position, "bass")
    elif staff_number == 1:
        # Top staff — positions 11-19 on the merged grid correspond
        # to positions 1-9 on the top physical staff.
        if position >= 11:
            return (position - 10, "treble")
        elif position == 10:
            # Gap — ledger line above bottom staff / below top staff
            return (0, "treble")
        else:
            # Note is in the bottom half of the grid but assigned to
            # the top staff by y-coordinate. The physical position is
            # in bass territory — use bass clef for pitch computation
            # regardless of the y-coordinate classifier's opinion.
            return (position, "bass")
    else:
        return (position, "treble")


def recompute_pitches_with_confidence(
    notes: list[dict],
    homr_staff_buckets: dict,
    clef_changes: list[dict],
    fifths: int = 0,
    staff_clefs_default: dict | None = None,
) -> list[dict]:
    """Second-pass pitch resolution with confidence-weighted selective override.

    For each parsed note, compute a geometric pitch from HOMR's
    per-note `position` + the effective clef for its (staff, measure).
    Compare against the transformer's pitch. Override ONLY when:
      - Disagreement >= 7 semitones (obvious hallucination), OR
      - Disagreement < 7 AND pitch_confidence < 0.5 (transformer uncertain)
    Otherwise, keep transformer pitch and flag the disagreement.

    Includes grand staff position remapping for merged 10-line staves.

    Args:
        notes: parsed notes (mutated in place; also returned)
        homr_staff_buckets: dict keyed (staff_number, measure_number)
            -> list of dicts with "position", "x", "y" in x-order.
        clef_changes: ordered list of {staff, measure, clef} entries
            (from parse_musicxml metadata.clef_changes).
        fifths: key signature (used for diatonic pitch computation).
        staff_clefs_default: {staff: clef_name} initial clef fallback.

    Returns:
        The same notes list with these fields added/updated per note:
            pitch_transformer (int) -- original
            pitch_geometric (int | None) -- may be None if no match
            pitch_confidence (float 0-1) -- overall confidence
            pitch_source ("transformer" | "geometric" | ...)
            pitch (int) -- final chosen value
    """
    if staff_clefs_default is None:
        staff_clefs_default = {}

    from collections import defaultdict

    # Detect grand staff: if any bucket key has staff_number 2, it's grand staff
    has_grand_staff = any(s == 2 for (s, _) in homr_staff_buckets.keys())

    # Flatten homr_buckets into per-staff sorted lists, applying
    # grand staff position remapping if needed.
    homr_by_staff: dict[int, list] = defaultdict(list)
    for (s, _m), bucket in homr_staff_buckets.items():
        for h in bucket:
            entry = dict(h)  # shallow copy
            if has_grand_staff and "position" in entry:
                remapped_pos, clef_hint = remap_grand_staff_position(
                    int(entry["position"]), s
                )
                entry["position_original"] = entry["position"]
                entry["position"] = remapped_pos
                entry["clef_hint"] = clef_hint
            homr_by_staff[s].append(entry)
    for s in homr_by_staff:
        homr_by_staff[s].sort(key=lambda h: (h["x"], h["y"]))

    # Group parsed notes by staff, preserving document order (which is
    # time-order within each staff for well-formed MusicXML).
    parsed_by_staff: dict[int, list] = defaultdict(list)
    for note in notes:
        s = note.get("staff") or 1
        parsed_by_staff[s].append(note)

    # Pre-init diagnostic fields (preserve any pitch_confidence already
    # set by the confidence monkey-patch)
    for note in notes:
        note["pitch_transformer"] = int(note.get("pitch") or 0)
        note["pitch_geometric"] = None
        if "pitch_confidence" not in note:
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
                note["pitch_source"] = "no_homr_data"
            continue

        # Greedy cursor-matching: walk both lists in x-order.
        # For each parsed note, advance the HOMR cursor allowing small
        # skips (up to 3) to jump over phantom noteheads.
        homr_cursor = 0
        matched = 0
        overridden = 0
        flagged = 0
        for note in parsed_list:
            m = note.get("measure") or 1

            # For grand staff, use the clef hint from remapping if available
            if has_grand_staff and homr_cursor < n_homr:
                effective_clef = homr_list[homr_cursor].get(
                    "clef_hint",
                    _effective_clef_at(clef_changes, s, m,
                                       default=staff_clefs_default.get(s, "treble")),
                )
            else:
                effective_clef = _effective_clef_at(
                    clef_changes, s, m,
                    default=staff_clefs_default.get(s, "treble"),
                )

            if homr_cursor >= n_homr:
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
                _apply_selective_override(note, homr_list[homr_cursor], effective_clef)
                homr_cursor += 1
            else:
                # Current position looks like a phantom. Look ahead up
                # to 3 to find the real note, skipping phantom(s).
                best_idx = homr_cursor
                best_diff = diff_cur
                lookahead = min(homr_cursor + 4, n_homr)
                for j in range(homr_cursor + 1, lookahead):
                    pos = homr_list[j].get("position")
                    if pos is None:
                        continue
                    eff_clef_j = homr_list[j].get("clef_hint", effective_clef)
                    _, g = _diatonic_pitch(int(pos), eff_clef_j)
                    diff = abs(t_midi - g)
                    if t_midi % 12 == g % 12:
                        diff = min(diff, 1)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = j
                _apply_selective_override(note, homr_list[best_idx],
                                          homr_list[best_idx].get("clef_hint", effective_clef))
                skipped = best_idx - homr_cursor
                if skipped > 0:
                    print(f"[pitch_from_position] staff {s}: skipped {skipped} phantom(s) at cursor {homr_cursor}")
                homr_cursor = best_idx + 1
            matched += 1

            # Count outcomes
            src = note.get("pitch_source", "")
            if "geometric" in src:
                overridden += 1
            elif "flagged" in src:
                flagged += 1

        print(f"[pitch_from_position] staff {s}: matched {matched}/{n_parsed}, "
              f"overridden={overridden}, flagged={flagged}")

    return notes


def _apply_selective_override(note: dict, homr_entry: dict, effective_clef: str) -> None:
    """Apply confidence-weighted selective pitch override for a single note.

    Override rules (from the plan):
      - Exact agreement (diff == 0): keep transformer, confidence = 1.0
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

    t_midi = int(note.get("pitch") or 0)
    _, geo_midi = _diatonic_pitch(int(position), effective_clef)
    note["pitch_geometric"] = geo_midi

    diff = abs(t_midi - geo_midi)

    if diff == 0:
        # Exact agreement — no change needed
        note["pitch_source"] = "agreement"
        return

    if diff == 1:
        # 1-semitone disagreement: transformer likely has an accidental
        # that geometric (diatonic-only) can't see. Trust transformer.
        note["pitch_source"] = "transformer_accidental"
        return

    # Get the decoder's pitch confidence (set by monkey-patch)
    # If not set (legacy data), default to 0.9 (trust transformer)
    decoder_confidence = note.get("pitch_confidence", 0.9)

    if diff >= 7:
        # Large disagreement (>= perfect fifth): override with geometric.
        # This catches Bug #20 territory (sub-octave hallucinations).
        _override_pitch(note, geo_midi, f"geometric:hallucination(diff={diff})")
    elif decoder_confidence < 0.5:
        # Moderate disagreement + low decoder confidence: override.
        _override_pitch(note, geo_midi,
                        f"geometric:low_conf(diff={diff},conf={decoder_confidence:.2f})")
    else:
        # Moderate disagreement + confident decoder: flag but don't override.
        # The decoder is reasonably sure, and the disagreement isn't extreme.
        note["pitch_source"] = f"flagged:disagree(diff={diff},conf={decoder_confidence:.2f})"
        # Keep transformer pitch — only update confidence to reflect uncertainty
        note["pitch_confidence"] = min(decoder_confidence, 0.7)


def _override_pitch(note: dict, geo_midi: int, reason: str) -> None:
    """Override a note's pitch with the geometric value."""
    note["pitch"] = geo_midi
    note["pitch_source"] = reason
    # Recalculate pitch name from MIDI
    octave_new = (geo_midi // 12) - 1
    semitone = geo_midi % 12
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note["pitch_name"] = f"{names[semitone]}{octave_new}"
    # Set confidence low to signal that correction was applied
    note["pitch_confidence"] = 0.3


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
