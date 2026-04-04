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
    # Alto: bottom line (pos 1) = F3, so pos 0 = E3
    # E is index 2, octave 3
    "alto": (2, 3),
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
