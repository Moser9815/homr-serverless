"""
Spatial pitch alignment — corrects transformer pitches using geometric
positions from segmentation, aligned by spatial order (not pitch).

Each segmentation notehead has (x, y, position, system_idx, sub_system,
staff_number). From position + clef we know the exact diatonic pitch.
We sort these left-to-right per (system, sub_system, staff), then align
with the transformer's note sequence using Needleman-Wunsch DP.
Phantoms and missed notes are handled as gaps.

This replaces the measure-bucket approach in pitch_from_position.py.
The bucket system failed because segmentation barline counting and
transformer MusicXML measure numbers are independent systems that
disagree (46 vs 27 measures on Drift Away). Spatial alignment doesn't
need measure numbers — both lists are in left-to-right reading order.

This code is licensed under AGPL-3.0 to comply with HOMR's license.
"""

from collections import defaultdict
from pitch_from_position import (
    position_to_midi,
    _diatonic_pitch,
    _effective_clef_at,
)


def apply_geometric_pitch(
    parsed_notes: list[dict],
    note_info: list[dict],
    clef_changes: list[dict] | None = None,
    staff_clefs_default: dict[int, str] | None = None,
    fifths: int = 0,
) -> list[dict]:
    """Correct transformer pitches using geometric positions.

    Matches segmentation noteheads to transformer notes by spatial order
    (x-coordinate vs beat position) using Needleman-Wunsch DP alignment.
    No measure bucketing — avoids the barline counting mismatch problem.

    Then applies pitch correction with:
    - Octave-aware override (keeps transformer accidentals)
    - Confidence-weighted thresholds
    - Clef changes from MusicXML metadata

    Args:
        parsed_notes: from parse_musicxml_to_json (mutated in place)
        note_info: from handler — each entry has system_idx, sub_system,
            staff_number, x, y, position
        clef_changes: list of {staff, measure, clef} from MusicXML
        staff_clefs_default: {1: "treble", 2: "bass"} initial clef per staff
        fifths: key signature

    Returns:
        parsed_notes with pitch corrections applied.
    """
    if staff_clefs_default is None:
        staff_clefs_default = {1: "treble", 2: "bass"}
    if clef_changes is None:
        clef_changes = []

    # Pre-init fields
    for n in parsed_notes:
        if "pitch_original_transformer" not in n:
            n["pitch_original_transformer"] = n["pitch"]
        n["pitch"] = n["pitch_original_transformer"]
        n["pitch_geometric"] = None
        n["pitch_source"] = "no_geo_data"

    # ---------------------------------------------------------------
    # Build geometric note list with pitch for both clefs
    # ---------------------------------------------------------------
    geo_notes = []
    for ni in note_info:
        pos = ni["position"]
        treble_midi, treble_name = position_to_midi(pos, "treble", fifths)
        bass_midi, bass_name = position_to_midi(pos, "bass", fifths)

        staff_num = ni.get("staff_number", 1)
        default_clef = staff_clefs_default.get(staff_num, "treble")
        if default_clef == "bass":
            primary_midi, primary_name = bass_midi, bass_name
        else:
            primary_midi, primary_name = treble_midi, treble_name

        geo_notes.append({
            "system": ni.get("system_idx", 0),
            "sub_system": ni.get("sub_system", 0),
            "staff": staff_num,
            "x": ni["x"],
            "y": ni["y"],
            "position": pos,
            "pitch": primary_midi,
            "pitch_name": primary_name,
            "pitch_treble": treble_midi,
            "pitch_bass": bass_midi,
            "name_treble": treble_name,
            "name_bass": bass_name,
        })

    # ---------------------------------------------------------------
    # Group geo notes by (system, sub_system, staff), sorted by x
    # ---------------------------------------------------------------
    geo_groups = defaultdict(list)
    for g in geo_notes:
        key = (g["system"], g["sub_system"], g["staff"])
        geo_groups[key].append(g)
    for key in geo_groups:
        geo_groups[key].sort(key=lambda g: (g["x"], g["y"]))

    # Build system order from geo
    system_order = []
    seen = set()
    for g in sorted(geo_notes, key=lambda g: (g["system"], g["sub_system"], g["y"])):
        key = (g["system"], g["sub_system"])
        if key not in seen:
            system_order.append(key)
            seen.add(key)

    print(f"[spatial_align] system_order: {system_order}")
    for row in system_order:
        counts = {}
        for (sys, sub, staff), nl in geo_groups.items():
            if (sys, sub) == row:
                counts[staff] = len(nl)
        print(f"  row {row}: {counts}")

    # Group transformer notes by staff, sorted by (measure, beat)
    trans_by_staff = defaultdict(list)
    for n in parsed_notes:
        staff = n.get("staff") or 1
        trans_by_staff[staff].append(n)
    for staff in trans_by_staff:
        trans_by_staff[staff].sort(key=lambda n: (n["measure"], n["beat"]))

    # ---------------------------------------------------------------
    # Align per staff: one pass through the full sequence
    # ---------------------------------------------------------------
    stats = {"matched": 0, "agreement": 0, "overridden": 0,
             "accidental": 0, "flagged": 0, "phantom": 0, "missed": 0,
             "octave_fix": 0}

    for staff in sorted(trans_by_staff.keys()):
        trans_list = trans_by_staff[staff]

        # Build full geo sequence for this staff, sorted by row then x
        full_geo = []
        for (sys, sub) in system_order:
            full_geo.extend(geo_groups.get((sys, sub, staff), []))

        if not full_geo:
            for n in trans_list:
                n["pitch_source"] = "no_geo_for_staff"
            stats["missed"] += len(trans_list)
            continue

        _align_and_correct(
            full_geo, trans_list, clef_changes,
            staff, staff_clefs_default, stats,
        )

    total = len(parsed_notes)
    print(f"[spatial_align] matched={stats['matched']}/{total}, "
          f"agreement={stats['agreement']}, overridden={stats['overridden']}, "
          f"octave_fix={stats['octave_fix']}, "
          f"accidental={stats['accidental']}, flagged={stats['flagged']}, "
          f"phantom={stats['phantom']}, missed={stats['missed']}")

    return parsed_notes


def _align_and_correct(
    geo_list: list[dict],
    trans_list: list[dict],
    clef_changes: list[dict],
    staff: int,
    staff_clefs_default: dict,
    stats: dict,
):
    """Align geo and trans sequences using DP, then apply corrections.

    Two-phase approach:
    1. Group both sequences into chord groups (same x / same beat)
    2. Align chord groups using Needleman-Wunsch DP (spatial order)
    3. Within each aligned chord pair, match notes by y-order
       (top-to-bottom on page = highest pitch first)
    4. Apply pitch correction to each matched pair

    This prevents the chord-swapping problem where the DP matches
    the top note of a chord to the bottom entry and vice versa.
    """
    # Phase 1: Group into chord groups
    geo_groups = _group_by_x(geo_list, threshold=15)
    trans_groups = _group_by_beat(trans_list)

    n = len(geo_groups)
    m = len(trans_groups)

    if n == 0 or m == 0:
        for t in trans_list:
            t["pitch_source"] = "no_geo_match"
            stats["missed"] += 1
        return

    GAP_PENALTY = 4

    # Build cost matrix using spatial proportionality of chord groups
    # Use the first note's x / beat as the group's position
    group_xs = [g[0]["x"] for g in geo_groups]
    group_beats = [t[0].get("beat", 0) + (t[0].get("measure", 1) - 1) * 100
                   for t in trans_groups]

    min_x = min(group_xs)
    max_x = max(group_xs)
    x_range = max(max_x - min_x, 1.0)

    min_beat = min(group_beats)
    max_beat = max(group_beats)
    beat_range = max(max_beat - min_beat, 1.0)

    match_cost = [[0.0] * m for _ in range(n)]
    for i in range(n):
        norm_x = (group_xs[i] - min_x) / x_range
        for j in range(m):
            norm_beat = (group_beats[j] - min_beat) / beat_range
            cost = abs(norm_x - norm_beat) * GAP_PENALTY * 1.5
            match_cost[i][j] = min(cost, GAP_PENALTY - 0.01)

    # DP alignment of chord groups
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * GAP_PENALTY
    for j in range(1, m + 1):
        dp[0][j] = j * GAP_PENALTY

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i-1][j-1] + match_cost[i-1][j-1],
                dp[i-1][j] + GAP_PENALTY,
                dp[i][j-1] + GAP_PENALTY,
            )

    # Traceback
    i, j = n, m
    group_alignment = []
    while i > 0 or j > 0:
        if (i > 0 and j > 0
                and abs(dp[i][j] - (dp[i-1][j-1] + match_cost[i-1][j-1])) < 1e-9):
            group_alignment.append((i-1, j-1))
            i -= 1
            j -= 1
        elif i > 0 and abs(dp[i][j] - (dp[i-1][j] + GAP_PENALTY)) < 1e-9:
            i -= 1  # phantom chord group
            stats["phantom"] += len(geo_groups[i])
        else:
            j -= 1  # missed chord group
            for t in trans_groups[j]:
                t["pitch_source"] = "no_geo_match"
            stats["missed"] += len(trans_groups[j])

    group_alignment.reverse()

    # Phase 2: Within each aligned chord pair, match by y-order
    for gi, ti in group_alignment:
        geo_chord = geo_groups[gi]
        trans_chord = trans_groups[ti]

        # Sort geo by y ascending (top of page = lowest y = highest pitch)
        geo_sorted = sorted(geo_chord, key=lambda g: g["y"])
        # Sort trans by pitch descending (highest pitch first)
        trans_sorted = sorted(trans_chord, key=lambda t: -(t.get("pitch_original_transformer") or t.get("pitch") or 0))

        # Match by position in sorted order
        pairs = min(len(geo_sorted), len(trans_sorted))
        for k in range(pairs):
            geo_note = geo_sorted[k]
            trans_note = trans_sorted[k]

            measure = trans_note.get("measure", 1)
            effective_clef = _effective_clef_at(
                clef_changes, staff, measure,
                default=staff_clefs_default.get(staff, "treble"),
            )
            _, geo_midi = _diatonic_pitch(int(geo_note["position"]), effective_clef)
            _apply_selective_override(trans_note, geo_midi, stats)

        # Unmatched extras
        if len(geo_sorted) > pairs:
            stats["phantom"] += len(geo_sorted) - pairs
        if len(trans_sorted) > pairs:
            for k in range(pairs, len(trans_sorted)):
                trans_sorted[k]["pitch_source"] = "no_geo_in_chord"
            stats["missed"] += len(trans_sorted) - pairs


def _group_by_x(notes, threshold=15):
    """Group notes with similar x into chord groups."""
    if not notes:
        return []
    groups = []
    current = [notes[0]]
    for n in notes[1:]:
        if abs(n["x"] - current[0]["x"]) <= threshold:
            current.append(n)
        else:
            groups.append(current)
            current = [n]
    groups.append(current)
    return groups


def _group_by_beat(notes):
    """Group transformer notes by (measure, beat)."""
    if not notes:
        return []
    groups = []
    current = [notes[0]]
    for n in notes[1:]:
        if (n["measure"] == current[0]["measure"]
                and n["beat"] == current[0]["beat"]):
            current.append(n)
        else:
            groups.append(current)
            current = [n]
    groups.append(current)
    return groups


def _apply_selective_override(
    note: dict, geo_midi: int, stats: dict,
) -> None:
    """Apply confidence-weighted selective pitch override.

    Ported from pitch_from_position.py with full logic:
    - Exact agreement
    - 1-semitone accidental preservation
    - Octave-aware override (keeps transformer pitch class + geometric octave)
    - Large diff (>=7) hallucination override
    - Confidence-weighted override for moderate diffs
    - Flagging for moderate diffs with high confidence
    """
    t_midi = int(note.get("pitch_original_transformer") or note.get("pitch") or 0)
    note["pitch_geometric"] = geo_midi

    diff = abs(t_midi - geo_midi)

    if diff == 0:
        note["pitch_source"] = "agreement"
        stats["agreement"] += 1
        stats["matched"] += 1
        return

    if diff == 1:
        note["pitch_source"] = "transformer_accidental"
        stats["accidental"] += 1
        stats["matched"] += 1
        return

    decoder_confidence = note.get("pitch_confidence", 0.9)

    # Octave-aware override: when transformer has right pitch class
    # but wrong octave, keep transformer's accidental + geometric's octave
    if diff >= 11:
        pitch_class_diff = abs(t_midi % 12 - geo_midi % 12)
        pitch_class_diff = min(pitch_class_diff, 12 - pitch_class_diff)
        if pitch_class_diff <= 1:
            new_midi = (geo_midi // 12) * 12 + (t_midi % 12)
            _override_pitch(
                note, new_midi,
                f"octave_fix:trans_class+geo_octave(diff={diff},t={t_midi},g={geo_midi})"
            )
            stats["octave_fix"] += 1
            stats["matched"] += 1
            return

    if diff >= 7:
        _override_pitch(note, geo_midi, f"geometric:hallucination(diff={diff})")
        stats["overridden"] += 1
    elif decoder_confidence < 0.5:
        _override_pitch(note, geo_midi,
                        f"geometric:low_conf(diff={diff},conf={decoder_confidence:.2f})")
        stats["overridden"] += 1
    else:
        note["pitch_source"] = f"flagged:disagree(diff={diff},conf={decoder_confidence:.2f})"
        note["pitch_confidence"] = min(decoder_confidence, 0.7)
        stats["flagged"] += 1

    stats["matched"] += 1


def _override_pitch(note: dict, new_midi: int, reason: str) -> None:
    """Override a note's pitch, recomputing the name from MIDI."""
    note["pitch"] = new_midi
    note["pitch_source"] = reason
    octave = (new_midi // 12) - 1
    semitone = new_midi % 12
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note["pitch_name"] = f"{names[semitone]}{octave}"
    note["pitch_confidence"] = 0.3
