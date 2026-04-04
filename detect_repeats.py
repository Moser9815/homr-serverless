"""
Repeat Barline Classifier — Detects forward/backward repeat barlines
by finding the two-dot pattern in the image near each barline bounding box.

HOMR's segmentation reliably finds all barlines (59 on Can-Can) with pixel
coordinates, but the transformer misclassifies repeat barlines (~50% miss rate).
This module examines the image directly to classify barlines by looking for the
distinctive two-dot pattern that distinguishes repeats from normal barlines.

Repeat barline anatomy:
  - Dots on LEFT of thick bar → backward repeat (end)
  - Dots on RIGHT of thick bar → forward repeat (start)
  - Dots sit in spaces 2 and 3 of the staff (center ± 0.5 * unit_size)

Approach: For each of the 59 barline bounding boxes, search a wide strip on
each side for the two-dot pattern. HOMR sometimes splits a repeat structure
into two boxes (thick + thin bar), so both boxes in a pair will detect the
same dots — we deduplicate when building repeat markers.

This code is licensed under AGPL-3.0 to comply with HOMR's license.
"""

import cv2
import numpy as np


def detect_repeat_barlines(
    image_path: str,
    barline_info: list[dict],
    staff_info: list[dict],
    total_measures: int = 0,
    notes: list[dict] | None = None,
    note_positions: list[dict] | None = None,
    rests: list[dict] | None = None,
    rest_positions: list[dict] | None = None,
    debug: bool = False,
) -> list[dict]:
    """
    Classify each barline as normal, forward repeat, or backward repeat.

    Args:
        image_path: Path to the original sheet music image
        barline_info: List of barline dicts from HOMR segmentation
            Each has: staff_idx, x (center), y (center), width, height
        staff_info: List of staff dicts from HOMR
            Each has: staff, min_x, max_x, min_y, max_y, unit_size
        total_measures: Expected total measures (from MusicXML, for validation)
        notes: Notes from MusicXML with measure numbers (for measure assignment)
        note_positions: Note pixel positions from segmentation (paired 1:1 with notes)
        debug: Print diagnostic info

    Returns:
        List of barline dicts with added fields:
        - 'type': 'normal', 'forward', or 'backward'
        - 'left_score': dot density score on left side
        - 'right_score': dot density score on right side
        - 'measure_before': measure number to the left (or None if staff edge)
        - 'measure_after': measure number to the right (or None if staff edge)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[detect_repeats] ERROR: Could not load {image_path}")
        return [{**bl, "type": "normal"} for bl in barline_info]

    # Binarize: dark pixels become 255 (white on black)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Build staff lookup
    staff_lookup = {s["staff"]: s for s in staff_info}

    # Identify edge barlines (first/last on each staff)
    edge_set = _find_edge_barlines(barline_info)

    # Score and classify each barline
    classified = []
    for bl in barline_info:
        staff = staff_lookup.get(bl["staff_idx"])
        if not staff:
            classified.append({**bl, "type": "normal", "left_score": 0, "right_score": 0})
            continue

        unit = staff["unit_size"]
        staff_center_y = (staff["min_y"] + staff["max_y"]) / 2

        left_score = _dot_pair_score(
            binary, bl, staff_center_y, unit, side="left"
        )
        right_score = _dot_pair_score(
            binary, bl, staff_center_y, unit, side="right"
        )

        is_edge = id(bl) in edge_set
        bl_type = _classify(left_score, right_score, is_edge=is_edge)

        result = {**bl, "type": bl_type, "left_score": round(left_score, 3), "right_score": round(right_score, 3)}
        classified.append(result)

        if debug and bl_type != "normal":
            edge_tag = " (edge)" if is_edge else ""
            print(
                f"[detect_repeats] {bl_type.upper()} barline at x={bl['x']:.0f} "
                f"staff={bl['staff_idx']} "
                f"L={left_score:.3f} R={right_score:.3f} "
                f"w={bl['width']:.1f} unit={unit:.1f}{edge_tag}"
            )

    # Assign measure numbers using note+rest data (accurate) or barline counting (fallback)
    _assign_measure_numbers(classified, staff_info, total_measures, notes, note_positions, rests, rest_positions, debug)

    if debug:
        repeats = [b for b in classified if b["type"] != "normal"]
        print(f"[detect_repeats] Found {len(repeats)} repeat barlines out of {len(classified)}")

    return classified


def _find_edge_barlines(barline_info: list[dict]) -> set[int]:
    """Return set of id()s for the first and last barline on each staff."""
    staff_groups: dict[int, list[dict]] = {}
    for bl in barline_info:
        idx = bl["staff_idx"]
        if idx not in staff_groups:
            staff_groups[idx] = []
        staff_groups[idx].append(bl)

    edge_ids = set()
    for barlines in staff_groups.values():
        by_x = sorted(barlines, key=lambda b: b["x"])
        edge_ids.add(id(by_x[0]))
        edge_ids.add(id(by_x[-1]))
    return edge_ids


def _dot_pair_score(
    binary: np.ndarray,
    barline: dict,
    staff_center_y: float,
    unit: float,
    side: str,
) -> float:
    """
    Score the presence of two dots at the expected positions on one side
    of a barline using connected component analysis.

    Instead of measuring raw pixel density (which picks up notes, stems,
    flags), we find actual dot-shaped objects: small, roughly circular
    connected components at the expected vertical positions.

    The dots sit in spaces 2 and 3 of the staff:
      dot_upper = staff_center - 0.5 * unit_size
      dot_lower = staff_center + 0.5 * unit_size

    Returns:
      1.0 = two dots found at both expected positions
      0.5 = one dot found
      0.0 = no dots
    """
    h, w = binary.shape

    # Expected dot center positions (spaces 2 and 3)
    dot_y_upper = staff_center_y - 0.5 * unit
    dot_y_lower = staff_center_y + 0.5 * unit

    # Search region: a vertical strip on one side of the barline,
    # spanning both dot positions with generous vertical margin.
    cx = barline["x"]
    gap = unit * 0.25    # skip past the barline line itself
    reach = unit * 1.2   # search far enough for paired boxes

    if side == "left":
        x1 = max(0, int(cx - gap - reach))
        x2 = max(0, int(cx - gap))
    else:
        x1 = min(w, int(cx + gap))
        x2 = min(w, int(cx + gap + reach))

    # Vertical: cover both dot positions with margin
    y1 = max(0, int(dot_y_upper - unit * 0.5))
    y2 = min(h, int(dot_y_lower + unit * 0.5))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    # Extract search region
    region = binary[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0

    # Find connected components in the search region
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        region, connectivity=8
    )

    # Expected dot size: ~0.3 * unit diameter → area ≈ π * (0.15*unit)²
    min_dot_area = (0.08 * unit) ** 2   # very small threshold
    max_dot_area = (0.45 * unit) ** 2   # dots are small
    max_aspect = 2.5  # dots are roughly circular

    dots_found = 0
    for label_idx in range(1, num_labels):  # skip background (0)
        area = stats[label_idx, cv2.CC_STAT_AREA]
        comp_w = stats[label_idx, cv2.CC_STAT_WIDTH]
        comp_h = stats[label_idx, cv2.CC_STAT_HEIGHT]
        comp_cy = centroids[label_idx][1] + y1  # convert back to image coords

        # Size filter: must be dot-sized
        if area < min_dot_area or area > max_dot_area:
            continue

        # Shape filter: roughly circular (not a stem or flag)
        if comp_w == 0 or comp_h == 0:
            continue
        aspect = max(comp_w, comp_h) / min(comp_w, comp_h)
        if aspect > max_aspect:
            continue

        # Position filter: must be near one of the expected dot Y positions
        dist_upper = abs(comp_cy - dot_y_upper)
        dist_lower = abs(comp_cy - dot_y_lower)
        if min(dist_upper, dist_lower) > unit * 0.4:
            continue

        dots_found += 1

    # Score: need 2 dots for a repeat barline
    if dots_found >= 2:
        return 1.0
    elif dots_found == 1:
        return 0.5
    return 0.0


def _classify(left_score: float, right_score: float, is_edge: bool = False) -> str:
    """
    Classify a barline based on dot-pair scores from connected component analysis.

    Scores are now clean:
      1.0 = two dot-shaped objects found at expected positions
      0.5 = one dot found (partial detection)
      0.0 = no dots

    A repeat barline has dots on exactly one side.
    """
    # Need at least one dot-pair (score >= 0.5) on at least one side
    if left_score < 0.5 and right_score < 0.5:
        return "normal"

    # Dots on one side only → clear repeat
    if left_score >= 0.5 and right_score < 0.5:
        return "backward"
    if right_score >= 0.5 and left_score < 0.5:
        return "forward"

    # Dots on both sides → the stronger side wins (one detection may
    # be picking up non-dot components on the other side)
    if left_score > right_score:
        return "backward"
    return "forward"


def _assign_measure_numbers(
    classified: list[dict],
    staff_info: list[dict],
    total_measures: int,
    notes: list[dict] | None,
    note_positions: list[dict] | None,
    rests: list[dict] | None,
    rest_positions: list[dict] | None,
    debug: bool,
) -> None:
    """
    Assign measure numbers to each barline.

    Primary method: combine notes + rests from MusicXML (with measure numbers)
    and segmentation (with staff_idx). Count elements per measure and per staff,
    then greedily fit measures to staves.

    Fallback: count barlines per staff if element data is unavailable.
    """
    # Combine notes and rests into unified element lists
    all_elements = []  # from MusicXML: each has 'measure'
    all_positions = []  # from segmentation: each has 'staff_idx', 'x'

    if notes and note_positions and len(notes) == len(note_positions):
        all_elements.extend(notes)
        all_positions.extend(note_positions)
    if rests and rest_positions and len(rests) == len(rest_positions):
        all_elements.extend(rests)
        all_positions.extend(rest_positions)

    if all_elements and all_positions and len(all_elements) == len(all_positions):
        _assign_from_elements(classified, all_elements, all_positions, debug)
    else:
        if debug:
            print(f"[detect_repeats] Element count mismatch (xml={len(all_elements)} "
                  f"seg={len(all_positions)}) — falling back to barline counting")
        _assign_from_barline_count(classified, staff_info, total_measures, debug)


def _assign_from_elements(
    classified: list[dict],
    elements: list[dict],
    positions: list[dict],
    debug: bool,
) -> None:
    """
    Assign measure numbers using element-count fitting (notes + rests).

    Step 1: Build staff → measure range by greedily fitting MusicXML element
    counts (notes + rests) to segmentation position counts per staff.

    Step 2: For each barline, interpolate its x position within the staff's
    x extent to determine which measure boundary it's at. This is robust
    against paired repeat barlines and missing barlines.
    """
    from collections import Counter
    import math

    # Count elements per measure (MusicXML) and per staff (segmentation)
    notes_per_measure = Counter(e["measure"] for e in elements)
    notes_per_staff = Counter(p["staff_idx"] for p in positions)

    measure_list = sorted(notes_per_measure.keys())
    staff_list = sorted(notes_per_staff.keys())

    # Greedy fit: walk measures in order, assign to staves by note count.
    # Start from measure 1 (not the first note-bearing measure) to include
    # rest-only measures at the beginning.
    staff_measure_ranges: dict[int, tuple[int, int]] = {}
    si_idx = 0
    staff_target = notes_per_staff[staff_list[si_idx]]
    accumulated = 0
    first_measure_on_staff = 1  # always start from m1

    for m in measure_list:
        count = notes_per_measure[m]
        if accumulated + count > staff_target and accumulated > 0 and si_idx + 1 < len(staff_list):
            # Close current staff
            staff_measure_ranges[staff_list[si_idx]] = (first_measure_on_staff, m - 1)
            si_idx += 1
            staff_target = notes_per_staff[staff_list[si_idx]]
            accumulated = 0
            first_measure_on_staff = m

        accumulated += count

    # Close final staff
    if si_idx < len(staff_list):
        last_m = measure_list[-1] if measure_list else 1
        staff_measure_ranges[staff_list[si_idx]] = (first_measure_on_staff, last_m)

    if debug:
        for si in sorted(staff_measure_ranges.keys()):
            first, last = staff_measure_ranges[si]
            print(f"[detect_repeats] Staff {si}: m{first}-m{last}")

    # Build staff lookup for x extent
    # Use barline positions for staff x range (more precise than staff_info min_x/max_x)
    staff_x_range: dict[int, tuple[float, float]] = {}
    for bl in classified:
        si = bl["staff_idx"]
        if si not in staff_x_range:
            staff_x_range[si] = (bl["x"], bl["x"])
        else:
            lo, hi = staff_x_range[si]
            staff_x_range[si] = (min(lo, bl["x"]), max(hi, bl["x"]))

    # For each barline, interpolate its position within the staff's measure range
    for bl in classified:
        si = bl["staff_idx"]
        if si not in staff_measure_ranges or si not in staff_x_range:
            bl["measure_before"] = None
            bl["measure_after"] = None
            continue

        first_m, last_m = staff_measure_ranges[si]
        x_lo, x_hi = staff_x_range[si]
        num_measures = last_m - first_m + 1
        staff_width = x_hi - x_lo

        if staff_width <= 0:
            bl["measure_before"] = first_m
            bl["measure_after"] = first_m
            continue

        # Proportional position: 0.0 = left edge, 1.0 = right edge
        proportion = (bl["x"] - x_lo) / staff_width

        # Map to nearest measure boundary.
        # Boundaries: 0 = left edge (before first_m), N = right edge (after last_m)
        boundary_pos = proportion * num_measures
        boundary = round(boundary_pos)
        boundary = max(0, min(num_measures, boundary))

        # Convert boundary to measure numbers
        m_before = first_m + boundary - 1  # measure to the left
        m_after = first_m + boundary        # measure to the right

        bl["measure_before"] = m_before if m_before >= first_m else None
        bl["measure_after"] = m_after if m_after <= last_m else None

    if debug:
        repeats = [bl for bl in classified if bl["type"] != "normal"]
        for bl in repeats:
            print(f"[detect_repeats] {bl['type']} at x={bl['x']:.0f} staff={bl['staff_idx']}: "
                  f"m_before={bl['measure_before']} m_after={bl['measure_after']}")


def _assign_from_barline_count(
    classified: list[dict],
    staff_info: list[dict],
    total_measures: int,
    debug: bool,
) -> None:
    """Fallback: assign measure numbers by counting barlines per staff."""
    staff_barlines: dict[int, list[dict]] = {}
    for bl in classified:
        idx = bl["staff_idx"]
        if idx not in staff_barlines:
            staff_barlines[idx] = []
        staff_barlines[idx].append(bl)

    for idx in staff_barlines:
        staff_barlines[idx].sort(key=lambda b: b["x"])

    current_measure = 1
    for staff_idx in sorted(staff_barlines.keys()):
        barlines = staff_barlines[staff_idx]
        n = len(barlines)
        measures_on_staff = max(1, n - 1)

        for i, bl in enumerate(barlines):
            if i == 0:
                bl["measure_before"] = None
                bl["measure_after"] = current_measure
            elif i == n - 1:
                bl["measure_before"] = current_measure + i - 1
                bl["measure_after"] = None
            else:
                bl["measure_before"] = current_measure + i - 1
                bl["measure_after"] = current_measure + i

        current_measure += measures_on_staff

    if debug:
        inferred = current_measure - 1
        if total_measures > 0 and inferred != total_measures:
            print(f"[detect_repeats] WARNING: inferred {inferred} measures "
                  f"(MusicXML says {total_measures})")


def build_repeat_markers(classified_barlines: list[dict], debug: bool = False) -> list[dict]:
    """
    Build repeat_markers from classified barlines.

    Pairs each backward repeat with the nearest preceding forward repeat
    (or measure 1 if none). Deduplicates when HOMR produces two boxes
    for one repeat structure (both boxes detect the same dots).

    Returns list matching the iOS RepeatMarker format:
    [{
        "start_measure": int,
        "end_measure": int,
        "repeat_count": 1,
        "volta_endings": null
    }]
    """
    forwards = []   # measure numbers where forward repeats start
    backwards = []  # measure numbers where backward repeats end

    for bl in classified_barlines:
        if bl["type"] == "forward":
            m = bl.get("measure_after")
            if m is not None:
                forwards.append(m)
        elif bl["type"] == "backward":
            m = bl.get("measure_before")
            if m is not None:
                backwards.append(m)

    # Deduplicate: paired boxes detect the same dots, producing two
    # entries for the same repeat point. Keep unique measure numbers.
    forwards = sorted(set(forwards))
    backwards = sorted(set(backwards))

    if debug:
        print(f"[detect_repeats] Forward starts at measures: {forwards}")
        print(f"[detect_repeats] Backward ends at measures: {backwards}")

    # Pair backwards with forwards
    repeat_markers = []
    available_forwards = list(forwards)

    for end_measure in backwards:
        start_measure = 1  # default: repeat from beginning
        best_forward = None
        for fs in reversed(available_forwards):
            if fs <= end_measure:
                start_measure = fs
                best_forward = fs
                break

        if best_forward is not None:
            available_forwards.remove(best_forward)

        repeat_markers.append({
            "start_measure": start_measure,
            "end_measure": end_measure,
            "repeat_count": 1,
            "volta_endings": None,
        })

    if debug:
        for rm in repeat_markers:
            print(
                f"[detect_repeats] Repeat: m{rm['start_measure']}–m{rm['end_measure']}"
            )

    return repeat_markers
