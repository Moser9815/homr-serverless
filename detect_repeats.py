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

    # Assign measure numbers
    _assign_measure_numbers(classified, staff_info, total_measures, debug)

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
    debug: bool,
) -> None:
    """
    Assign measure numbers to each barline based on position ordering.

    Within each staff, barlines sorted left-to-right separate consecutive
    measures. N barlines = N-1 measures. However, HOMR sometimes produces
    two boxes for a single repeat barline (thick + thin bar, ~17px apart).
    These paired boxes represent ONE measure boundary, not two. We detect
    pairs by proximity and assign both boxes the same measure boundary.
    """
    # Group by staff
    staff_barlines: dict[int, list[dict]] = {}
    for bl in classified:
        idx = bl["staff_idx"]
        if idx not in staff_barlines:
            staff_barlines[idx] = []
        staff_barlines[idx].append(bl)

    # Sort by x within each staff
    for idx in staff_barlines:
        staff_barlines[idx].sort(key=lambda b: b["x"])

    # Build staff lookup for unit_size
    staff_lookup = {s["staff"]: s for s in staff_info}

    # Walk through staves in order, assigning measure numbers.
    # HOMR produces two boxes for each repeat barline (thick + thin bar).
    # Only consolidate pairs where BOTH are classified as repeat types —
    # don't consolidate normal close barlines.
    current_measure = 1
    for staff_idx in sorted(staff_barlines.keys()):
        barlines = staff_barlines[staff_idx]
        staff = staff_lookup.get(staff_idx)
        unit = staff["unit_size"] if staff else 20.0
        n = len(barlines)

        # Identify repeat-classified pairs to consolidate
        boundary_ids = []
        boundary = 0
        i = 0
        while i < n:
            boundary_ids.append(boundary)
            # Only consolidate if: close together AND both are repeat types
            if i + 1 < n:
                gap = barlines[i + 1]["x"] - barlines[i]["x"]
                both_repeat = (
                    barlines[i]["type"] != "normal"
                    and barlines[i + 1]["type"] != "normal"
                )
                if gap < unit * 1.0 and both_repeat:
                    i += 1
                    boundary_ids.append(boundary)  # same boundary
            boundary += 1
            i += 1

        num_boundaries = boundary
        measures_on_staff = max(1, num_boundaries - 1)

        for i, bl in enumerate(barlines):
            b = boundary_ids[i]
            if b == 0:
                bl["measure_before"] = None
                bl["measure_after"] = current_measure
            elif b == num_boundaries - 1:
                bl["measure_before"] = current_measure + b - 1
                bl["measure_after"] = None
            else:
                bl["measure_before"] = current_measure + b - 1
                bl["measure_after"] = current_measure + b

        current_measure += measures_on_staff

    inferred_total = current_measure - 1
    if total_measures > 0 and inferred_total != total_measures:
        if debug:
            print(
                f"[detect_repeats] Inferred {inferred_total} measures "
                f"(MusicXML says {total_measures})"
            )


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
