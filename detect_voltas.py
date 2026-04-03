"""
Volta Bracket Detector — Post-processing for HOMR OMR output.

Volta brackets (1st/2nd/3rd endings) follow a consistent pattern:
- LEFT of backward repeat: 1+ closed brackets, each covering specific bars.
  Volta 1 is leftmost, volta 2 next, etc. Only one plays per pass.
- RIGHT of backward repeat: 1 open-ended bracket (highest number).
  Runs from the backward repeat to the next forward repeat start.
  Played on the final pass after skipping all left-side voltas.

Strategy:
1. Find horizontal bracket lines near repeat barlines
2. Read the number at each bracket's start via OCR
3. Use "2 implies 1" inference for OCR misreads
4. Count measures under left-side brackets using barline spacing
5. Right-side bracket length = distance to next forward repeat

This code is licensed under AGPL-3.0 to comply with HOMR's license.
"""

import cv2
import numpy as np


def detect_voltas(
    image_path: str,
    repeat_markers: list[dict],
    total_measures: int = 0,
    staff_info: list[dict] | None = None,
    barline_info: list[dict] | None = None,
) -> list[dict]:
    """Detect volta brackets near repeat barlines and merge with repeat markers."""
    if not repeat_markers:
        return repeat_markers

    img = cv2.imread(image_path)
    if img is None:
        return repeat_markers

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if staff_info:
        staves = [
            {"top": int(s["min_y"]), "bottom": int(s["max_y"]),
             "spacing": s["unit_size"], "index": i}
            for i, s in enumerate(staff_info)
        ]
    else:
        staves = _find_staves(gray, w, h)

    if len(staves) < 2:
        return repeat_markers

    # Binary image for bracket line detection
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )

    # Detect measure numbers on each staff for measure-to-staff mapping
    staff_measures = _detect_measure_numbers(img, staves, total_measures or 53)

    # Compute average barline spacing per staff (for measure width estimation)
    staff_barline_spacing = _compute_barline_spacing(barline_info, staves)

    num_staves = len(staves)
    mps = max(1, (total_measures or 53) / num_staves)

    result = list(repeat_markers)

    for idx, rm in enumerate(result):
        end_staff_idx = _find_staff_for_measure(rm["end_measure"], staves, staff_measures, mps)
        if end_staff_idx is None:
            continue

        # Find the next repeat's start measure (for open-ended volta length)
        next_start = _find_next_repeat_start(idx, result, total_measures)

        # Look for bracket lines in the gap ABOVE this staff
        brackets = _find_volta_brackets(img, binary, staves, end_staff_idx, w)

        if not brackets:
            # Cross-staff search: volta may be above a subsequent staff
            for check_idx in range(end_staff_idx + 1, min(end_staff_idx + 4, num_staves)):
                brackets = _find_volta_brackets(img, binary, staves, check_idx, w)
                if brackets:
                    break

        if brackets:
            avg_spacing = staff_barline_spacing.get(end_staff_idx, 0)
            voltas = _build_volta_endings(
                brackets, rm["end_measure"], next_start, avg_spacing
            )
            if voltas:
                num_voltas = max(v for iters in voltas.values() for v in iters)
                result[idx] = dict(rm, volta_endings=voltas, repeat_count=num_voltas)
                print(f"[volta] Repeat m{rm['start_measure']}-m{rm['end_measure']}: "
                      f"volta endings {voltas} (repeat_count={num_voltas})")

    return result


def _find_next_repeat_start(current_idx: int, repeat_markers: list[dict], total_measures: int) -> int:
    """Find the start_measure of the next repeat marker after the current one."""
    current_end = repeat_markers[current_idx]["end_measure"]
    best = total_measures + 1  # default: end of piece
    for i, rm in enumerate(repeat_markers):
        if i != current_idx and rm["start_measure"] > current_end:
            best = min(best, rm["start_measure"])
    return best


def _compute_barline_spacing(barline_info: list[dict] | None, staves: list[dict]) -> dict[int, float]:
    """Compute average distance between consecutive barlines on each staff."""
    if not barline_info:
        return {}

    # Group barlines by staff, sort by x
    staff_barlines: dict[int, list[float]] = {}
    for bl in barline_info:
        idx = bl["staff_idx"]
        if idx not in staff_barlines:
            staff_barlines[idx] = []
        staff_barlines[idx].append(bl["x"])

    result = {}
    for idx, positions in staff_barlines.items():
        positions.sort()
        if len(positions) >= 2:
            gaps = [positions[i+1] - positions[i] for i in range(len(positions) - 1)]
            # Filter out tiny gaps (paired repeat barlines ~17px) — use median
            # to be robust against these outliers
            result[idx] = float(np.median(gaps))

    return result


def _find_volta_brackets(
    img: np.ndarray,
    binary: np.ndarray,
    staves: list[dict],
    staff_idx: int,
    img_width: int,
) -> list[dict] | None:
    """
    Find volta bracket lines + numbers in the gap above the given staff.

    Returns list of bracket dicts with:
    - x, y, width, height (pixel coords)
    - number: volta number (1, 2, 3...)
    Sorted left-to-right. Returns None if no valid volta pair found.
    """
    if staff_idx == 0:
        return None

    prev_bottom = staves[staff_idx - 1]["bottom"]
    curr_top = staves[staff_idx]["top"]
    spacing = staves[staff_idx]["spacing"]
    gap = curr_top - prev_bottom

    if gap < spacing * 2:
        return None

    zone_top = prev_bottom + int(spacing)
    zone_bot = curr_top - int(spacing * 0.5)

    if zone_bot - zone_top < 10:
        return None

    zone_bin = binary[zone_top:zone_bot, :]

    # Find horizontal bracket lines
    min_bw = max(50, int(img_width * 0.03))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_bw, 1))
    horiz_lines = cv2.morphologyEx(zone_bin, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(horiz_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    brackets = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw > spacing * 5 and ch < spacing * 2 and cw < img_width * 0.7:
            brackets.append({
                "x": x,
                "y": y + zone_top,
                "width": cw,
                "height": ch,
            })

    if not brackets:
        return None

    brackets.sort(key=lambda b: b["x"])

    # OCR each bracket's left end to read the number
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    except ImportError:
        return None

    for bi, bracket in enumerate(brackets):
        # Crop near the left end of the bracket
        left_x = max(0, bracket["x"] - int(spacing))
        right_x = min(img_width, bracket["x"] + int(spacing * 5))
        top_y = bracket["y"]
        bot_y = min(img.shape[0], bracket["y"] + int(spacing * 3))

        if bot_y - top_y < 5 or right_x - left_x < 5:
            continue

        crop = img[top_y:bot_y, left_x:right_x, :]
        result, _ = ocr(crop)

        bracket["number"] = None
        if not result:
            continue

        for box, text, conf in result:
            tc = text.strip().rstrip(".,;:")
            text_y = int(min(p[1] for p in box))
            if text_y > spacing * 2:
                continue

            if tc in ("1", "I", "l"):
                bracket["number"] = 1
            elif tc == "2":
                bracket["number"] = 2
            elif tc == "3":
                bracket["number"] = 3
            elif tc == "4":
                bracket["number"] = 4

    # Infer "1" from "2": if bracket[i] is "2" and bracket[i-1] has no number
    for bi in range(len(brackets)):
        num = brackets[bi].get("number")
        if num == 2 and bi > 0 and brackets[bi - 1].get("number") is None:
            brackets[bi - 1]["number"] = 1

    # Filter to only brackets with identified numbers
    numbered = [b for b in brackets if b.get("number") is not None]
    if len(numbered) < 2:
        return None

    # Need at least two different volta numbers
    numbers = set(b["number"] for b in numbered)
    if len(numbers) < 2:
        return None

    return numbered


def _build_volta_endings(
    brackets: list[dict],
    end_measure: int,
    next_repeat_start: int,
    avg_barline_spacing: float,
) -> dict[str, list[int]] | None:
    """
    Build volta endings from detected brackets.

    Left-side brackets (closed): measure count from bracket width / barline spacing.
    Working backward from end_measure.

    Right-side bracket (open-ended, highest number): from end_measure+1 to
    next_repeat_start-1.
    """
    if not brackets or avg_barline_spacing <= 0:
        return None

    # Sort by x position
    brackets = sorted(brackets, key=lambda b: b["x"])

    # Find the highest volta number — that's the open-ended one on the right
    max_volta = max(b["number"] for b in brackets)

    # Separate left-side (closed) and right-side (open) brackets
    # The right-side bracket is the one with the highest number
    # In most cases: volta 1 left, volta 2 right (2-volta)
    # Or: volta 1 left, volta 2 left, volta 3 right (3-volta)
    right_brackets = [b for b in brackets if b["number"] == max_volta]
    left_brackets = [b for b in brackets if b["number"] != max_volta]

    # Sort left brackets by number (volta 1, 2, ...)
    left_brackets.sort(key=lambda b: b["number"])

    volta_endings = {}

    # Process left-side brackets: work backward from end_measure
    cursor = end_measure  # the last measure before the backward repeat
    for bracket in reversed(left_brackets):
        # Estimate how many measures this bracket covers
        num_measures = max(1, round(bracket["width"] / avg_barline_spacing))

        # Assign measures working backward from cursor
        start = cursor - num_measures + 1
        for m in range(start, cursor + 1):
            volta_endings[str(m)] = [bracket["number"]]

        cursor = start - 1  # next bracket ends before this one starts

    # Process right-side bracket: end_measure+1 to next_repeat_start-1
    if right_brackets:
        right_start = end_measure + 1
        right_end = next_repeat_start - 1
        if right_end >= right_start:
            for m in range(right_start, right_end + 1):
                volta_endings[str(m)] = [max_volta]

    if volta_endings:
        print(f"[volta] Bracket analysis: left={[(b['number'], round(b['width']/avg_barline_spacing)) for b in left_brackets]}bars "
              f"right=volta{max_volta} m{end_measure+1}-m{next_repeat_start-1}")

    return volta_endings if volta_endings else None


def _find_staff_for_measure(
    measure: int,
    staves: list[dict],
    staff_measures: dict[int, list[int]],
    measures_per_staff: float,
) -> int | None:
    """Find which staff index contains a given measure number."""
    for si, measures in staff_measures.items():
        if measure in measures:
            return si
        if measures and min(measures) <= measure <= max(measures):
            return si

    sorted_staves = sorted(staff_measures.keys())
    for i, si in enumerate(sorted_staves):
        curr_max = max(staff_measures[si])
        if i + 1 < len(sorted_staves):
            next_min = min(staff_measures[sorted_staves[i + 1]])
            if curr_max < measure < next_min:
                return si

    num_staves = len(staves)
    est = min(int((measure - 1) / measures_per_staff), num_staves - 1)
    return est


def _detect_measure_numbers(img, staves, total_measures):
    """Detect measure numbers above each staff using OCR."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return {}

    ocr = RapidOCR()
    staff_measures = {}

    for i, staff in enumerate(staves):
        spacing = staff["spacing"]
        if i > 0:
            zone_top = staves[i - 1]["bottom"] + int(spacing * 2)
        else:
            zone_top = max(0, staff["top"] - int(spacing * 8))
        zone_bottom = staff["top"]

        if zone_bottom - zone_top < 10:
            continue

        zone_height = zone_bottom - zone_top
        measure_y_min = zone_top + zone_height * 0.5

        zone = img[zone_top:zone_bottom, :, :]
        result, _ = ocr(zone)
        if not result:
            continue

        measures = []
        for box, text, conf in result:
            tc = text.strip().rstrip(".,;:")
            box_y = int(min(p[1] for p in box)) + zone_top
            if tc.isdigit() and 2 <= len(tc) <= 3 and box_y >= measure_y_min:
                num = int(tc)
                if 1 <= num <= total_measures + 20:
                    measures.append(num)

        if measures:
            staff_measures[i] = sorted(set(measures))

    return staff_measures


def _find_staves(gray, w, h):
    """Fallback: find staves via OpenCV morphology."""
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 10)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 8, 1))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    h_proj = np.sum(horiz_lines, axis=1) / 255
    staff_rows = np.where(h_proj > w * 0.15)[0]

    if len(staff_rows) == 0:
        return []

    groups = []
    current = [staff_rows[0]]
    for i in range(1, len(staff_rows)):
        if staff_rows[i] - staff_rows[i - 1] <= 5:
            current.append(staff_rows[i])
        else:
            groups.append(int(np.mean(current)))
            current = [staff_rows[i]]
    groups.append(int(np.mean(current)))

    staves = []
    current_staff = [groups[0]]
    for i in range(1, len(groups)):
        if groups[i] - groups[i - 1] < 50:
            current_staff.append(groups[i])
        else:
            if len(current_staff) >= 4:
                top, bottom = current_staff[0], current_staff[-1]
                spacing = (bottom - top) / max(1, len(current_staff) - 1)
                staves.append({"top": top, "bottom": bottom, "spacing": spacing, "index": len(staves)})
            current_staff = [groups[i]]
    if len(current_staff) >= 4:
        top, bottom = current_staff[0], current_staff[-1]
        spacing = (bottom - top) / max(1, len(current_staff) - 1)
        staves.append({"top": top, "bottom": bottom, "spacing": spacing, "index": len(staves)})

    return staves
