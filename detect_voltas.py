"""
Volta Bracket Detector — Post-processing for HOMR OMR output.

Strategy:
1. Start from HOMR's detected repeat barlines (from MusicXML)
2. Find horizontal bracket lines near those repeats
3. Read the number at the bracket's start, requiring co-location
4. If we find "2", infer "1" from the bracket to its left

This code is licensed under AGPL-3.0 to comply with HOMR's license.
"""

import cv2
import numpy as np


_volta_debug = {}  # Module-level debug info, accessible after detection

def detect_voltas(
    image_path: str,
    repeat_markers: list[dict],
    total_measures: int = 0,
    staff_info: list[dict] | None = None,
    barline_info: list[dict] | None = None,
) -> list[dict]:
    """Detect volta brackets near repeat barlines and merge with repeat markers."""
    global _volta_debug
    _volta_debug = {"repeat_markers_in": len(repeat_markers), "checks": []}

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

    # For each repeat marker, look for volta brackets near it
    num_staves = len(staves)
    mps = max(1, (total_measures or 53) / num_staves)

    result = list(repeat_markers)

    for idx, rm in enumerate(result):
        check = {"repeat": f"m{rm['start_measure']}-m{rm['end_measure']}"}

        # Which staff is the repeat's end measure on?
        end_staff_idx = _find_staff_for_measure(rm["end_measure"], staves, staff_measures, mps)
        check["end_staff_idx"] = end_staff_idx
        if end_staff_idx is None:
            check["result"] = "no staff found"
            _volta_debug["checks"].append(check)
            continue

        # Look for bracket lines in the gap ABOVE this staff
        volta_pair, pair_debug = _find_volta_pair_near_repeat(
            img, binary, staves, end_staff_idx, w
        )
        check["primary_check"] = pair_debug

        if volta_pair:
            voltas = _build_volta_endings(
                volta_pair, rm["start_measure"], rm["end_measure"]
            )
            result[idx] = dict(rm, volta_endings=voltas)
            check["result"] = f"found via primary: {voltas}"
            print(f"[volta] Repeat m{rm['start_measure']}-m{rm['end_measure']}: "
                  f"added volta endings {voltas}")

        # Also check the gap above the NEXT staff (cross-staff voltas)
        if not volta_pair and end_staff_idx + 1 < num_staves:
            volta_pair, pair_debug2 = _find_volta_pair_near_repeat(
                img, binary, staves, end_staff_idx + 1, w
            )
            check["cross_staff_check"] = pair_debug2
            if volta_pair:
                voltas = _build_volta_endings(
                    volta_pair, rm["start_measure"], rm["end_measure"]
                )
                result[idx] = dict(rm, volta_endings=voltas)
                check["result"] = f"found via cross-staff: {voltas}"
                print(f"[volta] Repeat m{rm['start_measure']}-m{rm['end_measure']}: "
                      f"added volta endings {voltas} (cross-staff)")

        if "result" not in check:
            check["result"] = "not found"

        _volta_debug["checks"].append(check)

    return result


def _find_volta_pair_near_repeat(
    img: np.ndarray,
    binary: np.ndarray,
    staves: list[dict],
    staff_idx: int,
    img_width: int,
) -> tuple[list[int] | None, dict]:
    """
    Look for volta bracket lines + numbers in the gap above the given staff.
    Returns (sorted list of volta numbers [1, 2] or None, debug_info dict).
    """
    debug = {"staff_idx": staff_idx}

    if staff_idx == 0:
        debug["skip"] = "staff_idx == 0"
        return None, debug

    prev_bottom = staves[staff_idx - 1]["bottom"]
    curr_top = staves[staff_idx]["top"]
    spacing = staves[staff_idx]["spacing"]
    gap = curr_top - prev_bottom
    debug["zone"] = f"y={prev_bottom + int(spacing)}-{curr_top - int(spacing * 0.5)}"
    debug["gap"] = gap
    debug["spacing"] = spacing

    if gap < spacing * 2:
        debug["skip"] = f"gap {gap} < spacing*2 {spacing*2}"
        return None, debug

    zone_top = prev_bottom + int(spacing)
    zone_bot = curr_top - int(spacing * 0.5)

    if zone_bot - zone_top < 10:
        debug["skip"] = "zone too small"
        return None, debug

    zone_bin = binary[zone_top:zone_bot, :]

    # Find horizontal bracket lines
    min_bw = max(50, int(img_width * 0.03))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_bw, 1))
    horiz_lines = cv2.morphologyEx(zone_bin, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(horiz_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_contours = []
    brackets = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        all_contours.append({"x": x, "y": y + zone_top, "w": cw, "h": ch})
        if cw > spacing * 5 and ch < spacing * 2 and cw < img_width * 0.7:
            brackets.append({
                "x": x,
                "y": y + zone_top,
                "width": cw,
                "height": ch,
            })

    debug["contours_total"] = len(all_contours)
    debug["contours_all"] = all_contours[:10]  # first 10 for debugging
    debug["brackets_passing_filter"] = len(brackets)

    if not brackets:
        debug["skip"] = f"no brackets (had {len(all_contours)} contours, none passed filter)"
        return None, debug

    brackets.sort(key=lambda b: b["x"])

    # OCR each bracket's left end to read the number
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    except ImportError:
        debug["skip"] = "rapidocr not available"
        return None, debug

    identified = {}
    ocr_results = []
    for bi, bracket in enumerate(brackets):
        left_x = max(0, bracket["x"] - int(spacing))
        right_x = min(img_width, bracket["x"] + int(spacing * 5))
        top_y = bracket["y"]
        bot_y = min(img.shape[0], bracket["y"] + int(spacing * 3))

        if bot_y - top_y < 5 or right_x - left_x < 5:
            continue

        crop = img[top_y:bot_y, left_x:right_x, :]
        result, _ = ocr(crop)

        bracket_ocr = {"bracket_idx": bi, "crop": f"y={top_y}-{bot_y} x={left_x}-{right_x}", "texts": []}
        if result:
            for box, text, conf in result:
                tc = text.strip().rstrip(".,;:")
                text_y = int(min(p[1] for p in box))
                bracket_ocr["texts"].append({"text": tc, "conf": round(conf, 2), "y": text_y})
                if text_y > spacing * 2:
                    continue

                if tc in ("1", "I", "l"):
                    identified[bi] = 1
                elif tc == "2":
                    identified[bi] = 2
                elif tc == "3":
                    identified[bi] = 3

        ocr_results.append(bracket_ocr)

    debug["ocr_results"] = ocr_results
    debug["identified_before_infer"] = dict(identified)

    # Infer "1" from "2"
    for bi, num in list(identified.items()):
        if num == 2 and bi > 0 and (bi - 1) not in identified:
            identified[bi - 1] = 1
        elif num == 2 and bi == 0:
            identified[-1] = 1  # synthetic

    debug["identified_after_infer"] = dict(identified)

    volta_numbers = sorted(set(identified.values()))
    debug["volta_numbers"] = volta_numbers
    if len(volta_numbers) >= 2:
        return volta_numbers, debug

    debug["skip"] = f"insufficient volta numbers: {volta_numbers}"
    return None, debug


def _find_staff_for_measure(
    measure: int,
    staves: list[dict],
    staff_measures: dict[int, list[int]],
    measures_per_staff: float,
) -> int | None:
    """Find which staff index contains a given measure number."""
    # First try exact match from OCR-detected measure numbers
    for si, measures in staff_measures.items():
        if measure in measures:
            return si
        # Also check if measure falls within the range of this staff
        if measures and min(measures) <= measure <= max(measures):
            return si

    # Check if it falls in the range between detected measures of adjacent staves
    sorted_staves = sorted(staff_measures.keys())
    for i, si in enumerate(sorted_staves):
        curr_max = max(staff_measures[si])
        if i + 1 < len(sorted_staves):
            next_min = min(staff_measures[sorted_staves[i + 1]])
            if curr_max < measure < next_min:
                return si  # measure is after this staff's last detected number

    # Fallback to estimation
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


def _build_volta_endings(volta_numbers, start_measure, end_measure):
    """Build volta endings dict."""
    num_voltas = len(volta_numbers)
    measures_for_voltas = min(num_voltas, end_measure - start_measure + 1)
    volta_start = end_measure - measures_for_voltas + 1
    return {str(volta_start + i): [vnum] for i, vnum in enumerate(volta_numbers)}


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
