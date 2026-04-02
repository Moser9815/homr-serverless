"""
Volta Bracket Detector — Post-processing for HOMR OMR output.

Uses a structural approach: find horizontal bracket lines between staves,
then read the number ("1" or "2") near the left end of each bracket.

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
    """
    Detect volta brackets by finding horizontal lines between staves
    and reading the number at their left end.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[volta] Could not read image: {image_path}")
        return repeat_markers

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Get staff positions (prefer HOMR data, fall back to OpenCV)
    if staff_info:
        staves = [
            {"top": int(s["min_y"]), "bottom": int(s["max_y"]),
             "spacing": s["unit_size"], "index": i}
            for i, s in enumerate(staff_info)
        ]
        print(f"[volta] Using {len(staves)} staves from HOMR")
    else:
        staves = _find_staves(gray, w, h)
        print(f"[volta] Found {len(staves)} staves via OpenCV")

    if len(staves) < 2:
        return repeat_markers

    # For each inter-staff gap, look for volta bracket lines + numbers
    volta_pairs = _find_volta_brackets(img, gray, staves, w)

    if not volta_pairs:
        print("[volta] No volta brackets found")
        return repeat_markers

    print(f"[volta] Found {len(volta_pairs)} volta bracket pairs")

    # Map volta pairs to measures using OCR-detected measure numbers
    if total_measures == 0:
        all_m = [rm["end_measure"] for rm in repeat_markers] if repeat_markers else [1]
        total_measures = max(all_m)

    staff_measures = _detect_measure_numbers(img, staves, total_measures)
    print(f"[volta] Staff measures: {staff_measures}")

    return _merge_voltas(volta_pairs, staves, staff_measures, repeat_markers, total_measures)


def _find_volta_brackets(
    img: np.ndarray,
    gray: np.ndarray,
    staves: list[dict],
    img_width: int,
) -> list[dict]:
    """
    Find volta brackets in each inter-staff gap.

    A volta bracket is:
    - A horizontal line above the staff that ISN'T a staff line
    - With "1", "2", or "I" text near its left end
    """
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )

    volta_pairs = []

    for i in range(1, len(staves)):
        prev_bottom = staves[i - 1]["bottom"]
        curr_top = staves[i]["top"]
        spacing = staves[i]["spacing"]
        gap = curr_top - prev_bottom

        if gap < spacing * 2:
            continue  # gap too small for a volta bracket

        # Search zone: the full gap between staves, but skip the staff lines
        zone_top = prev_bottom + int(spacing)
        zone_bot = curr_top - int(spacing * 0.5)

        if zone_bot - zone_top < 10:
            continue

        zone_bin = binary[zone_top:zone_bot, :]
        zh = zone_bot - zone_top

        # Find horizontal lines in this zone (volta bracket lines)
        # Use a shorter kernel than staff line detection — bracket lines are
        # narrower than staff lines (they span 1-3 measures, not the full width)
        min_bracket_width = max(50, int(img_width * 0.03))  # ~3% of width
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_bracket_width, 1))
        horiz_lines = cv2.morphologyEx(zone_bin, cv2.MORPH_OPEN, horiz_kernel)

        # Find contours of horizontal lines
        contours, _ = cv2.findContours(horiz_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        brackets = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Bracket must be: wider than ~5 staff spacings, thin, not full-width
            if cw > spacing * 5 and ch < spacing * 2 and cw < img_width * 0.7:
                brackets.append({
                    "x": x,
                    "y": y + zone_top,
                    "width": cw,
                    "height": ch,
                    "staff_below": i,
                })

        if not brackets:
            continue

        # For each bracket, look for "1" or "2" near its left end
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR()
        except ImportError:
            continue

        for bracket in brackets:
            # Crop a small area near the left end of the bracket
            # The number sits just below and slightly right of the bracket's left edge
            left_x = max(0, bracket["x"] - int(spacing))
            right_x = min(img_width, bracket["x"] + int(spacing * 5))
            top_y = bracket["y"]
            bot_y = min(img.shape[0], bracket["y"] + int(spacing * 3))

            if bot_y - top_y < 5 or right_x - left_x < 5:
                continue

            crop = img[top_y:bot_y, left_x:right_x, :]
            result, _ = ocr(crop)

            if not result:
                continue

            volta_number = None
            for box, text, conf in result:
                tc = text.strip().rstrip(".,;:")
                if tc in ("1", "I", "l"):
                    volta_number = 1
                elif tc == "2":
                    volta_number = 2
                elif tc == "3":
                    volta_number = 3

            if volta_number is not None:
                volta_pairs.append({
                    "number": volta_number,
                    "bracket_x": bracket["x"],
                    "bracket_y": bracket["y"],
                    "bracket_width": bracket["width"],
                    "staff_below": bracket["staff_below"],
                })

    return volta_pairs


def _detect_measure_numbers(
    img: np.ndarray,
    staves: list[dict],
    total_measures: int,
) -> dict[int, list[int]]:
    """Detect measure numbers above each staff using OCR."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return {}

    ocr = RapidOCR()
    staff_measures: dict[int, list[int]] = {}

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
        measure_y_min = zone_top + zone_height * 0.5  # measures are in lower half

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


def _merge_voltas(
    volta_pairs: list[dict],
    staves: list[dict],
    staff_measures: dict[int, list[int]],
    repeat_markers: list[dict],
    total_measures: int,
) -> list[dict]:
    """Merge detected volta brackets into repeat markers."""
    num_staves = len(staves)
    measures_per_staff = max(1, total_measures / num_staves)

    # Group volta pairs by the staff they're above
    by_staff: dict[int, list[dict]] = {}
    for vp in volta_pairs:
        si = vp["staff_below"]
        by_staff.setdefault(si, []).append(vp)

    # Filter: need at least 2 different numbers for a valid volta pair
    valid_groups = {}
    for si, pairs in by_staff.items():
        nums = set(p["number"] for p in pairs)
        if len(nums) >= 2:
            valid_groups[si] = pairs
        else:
            print(f"[volta] Dropping staff {si}: only found volta numbers {nums}")

    if not valid_groups:
        return repeat_markers

    result = list(repeat_markers)

    for si, pairs in valid_groups.items():
        volta_numbers = sorted(set(p["number"] for p in pairs))

        # Determine measure range using OCR measure numbers
        # The volta is above staff si, so measures come from staff si-1 or si
        prev_measures = staff_measures.get(si - 1, [])
        curr_measures = staff_measures.get(si, [])

        if prev_measures and curr_measures:
            prev_last = max(prev_measures)
            curr_first = min(curr_measures)
            if curr_first - prev_last > measures_per_staff * 1.5:
                # Big gap — volta is in undetected staves between
                last_m = curr_first - 1
                first_m = max(1, last_m - int(measures_per_staff) + 1)
            else:
                first_m = min(prev_measures)
                last_m = max(prev_measures)
        elif prev_measures:
            first_m = min(prev_measures)
            last_m = max(prev_measures)
        elif curr_measures:
            curr_first = min(curr_measures)
            last_m = curr_first - 1
            first_m = max(1, last_m - int(measures_per_staff) + 1)
        else:
            first_m = int((si - 1) * measures_per_staff) + 1
            last_m = min(int(si * measures_per_staff), total_measures)

        # Check if an existing repeat marker falls in this range
        # Also check the range of the staff ABOVE (volta brackets sit between staves)
        # and any repeat that ends BEFORE the current staff's first measure
        check_ranges = [(first_m, last_m)]
        if si - 1 >= 0:
            prev_curr = staff_measures.get(si - 1, [])
            if prev_curr:
                # Use the full range of the previous staff.
                # First staff starts at measure 1; others start after
                # the staff before them.
                prev_prev = staff_measures.get(si - 2, [])
                range_start = max(prev_prev) + 1 if prev_prev else 1
                check_ranges.append((range_start, max(prev_curr)))
            else:
                check_ranges.append((1, first_m))

        matched = False
        for range_first, range_last in check_ranges:
            for idx, rm in enumerate(result):
                if range_first <= rm["end_measure"] <= range_last:
                    voltas = _build_volta_endings(volta_numbers, rm["start_measure"], rm["end_measure"])
                    result[idx] = dict(rm, volta_endings=voltas)
                    print(f"[volta] Repeat m{rm['start_measure']}-m{rm['end_measure']}: "
                          f"added volta endings {voltas}")
                    matched = True
                    break
            if matched:
                break

        if not matched:
            voltas = _build_volta_endings(volta_numbers, first_m, last_m)
            result.append({
                "start_measure": first_m,
                "end_measure": last_m,
                "repeat_count": 1,
                "volta_endings": voltas,
            })
            print(f"[volta] Created NEW repeat m{first_m}-m{last_m} with volta endings {voltas}")

    return result


def _build_volta_endings(volta_numbers, start_measure, end_measure):
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
