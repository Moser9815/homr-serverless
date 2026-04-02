"""
Volta Bracket Detector — Post-processing for HOMR OMR output.

HOMR's transformer doesn't reliably detect volta brackets (1st/2nd endings)
due to training data scarcity. This module uses OCR to detect "1" / "2" text
above staves and creates repeat markers with volta endings.

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
    Detect volta brackets and merge with HOMR's repeat markers.

    If staff_info/barline_info are provided (from HOMR's Python API),
    uses those for accurate staff positions. Otherwise falls back to
    OpenCV-based staff detection.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[volta] Could not read image: {image_path}")
        return repeat_markers

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Use HOMR's staff data if available, otherwise fall back to OpenCV
    if staff_info:
        staves = []
        for si, s in enumerate(staff_info):
            staves.append({
                "top": int(s["min_y"]),
                "bottom": int(s["max_y"]),
                "spacing": s["unit_size"],
                "index": si,
            })
        print(f"[volta] Using {len(staves)} staves from HOMR API")
    else:
        staves = _find_staves(gray, w, h)
        if not staves:
            print("[volta] No staves found")
            return repeat_markers
        print(f"[volta] Found {len(staves)} staves via OpenCV")

    volta_texts, staff_measures = _find_volta_texts_and_measures(img, staves)
    if not volta_texts:
        print("[volta] No volta text detected")
        return repeat_markers

    summary = [(v["number"], v["staff_index"], v["x"]) for v in volta_texts]
    print(f"[volta] Found {len(volta_texts)} volta texts: {summary}")
    print(f"[volta] Staff measures: {staff_measures}")

    if total_measures == 0:
        all_m = [rm["end_measure"] for rm in repeat_markers] if repeat_markers else [1]
        total_measures = max(all_m)

    # Group volta texts by staff, then merge adjacent staves that together
    # form a volta pair (e.g., "1" on staff N and "2" on staff N+1)
    by_staff: dict[int, list[dict]] = {}
    for vt in volta_texts:
        by_staff.setdefault(vt["staff_index"], []).append(vt)

    # Merge adjacent staves: if staff N has only "1" or "2" and staff N+1
    # has only the other, combine them under the higher staff index
    staff_indices = sorted(by_staff.keys())
    for i in range(len(staff_indices) - 1):
        si_a = staff_indices[i]
        si_b = staff_indices[i + 1]
        if si_b - si_a == 1:  # Adjacent staves
            nums_a = set(v["number"] for v in by_staff[si_a])
            nums_b = set(v["number"] for v in by_staff[si_b])
            # If together they form a complete pair but individually they don't
            if len(nums_a) < 2 and len(nums_b) < 2 and len(nums_a | nums_b) >= 2:
                # Merge into the higher staff (where the music continues)
                by_staff[si_b] = by_staff.get(si_b, []) + by_staff.pop(si_a, [])
                print(f"[volta] Merged volta texts from staff {si_a} into staff {si_b}")

    # For each staff with a volta pair, determine the measure range
    # using OCR-detected measure numbers (not estimation)
    volta_groups = []
    num_staves = len(staves)
    measures_per_staff = max(1, total_measures / num_staves)

    for si, texts in by_staff.items():
        volta_numbers = sorted(set(v["number"] for v in texts))
        if len(volta_numbers) < 2:
            continue

        # Volta text above staff N belongs to the staff ABOVE (N-1).
        # The volta bracket marks the end of a repeated section on the
        # previous line. Use the previous staff's measures, or if the current
        # staff has measures, the measures just before the current staff's first.
        prev_si = si - 1
        prev_measures = staff_measures.get(prev_si, [])
        curr_measures = staff_measures.get(si, [])

        if prev_measures and curr_measures:
            # If there's a big gap between the previous staff's last measure
            # and this staff's first measure, the volta is in the gap
            # (on staves our detector missed). Use curr_measures to infer.
            prev_last = max(prev_measures)
            curr_first = min(curr_measures)
            if curr_first - prev_last > measures_per_staff * 1.5:
                # Big gap — volta is for measures in the gap, not on prev staff
                last_m = curr_first - 1
                first_m = max(1, last_m - int(measures_per_staff) + 1)
            else:
                # Small gap — volta belongs to the end of previous staff
                first_m = min(prev_measures)
                last_m = max(prev_measures)
        elif prev_measures:
            first_m = min(prev_measures)
            last_m = max(prev_measures)
        elif curr_measures:
            # Infer: the volta is for measures just before this staff's first measure
            curr_first = min(curr_measures)
            last_m = curr_first - 1
            first_m = max(1, last_m - int(measures_per_staff) + 1)
        else:
            first_m = int(si * measures_per_staff) + 1
            last_m = min(int((si + 1) * measures_per_staff), total_measures)

        volta_groups.append({
            "staff_index": si,
            "volta_numbers": volta_numbers,
            "first_measure": first_m,
            "last_measure": last_m,
        })

    if not volta_groups:
        return repeat_markers

    # For each volta group, find matching repeat OR create new one.
    # Match criterion: the repeat's end measure must be WITHIN this staff's
    # measure range (based on real OCR-detected measure numbers).
    result = list(repeat_markers)
    used_repeats = set()

    for vg in volta_groups:
        best_match = None
        for i, rm in enumerate(result):
            if i in used_repeats:
                continue
            # The repeat's end must fall within this staff's measure range
            if vg["first_measure"] <= rm["end_measure"] <= vg["last_measure"]:
                best_match = i
                break

        if best_match is not None:
            rm = result[best_match]
            voltas = _build_volta_endings(
                vg["volta_numbers"], rm["start_measure"], rm["end_measure"]
            )
            result[best_match] = dict(rm, volta_endings=voltas)
            used_repeats.add(best_match)
            print(f"[volta] Repeat m{rm['start_measure']}-m{rm['end_measure']}: "
                  f"added volta endings {voltas}")
        else:
            # No matching repeat — create a new one
            end = vg["last_measure"]
            start = vg["first_measure"]
            voltas = _build_volta_endings(vg["volta_numbers"], start, end)

            new_rm = {
                "start_measure": start,
                "end_measure": end,
                "repeat_count": 1,
                "volta_endings": voltas,
            }
            result.append(new_rm)
            print(f"[volta] Created NEW repeat m{start}-m{end} "
                  f"with volta endings {voltas}")

    return result


def _find_staves(gray: np.ndarray, w: int, h: int) -> list[dict]:
    """Find staff positions using morphological line detection."""
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )

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
                top = current_staff[0]
                bottom = current_staff[-1]
                spacing = (bottom - top) / max(1, len(current_staff) - 1)
                staves.append({
                    "top": top, "bottom": bottom,
                    "spacing": spacing, "index": len(staves),
                })
            current_staff = [groups[i]]
    if len(current_staff) >= 4:
        top = current_staff[0]
        bottom = current_staff[-1]
        spacing = (bottom - top) / max(1, len(current_staff) - 1)
        staves.append({
            "top": top, "bottom": bottom,
            "spacing": spacing, "index": len(staves),
        })

    return staves


def _find_volta_texts_and_measures(img: np.ndarray, staves: list[dict]) -> tuple[list[dict], dict[int, list[int]]]:
    """
    Find volta text ('1', '2') and measure numbers above each staff.
    Returns (volta_texts, staff_measures) where staff_measures maps
    staff_index → list of measure numbers found above that staff.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        print("[volta] rapidocr not available")
        return [], {}

    ocr = RapidOCR()
    volta_texts = []
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

        zone = img[zone_top:zone_bottom, :, :]
        result, _ = ocr(zone)
        if not result:
            continue

        zone_height = zone_bottom - zone_top
        volta_y_max = zone_top + zone_height * 0.6

        measures_on_staff = []

        for box, text, conf in result:
            text_clean = text.strip().rstrip(".,;:")
            box_y = int(min(p[1] for p in box)) + zone_top
            box_x = int(min(p[0] for p in box))
            box_width = int(max(p[0] for p in box)) - box_x

            # Volta text: single digit 1-3, in upper portion, narrow
            # OCR sometimes reads "1" as "I" (capital i) or "l" (lowercase L)
            volta_number = None
            if text_clean in ("1", "I", "l") and box_y < volta_y_max:
                volta_number = 1
            elif text_clean in ("2",) and box_y < volta_y_max:
                volta_number = 2
            elif text_clean in ("3",) and box_y < volta_y_max:
                volta_number = 3

            if volta_number is not None:
                if box_width < spacing * 4:
                    volta_texts.append({
                        "number": volta_number,
                        "x": box_x,
                        "y": box_y,
                        "staff_index": staff["index"],
                        "confidence": conf,
                    })

            # Measure number: 2-3 digit number in lower portion (near staff)
            elif text_clean.isdigit() and 2 <= len(text_clean) <= 3 and box_y >= volta_y_max:
                num = int(text_clean)
                if 1 <= num <= 200:  # reasonable measure range
                    measures_on_staff.append(num)

        if measures_on_staff:
            staff_measures[staff["index"]] = sorted(measures_on_staff)

    return volta_texts, staff_measures


def _build_volta_endings(
    volta_numbers: list[int],
    start_measure: int,
    end_measure: int,
) -> dict[str, list[int]]:
    """Build volta endings dict from detected volta numbers."""
    num_voltas = len(volta_numbers)
    measures_for_voltas = min(num_voltas, end_measure - start_measure + 1)
    volta_start = end_measure - measures_for_voltas + 1

    volta_endings = {}
    for i, vnum in enumerate(volta_numbers):
        measure = volta_start + i
        volta_endings[str(measure)] = [vnum]

    return volta_endings
