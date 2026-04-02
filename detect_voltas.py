"""
Volta Bracket Detector — Post-processing for HOMR OMR output.

HOMR's transformer doesn't reliably detect volta brackets (1st/2nd endings)
due to training data scarcity. This module uses OpenCV + OCR to detect them
from the original image and merge them with HOMR's repeat barlines.

This code is licensed under AGPL-3.0 to comply with HOMR's license.
"""

import cv2
import numpy as np
from typing import Optional


def detect_voltas(
    image_path: str,
    repeat_markers: list[dict],
) -> list[dict]:
    """
    Detect volta brackets in a sheet music image and merge with repeat markers.

    Args:
        image_path: Path to the sheet music image
        repeat_markers: Repeat markers from HOMR's MusicXML (list of dicts
            with start_measure, end_measure, repeat_count, volta_endings)

    Returns:
        Updated repeat_markers with volta_endings populated where detected.
    """
    if not repeat_markers:
        return repeat_markers

    img = cv2.imread(image_path)
    if img is None:
        print(f"[volta] Could not read image: {image_path}")
        return repeat_markers

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Step 1: Find staff positions
    staves = _find_staves(gray, w, h)
    if not staves:
        print("[volta] No staves found")
        return repeat_markers

    print(f"[volta] Found {len(staves)} staves")

    # Step 2: Find volta text ("1", "2", "1.", "2.") above each staff
    volta_texts = _find_volta_texts(img, staves)
    if not volta_texts:
        print("[volta] No volta text detected")
        return repeat_markers

    print(f"[volta] Found {len(volta_texts)} volta texts: {volta_texts}")

    # Step 3: Map volta texts to staves → measures → repeat markers
    updated = _merge_voltas_into_repeats(volta_texts, staves, repeat_markers)
    return updated


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

    # Group consecutive rows into individual lines
    groups = []
    current = [staff_rows[0]]
    for i in range(1, len(staff_rows)):
        if staff_rows[i] - staff_rows[i - 1] <= 5:
            current.append(staff_rows[i])
        else:
            groups.append(int(np.mean(current)))
            current = [staff_rows[i]]
    groups.append(int(np.mean(current)))

    # Group lines into 5-line staves
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
                    "top": top,
                    "bottom": bottom,
                    "spacing": spacing,
                    "index": len(staves),
                })
            current_staff = [groups[i]]
    if len(current_staff) >= 4:
        top = current_staff[0]
        bottom = current_staff[-1]
        spacing = (bottom - top) / max(1, len(current_staff) - 1)
        staves.append({
            "top": top,
            "bottom": bottom,
            "spacing": spacing,
            "index": len(staves),
        })

    return staves


def _find_volta_texts(img: np.ndarray, staves: list[dict]) -> list[dict]:
    """Find "1", "2" (or "1.", "2.") text in the region above each staff."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        print("[volta] rapidocr not available, skipping volta detection")
        return []

    ocr = RapidOCR()
    volta_texts = []
    h, w = img.shape[:2]

    for staff in staves:
        spacing = staff["spacing"]
        zone_top = max(0, staff["top"] - int(spacing * 8))
        zone_bottom = staff["top"]

        if zone_bottom - zone_top < 10:
            continue

        zone = img[zone_top:zone_bottom, :, :]
        result, _ = ocr(zone)

        if not result:
            continue

        # Measure numbers vs volta numbers:
        # - Volta numbers ("1", "2") sit HIGHER (closer to the bracket line, farther from staff)
        # - Measure numbers sit LOWER (closer to the staff)
        # Volta numbers are typically in the top 40% of the zone, measure numbers in the bottom 40%
        zone_height = zone_bottom - zone_top
        # Volta numbers sit high above the staff (near the bracket line).
        # Measure numbers sit low (right above the staff lines).
        # The dividing line is roughly 60% down from the zone top.
        volta_y_max = zone_top + zone_height * 0.6

        for box, text, conf in result:
            text_clean = text.strip().rstrip(".,;:")
            box_y = int(min(p[1] for p in box)) + zone_top
            box_x = int(min(p[0] for p in box))
            box_x_max = int(max(p[0] for p in box))
            box_width = box_x_max - box_x

            # Must be a small number (1-3) in the upper portion of the zone
            if text_clean in ("1", "2", "3") and box_y < volta_y_max:
                # Additional filter: volta numbers are small text (narrow width)
                # Measure numbers like "14", "19" are wider
                if box_width < spacing * 4:
                    volta_texts.append({
                        "number": int(text_clean),
                        "x": box_x,
                        "y": box_y,
                        "staff_index": staff["index"],
                        "confidence": conf,
                    })

    return volta_texts


def _merge_voltas_into_repeats(
    volta_texts: list[dict],
    staves: list[dict],
    repeat_markers: list[dict],
) -> list[dict]:
    """Map detected volta texts to repeat markers based on position."""
    if not volta_texts or not repeat_markers:
        return repeat_markers

    # Group volta texts by staff
    by_staff = {}
    for vt in volta_texts:
        si = vt["staff_index"]
        by_staff.setdefault(si, []).append(vt)

    # For each staff with volta texts, sort by x position
    for si in by_staff:
        by_staff[si].sort(key=lambda v: v["x"])

    # We need to figure out which staff index corresponds to which measures.
    # Assumption: staves are in order, each staff contains roughly
    # total_measures / num_staves measures.
    # We map each repeat marker to the staff that contains its measures.
    num_staves = len(staves)
    if num_staves == 0:
        return repeat_markers

    # Estimate measures per staff from repeat marker data
    all_measures = set()
    for rm in repeat_markers:
        for m in range(rm["start_measure"], rm["end_measure"] + 1):
            all_measures.add(m)
    total_measures = max(all_measures) if all_measures else 1
    measures_per_staff = max(1, total_measures / num_staves)

    updated = []
    for rm in repeat_markers:
        rm_copy = dict(rm)

        # Which staff contains this repeat's end measure?
        # (volta brackets appear at the end of the repeated section)
        end_staff_idx = min(
            int((rm["end_measure"] - 1) / measures_per_staff),
            num_staves - 1
        )

        # Check if this staff has volta texts
        staff_voltas = by_staff.get(end_staff_idx, [])
        if not staff_voltas:
            # Also check the staff before (volta might span the barline)
            staff_voltas = by_staff.get(end_staff_idx - 1, [])

        if staff_voltas:
            # Build volta endings map
            # Sort by volta number, assign to measures near the end of the repeat
            volta_numbers = sorted(set(v["number"] for v in staff_voltas))
            if len(volta_numbers) >= 2:
                # We have at least a 1st and 2nd ending
                # Assign them to the last N measures of the repeat range
                end = rm["end_measure"]
                start = rm["start_measure"]
                num_voltas = len(volta_numbers)

                # Each volta gets roughly equal measures at the end
                # Simple heuristic: 2 endings, last 2 measures
                measures_for_voltas = min(num_voltas, end - start + 1)
                volta_start = end - measures_for_voltas + 1

                volta_endings = {}
                for i, vnum in enumerate(volta_numbers):
                    measure = volta_start + i
                    volta_endings[str(measure)] = [vnum]

                rm_copy["volta_endings"] = volta_endings
                print(f"[volta] Repeat m{start}-m{end}: added volta endings {volta_endings}")

        updated.append(rm_copy)

    return updated
