"""
Visual clef classification from image regions.

Classifies clef symbols detected by HOMR's segmentation as treble (G),
bass (F), or alto (C) based on visual features of the clef image.

The three clef types have distinctive shapes:
- Treble (G clef): Very tall, extends well above and below the staff.
  Height typically 1.8-3x the staff height.
- Bass (F clef): Contained within the staff height. Has two dots.
  Height typically 0.5-1.2x staff height.
- Alto (C clef): About the same height as the staff. Symmetric.
  Height typically 0.8-1.3x staff height.

This code is licensed under AGPL-3.0 to comply with HOMR's license.
"""

import cv2
import numpy as np


def classify_clef_from_image(
    image: np.ndarray,
    clef_box: dict,
    staff_info: dict,
) -> str:
    """
    Classify a clef symbol from its image region.

    Args:
        image: Original image (BGR or grayscale)
        clef_box: Bounding box dict with x, y, width, height (pixel coords)
        staff_info: Staff dict with min_y, max_y, unit_size

    Returns:
        "treble", "bass", or "alto"
    """
    staff_height = staff_info["max_y"] - staff_info["min_y"]
    if staff_height <= 0:
        return "treble"

    clef_height = clef_box.get("height", 0)
    height_ratio = clef_height / staff_height

    # Primary discriminator: height relative to staff
    # Treble clefs are distinctively tall (extend above and below staff)
    if height_ratio > 1.5:
        return "treble"
    elif height_ratio < 0.7:
        return "bass"

    # For ambiguous cases, analyze the image region
    cx = clef_box.get("x", 0)
    cy = clef_box.get("y", 0)
    w = clef_box.get("width", 0)
    h = clef_box.get("height", 0)

    x1 = max(0, int(cx - w * 0.6))
    y1 = max(0, int(cy - h * 0.6))
    x2 = min(image.shape[1], int(cx + w * 0.6))
    y2 = min(image.shape[0], int(cy + h * 0.6))

    if x2 <= x1 or y2 <= y1:
        return "treble"

    region = image[y1:y2, x1:x2]

    # Convert to grayscale and binarize
    if len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Analyze vertical symmetry (alto clefs are very symmetric)
    rh, rw = binary.shape
    if rw > 4:
        left_half = binary[:, : rw // 2]
        right_half = cv2.flip(binary[:, rw // 2 :], 1)
        # Resize to match
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_crop = left_half[:, :min_w]
        right_crop = right_half[:, :min_w]
        symmetry = np.sum(left_crop == right_crop) / max(left_crop.size, 1)
    else:
        symmetry = 0.5

    # Alto clefs are highly symmetric (>0.8)
    if symmetry > 0.8 and 0.8 < height_ratio < 1.4:
        return "alto"

    # Check aspect ratio
    aspect = h / max(w, 1)
    if aspect > 2.5:
        return "treble"
    elif aspect < 1.5:
        return "bass"

    # Default to treble (most common)
    return "treble"


def find_clef_for_staff(
    image: np.ndarray,
    staff_info: dict,
    clef_key_boxes: list[dict] | None = None,
) -> tuple[str, float]:
    """
    Find and classify the clef for a specific staff.

    Uses two strategies:
    1. If clef_key_boxes are detected, find the leftmost one on this staff
    2. If no boxes detected, crop the left edge of the staff image directly
       (the clef symbol is always at the far left)

    Args:
        image: Original image
        staff_info: Staff position dict with min_x, max_x, min_y, max_y, unit_size
        clef_key_boxes: All detected clef/key bounding boxes (may be empty)

    Returns:
        (clef, confidence) — e.g. ("treble", 0.95) or ("bass", 0.6)
        confidence > 0.7 = high, 0.4-0.7 = medium, < 0.4 = low
    """
    staff_min_y = staff_info["min_y"]
    staff_max_y = staff_info["max_y"]
    staff_min_x = staff_info["min_x"]
    staff_height = staff_max_y - staff_min_y
    staff_width = staff_info["max_x"] - staff_min_x

    # Strategy 1: Use detected clef/key boxes
    if clef_key_boxes:
        candidates = []
        margin_y = staff_height * 0.5
        left_cutoff = staff_min_x + staff_width * 0.2

        for box in clef_key_boxes:
            bx = box.get("x", 0)
            by = box.get("y", 0)
            bh = box.get("height", 0)

            box_top = by - bh / 2
            box_bot = by + bh / 2
            if box_bot < staff_min_y - margin_y or box_top > staff_max_y + margin_y:
                continue
            if bx > left_cutoff:
                continue
            candidates.append(box)

        if candidates:
            clef_box = min(candidates, key=lambda b: b.get("x", 0))
            # Height ratio gives high confidence
            h_ratio = clef_box.get("height", 0) / max(staff_height, 1)
            if h_ratio > 1.8:
                return "treble", 0.95
            elif h_ratio < 0.6:
                return "bass", 0.90
            else:
                result = classify_clef_from_image(image, clef_box, staff_info)
                return result, 0.7  # Medium confidence from image analysis

    # Strategy 2: Crop the left edge of the staff image directly
    return _classify_clef_from_staff_edge(image, staff_info)


def _classify_clef_from_staff_edge(
    image: np.ndarray,
    staff_info: dict,
) -> tuple[str, float]:
    """
    Classify clef by analyzing the left edge of the staff image.

    Treble clefs (G clef) extend well above and below the staff lines.
    Bass clefs (F clef) stay mostly within the staff area.

    Returns (clef, confidence).
    """
    staff_min_x = staff_info["min_x"]
    staff_min_y = staff_info["min_y"]
    staff_max_y = staff_info["max_y"]
    staff_height = staff_max_y - staff_min_y
    staff_width = staff_info["max_x"] - staff_min_x
    unit_size = staff_info.get("unit_size", staff_height / 4)

    # Crop the clef region: skip system barline (first unit_size),
    # then capture ~3 unit_sizes (the clef symbol width).
    # Using unit_size is more robust than a percentage of staff width.
    margin = staff_height * 0.6  # Allow for treble clef extending above/below
    barline_skip = max(unit_size * 0.5, 5)  # Skip past the system barline
    clef_width = max(unit_size * 3.5, 40)   # Clef symbol width
    x1 = max(0, int(staff_min_x + barline_skip))
    x2 = min(image.shape[1], int(staff_min_x + barline_skip + clef_width))
    y1 = max(0, int(staff_min_y - margin))
    y2 = min(image.shape[0], int(staff_max_y + margin))

    if x2 <= x1 or y2 <= y1:
        return "treble"

    region = image[y1:y2, x1:x2]

    # Convert to grayscale and binarize
    if len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Measure vertical extent of ink in the clef region
    # Project horizontally: sum of black pixels per row
    row_sums = np.sum(binary > 0, axis=1)
    threshold = binary.shape[1] * 0.05  # At least 5% of width has ink

    ink_rows = np.where(row_sums > threshold)[0]
    if len(ink_rows) == 0:
        return "treble", 0.1  # Very low confidence

    # Filter: if most columns have full-height ink, this is a barline/bracket/
    # page margin, NOT a clef symbol. Reduce to very low confidence.
    col_sums = np.sum(binary > 0, axis=0)
    cols_with_full_ink = np.sum(col_sums > binary.shape[0] * 0.3)
    if cols_with_full_ink > binary.shape[1] * 0.5:
        # More than half the columns are full-height ink → barline/bracket
        return "treble", 0.1  # Can't determine clef from this region

    ink_top = ink_rows[0]
    ink_bottom = ink_rows[-1]
    ink_height = ink_bottom - ink_top

    # Staff position in the cropped region
    staff_top_in_crop = int(staff_min_y - y1)
    staff_bot_in_crop = int(staff_max_y - y1)

    # How much does the ink extend above and below the staff?
    above_staff = max(0, staff_top_in_crop - ink_top)
    below_staff = max(0, ink_bottom - staff_bot_in_crop)
    total_extension = above_staff + below_staff

    # Treble clef: extends significantly above AND below staff
    # Bass clef: stays mostly within staff bounds
    extension_ratio = total_extension / max(staff_height, 1)

    if extension_ratio > 0.8:
        return "treble", 0.95  # Very clearly extends beyond staff
    elif extension_ratio > 0.6:
        return "treble", 0.75
    elif extension_ratio < 0.15:
        return "bass", 0.90  # Very clearly contained within staff
    elif extension_ratio < 0.3:
        return "bass", 0.70

    # Ambiguous zone (0.3-0.6) — lower confidence
    if above_staff > below_staff * 1.5:
        return "treble", 0.5
    elif below_staff > above_staff * 1.5:
        return "bass", 0.5

    # Still ambiguous — analyze ink density in upper vs lower half
    mid = (staff_top_in_crop + staff_bot_in_crop) // 2
    upper_ink = np.sum(binary[:mid, :] > 0)
    lower_ink = np.sum(binary[mid:, :] > 0)

    if upper_ink > lower_ink * 1.3:
        return "treble", 0.4
    elif lower_ink > upper_ink * 1.3:
        return "bass", 0.4

    return "treble", 0.2  # Very uncertain, default
