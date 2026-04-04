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
    clef_key_boxes: list[dict],
) -> str:
    """
    Find and classify the clef for a specific staff.

    The clef is the leftmost clefs_keys symbol within the staff's
    vertical range, positioned near the left edge of the staff.

    Args:
        image: Original image
        staff_info: Staff position dict with min_x, max_x, min_y, max_y, unit_size
        clef_key_boxes: All detected clef/key bounding boxes

    Returns:
        "treble", "bass", or "alto"
    """
    staff_min_y = staff_info["min_y"]
    staff_max_y = staff_info["max_y"]
    staff_min_x = staff_info["min_x"]
    staff_height = staff_max_y - staff_min_y
    staff_width = staff_info["max_x"] - staff_min_x

    # Find clef/key boxes within this staff's vertical range
    # and near the left edge (first 20% of staff width)
    candidates = []
    margin_y = staff_height * 0.5  # Allow some vertical overshoot
    left_cutoff = staff_min_x + staff_width * 0.2

    for box in clef_key_boxes:
        bx = box.get("x", 0)
        by = box.get("y", 0)
        bh = box.get("height", 0)

        # Check vertical overlap with staff
        box_top = by - bh / 2
        box_bot = by + bh / 2
        if box_bot < staff_min_y - margin_y or box_top > staff_max_y + margin_y:
            continue

        # Check horizontal position (near left edge)
        if bx > left_cutoff:
            continue

        candidates.append(box)

    if not candidates:
        return "treble"  # Default

    # Pick the leftmost candidate (that's the clef; key sig is to its right)
    clef_box = min(candidates, key=lambda b: b.get("x", 0))

    return classify_clef_from_image(image, clef_box, staff_info)
