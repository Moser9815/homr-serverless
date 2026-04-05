"""
Clef classification from image regions.

Uses an SVM trained on 880 MUSCIMA++ clef samples (98.3% accuracy)
to classify clef symbols as treble (G), bass (F), or alto (C).

Falls back to a heuristic edge-based approach if the model is unavailable.

This code is licensed under AGPL-3.0 to comply with HOMR's license.
"""

import os
import pickle

import cv2
import numpy as np

# Model path (relative to this file)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clef_model.pkl")
_model_cache = None

# Map MUSCIMA++ class names to our clef names
_CLASS_TO_CLEF = {"gClef": "treble", "fClef": "bass", "cClef": "alto"}


def _load_model():
    """Load the trained clef classifier model."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not os.path.exists(_MODEL_PATH):
        return None
    with open(_MODEL_PATH, "rb") as f:
        _model_cache = pickle.load(f)
    return _model_cache


def _classify_with_model(region_gray: np.ndarray) -> tuple[str, float]:
    """Classify a clef region using the trained SVM model."""
    model_data = _load_model()
    if model_data is None:
        return "treble", 0.1

    model = model_data["model"]
    class_names = model_data["class_names"]
    input_size = model_data.get("input_size", 64)

    # Resize to model input size
    resized = cv2.resize(region_gray, (input_size, input_size), interpolation=cv2.INTER_AREA)
    features = (resized.astype(np.float32) / 255.0).flatten().reshape(1, -1)

    # Predict with probability
    pred_class = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = float(proba[pred_class])

    clef_name = _CLASS_TO_CLEF.get(class_names[pred_class], "treble")
    return clef_name, confidence


def classify_from_dewarped(dewarped_staff: np.ndarray) -> tuple[str, float]:
    """
    Classify clef from a dewarped staff image.

    The dewarped image is a clean horizontal strip where the clef
    is at the far left. Crops the leftmost 8% for classification.

    Args:
        dewarped_staff: Grayscale dewarped staff image from HOMR

    Returns:
        (clef, confidence) — e.g. ("bass", 0.74)
    """
    if dewarped_staff is None or dewarped_staff.size == 0:
        return "treble", 0.1

    gray = dewarped_staff if len(dewarped_staff.shape) == 2 else cv2.cvtColor(dewarped_staff, cv2.COLOR_BGR2GRAY)

    # Normalize to uint8 if needed
    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)

    # Crop leftmost 8% — captures clef + key sig, avoids time sig/barlines
    w = gray.shape[1]
    crop = gray[:, : max(32, int(w * 0.08))]

    return _classify_with_model(crop)


def find_clef_for_staff(
    image: np.ndarray,
    staff_info: dict,
    clef_key_boxes: list[dict] | None = None,
    dewarped_staff: np.ndarray | None = None,
) -> tuple[str, float]:
    """
    Find and classify the clef for a specific staff.

    Best results come from passing a dewarped staff image (from HOMR's
    prepare_staff_image). Falls back to cropping the original image.

    Args:
        image: Original image (BGR or grayscale)
        staff_info: Staff position dict with min_x, max_x, min_y, max_y, unit_size
        clef_key_boxes: Detected clef/key bounding boxes (used if available)
        dewarped_staff: Dewarped staff image from HOMR (best input)

    Returns:
        (clef, confidence) — e.g. ("treble", 0.97)
    """
    # Best path: use the dewarped staff image
    if dewarped_staff is not None:
        return classify_from_dewarped(dewarped_staff)

    staff_min_y = staff_info["min_y"]
    staff_max_y = staff_info["max_y"]
    staff_min_x = staff_info["min_x"]
    staff_height = staff_max_y - staff_min_y
    staff_width = staff_info["max_x"] - staff_min_x
    unit_size = staff_info.get("unit_size", staff_height / 4)

    # Strategy 1: Use detected clef/key bounding boxes if available
    if clef_key_boxes:
        candidates = []
        margin_y = staff_height * 0.5
        left_cutoff = staff_min_x + staff_width * 0.2

        for box in clef_key_boxes:
            bx, by = box.get("x", 0), box.get("y", 0)
            bh = box.get("height", 0)
            if by + bh / 2 < staff_min_y - margin_y or by - bh / 2 > staff_max_y + margin_y:
                continue
            if bx > left_cutoff:
                continue
            candidates.append(box)

        if candidates:
            clef_box = min(candidates, key=lambda b: b.get("x", 0))
            # Crop the clef region
            bx, by = clef_box["x"], clef_box["y"]
            bw, bh = clef_box["width"], clef_box["height"]
            pad = max(bw, bh) * 0.1
            x1 = max(0, int(bx - bw / 2 - pad))
            y1 = max(0, int(by - bh / 2 - pad))
            x2 = min(image.shape[1], int(bx + bw / 2 + pad))
            y2 = min(image.shape[0], int(by + bh / 2 + pad))

            region = image[y1:y2, x1:x2]
            if region.size > 0:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
                return _classify_with_model(gray)

    # Strategy 2: Crop the left edge of the staff image
    # Skip past system barline, capture the clef symbol area
    barline_skip = max(unit_size * 0.5, 5)
    clef_width = max(unit_size * 3.5, 40)
    margin = staff_height * 0.6

    x1 = max(0, int(staff_min_x + barline_skip))
    x2 = min(image.shape[1], int(staff_min_x + barline_skip + clef_width))
    y1 = max(0, int(staff_min_y - margin))
    y2 = min(image.shape[0], int(staff_max_y + margin))

    if x2 <= x1 or y2 <= y1:
        return "treble", 0.1

    region = image[y1:y2, x1:x2]
    if len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region

    # Check for barline/bracket pattern (full-height vertical lines)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_sums = np.sum(binary > 0, axis=0)
    cols_with_full_ink = np.sum(col_sums > binary.shape[0] * 0.3)
    if cols_with_full_ink > binary.shape[1] * 0.5:
        return "treble", 0.1  # Barline/bracket, not a clef

    return _classify_with_model(gray)
