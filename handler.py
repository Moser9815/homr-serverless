"""
HOMR OMR — RunPod Serverless Handler

Calls HOMR's Python API directly (not CLI) to access internal staff/barline
data for accurate volta detection and note pixel positions.

This code is licensed under AGPL-3.0 to comply with HOMR's license.
"""

import base64
import io
import os
import tempfile
import time
import traceback

from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

import runpod

from parse_musicxml import parse_musicxml_to_json
from detect_voltas import detect_voltas


def decode_image(base64_data: str) -> Image.Image:
    """Decode base64 image data to PIL Image, handling EXIF orientation."""
    image_bytes = base64.b64decode(base64_data)
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def run_homr_api(image_path: str, use_gpu: bool = True) -> tuple[str, list[dict], list[dict]]:
    """
    Run HOMR via Python API. Returns:
    - musicxml_path: path to the output MusicXML file
    - staff_info: list of staff dicts with pixel positions
    - barline_info: list of barline dicts with pixel positions
    """
    from homr.main import (
        ProcessingConfig,
        detect_staffs_in_image,
        parse_staffs,
    )
    from homr.transformer.configs import Config as TransformerConfig
    from homr.simple_logging import eprint
    from homr.music_xml_generator import generate_xml, XmlGeneratorArguments

    print(f"[HOMR] Processing {image_path} via Python API")
    start = time.time()

    config = ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        use_gpu_inference=use_gpu,
    )

    # Step 1: Detect staffs (segmentation + staff detection)
    multi_staffs, image, debug, title_future = detect_staffs_in_image(image_path, config)

    elapsed_detect = time.time() - start
    print(f"[HOMR] Staff detection: {elapsed_detect:.1f}s")

    # Step 3: Run transformer (symbol recognition)
    transformer_config = TransformerConfig()
    transformer_config.use_gpu_inference = use_gpu

    result_staffs = parse_staffs(
        debug, multi_staffs, image,
        selected_staff=-1, config=transformer_config,
    )

    title = ""
    try:
        title = title_future.result(30)
    except Exception:
        pass

    # Step 4: Extract staff pixel data, scaled to original image coordinates
    # HOMR resizes the image before processing — need to scale back
    import cv2 as _cv2
    original = _cv2.imread(image_path)
    orig_h, orig_w = original.shape[:2]
    proc_h, proc_w = image.shape[:2]
    scale_x = orig_w / proc_w
    scale_y = orig_h / proc_h

    staff_info = []
    barline_info = []
    staff_counter = 0

    for ms_idx, multi_staff in enumerate(multi_staffs):
        for s_idx, staff in enumerate(multi_staff.staffs):
            staff_data = {
                "multi_staff": ms_idx,
                "staff": staff_counter,
                "min_x": float(staff.min_x * scale_x),
                "max_x": float(staff.max_x * scale_x),
                "min_y": float(staff.min_y * scale_y),
                "max_y": float(staff.max_y * scale_y),
                "unit_size": float(staff.average_unit_size * scale_y),
                "is_grand": staff.is_grandstaff,
            }
            staff_info.append(staff_data)

            # Extract barlines from staff symbols
            for barline in staff.get_bar_lines():
                cx, cy = barline.box.center
                bw, bh = barline.box.size
                barline_info.append({
                    "staff_idx": staff_counter,
                    "x": float(cx * scale_x),
                    "y": float(cy * scale_y),
                    "width": float(bw * scale_x),
                    "height": float(bh * scale_y),
                })

            staff_counter += 1

    print(f"[HOMR] Extracted {len(staff_info)} staves, {len(barline_info)} barlines "
          f"(scale: {scale_x:.3f}x{scale_y:.3f})")

    # Step 5: Generate MusicXML
    xml_generator_args = XmlGeneratorArguments()
    xml = generate_xml(xml_generator_args, result_staffs, title)

    musicxml_path = os.path.splitext(image_path)[0] + ".musicxml"
    xml.write(musicxml_path)

    elapsed_total = time.time() - start
    print(f"[HOMR] Total: {elapsed_total:.1f}s, {len(result_staffs)} parsed staves")

    return musicxml_path, staff_info, barline_info


def handler(event):
    """RunPod serverless handler."""
    try:
        job_input = event.get("input", {})

        image_data = job_input.get("image")
        if not image_data:
            return {"error": "Missing required 'image' field (base64-encoded image)"}

        clef = job_input.get("clef", "treble")
        tempo = int(job_input.get("tempo", 120))
        time_signature = job_input.get("time_signature", "4/4")

        start_time = time.time()

        img = decode_image(image_data)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp, format="PNG")
            tmp_path = tmp.name

        try:
            # Run HOMR via Python API (returns staff/barline data)
            musicxml_path, staff_info, barline_info = run_homr_api(tmp_path, use_gpu=True)

            with open(musicxml_path, "r", encoding="utf-8") as f:
                musicxml_content = f.read()

            parsed = parse_musicxml_to_json(
                musicxml_content,
                default_clef=clef,
                default_tempo=tempo,
                default_time_signature=time_signature,
            )

            # Post-process: detect voltas using HOMR's own staff/barline data
            repeat_markers = parsed.get("repeat_markers", [])
            volta_status = "skipped"
            try:
                total_m = parsed.get("metadata", {}).get("total_measures", 0)
                repeat_markers = detect_voltas(
                    tmp_path, repeat_markers, total_m,
                    staff_info=staff_info,
                    barline_info=barline_info,
                )
                has_voltas = any(rm.get("volta_endings") for rm in repeat_markers)
                volta_status = "detected" if has_voltas else "none_found"
            except Exception as e:
                volta_status = f"error: {e}"
                print(f"[volta] Detection failed (non-fatal): {e}")
                traceback.print_exc()

            processing_time = time.time() - start_time

            notes = parsed.get("notes", [])
            rests = parsed.get("rests", [])
            metadata = parsed.get("metadata", {})
            metadata["processing_time"] = round(processing_time, 2)
            metadata["detection_method"] = "homr"
            metadata["volta_detection"] = volta_status
            metadata["staves_detected"] = len(staff_info)
            metadata["barlines_detected"] = len(barline_info)


            return {
                "success": True,
                "notes": notes,
                "rests": rests,
                "repeat_markers": repeat_markers,
                "note_count": len(notes),
                "rest_count": len(rests),
                "repeat_count": len(repeat_markers),
                "metadata": metadata,
                "staff_positions": staff_info,
                "barline_positions": barline_info,
                "musicxml": musicxml_content,
                "message": f"HOMR processed {len(notes)} notes, {len(rests)} rests, "
                           f"{len(repeat_markers)} repeats in {processing_time:.1f}s",
            }

        finally:
            for path in [tmp_path, tmp_path.replace(".png", ".musicxml"),
                         tmp_path.replace(".png", ".xml")]:
                if os.path.exists(path):
                    os.unlink(path)

    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
