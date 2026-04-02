"""
HOMR OMR — RunPod Serverless Handler

Receives a base64-encoded sheet music image, runs HOMR optical music recognition,
parses the MusicXML output, and returns structured note/rest data.

This code is licensed under AGPL-3.0 to comply with HOMR's license.
"""

import base64
import json
import os
import subprocess
import tempfile
import time
import traceback

from PIL import Image, ImageOps
import io

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # HEIC support optional

import runpod

from parse_musicxml import parse_musicxml_to_json
from detect_voltas import detect_voltas


def decode_image(base64_data: str) -> Image.Image:
    """Decode base64 image data to PIL Image, handling EXIF orientation."""
    image_bytes = base64.b64decode(base64_data)
    img = Image.open(io.BytesIO(image_bytes))

    # Apply EXIF orientation (handles all 8 orientations)
    img = ImageOps.exif_transpose(img)

    # Convert to RGB if needed
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    return img


def run_homr(image_path: str, use_gpu: bool = True) -> str:
    """Run HOMR on an image file, return path to the output MusicXML."""
    gpu_flag = "auto" if use_gpu else "no"
    cmd = ["homr", "--gpu", gpu_flag, image_path]

    print(f"[HOMR] Running: {' '.join(cmd)}")
    start = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )

    elapsed = time.time() - start
    print(f"[HOMR] Completed in {elapsed:.1f}s, exit code: {result.returncode}")

    if result.stdout:
        print(f"[HOMR] stdout: {result.stdout[:500]}")
    if result.stderr:
        print(f"[HOMR] stderr: {result.stderr[:500]}")

    if result.returncode != 0:
        raise RuntimeError(f"HOMR failed (exit {result.returncode}): {result.stderr}")

    # HOMR writes output alongside input: image.png → image.musicxml
    base_name = os.path.splitext(image_path)[0]
    musicxml_path = base_name + ".musicxml"

    if not os.path.exists(musicxml_path):
        # Also check for .xml extension
        alt_path = base_name + ".xml"
        if os.path.exists(alt_path):
            musicxml_path = alt_path
        else:
            raise RuntimeError(
                f"HOMR did not produce output file. "
                f"Expected: {musicxml_path}. "
                f"Files in dir: {os.listdir(os.path.dirname(image_path))}"
            )

    return musicxml_path


def handler(event):
    """RunPod serverless handler."""
    try:
        job_input = event.get("input", {})

        # Required: base64-encoded image
        image_data = job_input.get("image")
        if not image_data:
            return {"error": "Missing required 'image' field (base64-encoded image)"}

        # Optional parameters (passed through to response metadata)
        clef = job_input.get("clef", "treble")
        tempo = int(job_input.get("tempo", 120))
        time_signature = job_input.get("time_signature", "4/4")

        start_time = time.time()

        # Decode image
        img = decode_image(image_data)

        # Save to temp file (HOMR reads from disk)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp, format="PNG")
            tmp_path = tmp.name

        try:
            # Run HOMR
            musicxml_path = run_homr(tmp_path)

            # Read MusicXML output
            with open(musicxml_path, "r", encoding="utf-8") as f:
                musicxml_content = f.read()

            # Parse MusicXML to structured JSON
            parsed = parse_musicxml_to_json(
                musicxml_content,
                default_clef=clef,
                default_tempo=tempo,
                default_time_signature=time_signature,
            )

            processing_time = time.time() - start_time

            # Post-process: detect volta brackets HOMR missed
            repeat_markers = parsed.get("repeat_markers", [])
            volta_status = "skipped"
            if repeat_markers:
                try:
                    repeat_markers = detect_voltas(tmp_path, repeat_markers)
                    has_voltas = any(rm.get("volta_endings") for rm in repeat_markers)
                    volta_status = "detected" if has_voltas else "none_found"
                except Exception as e:
                    volta_status = f"error: {e}"
                    print(f"[volta] Detection failed (non-fatal): {e}")
                    import traceback
                    traceback.print_exc()

            # Build response
            notes = parsed.get("notes", [])
            rests = parsed.get("rests", [])
            metadata = parsed.get("metadata", {})
            metadata["processing_time"] = round(processing_time, 2)
            metadata["detection_method"] = "homr"
            metadata["volta_detection"] = volta_status

            return {
                "success": True,
                "notes": notes,
                "rests": rests,
                "repeat_markers": repeat_markers,
                "note_count": len(notes),
                "rest_count": len(rests),
                "repeat_count": len(repeat_markers),
                "metadata": metadata,
                "musicxml": musicxml_content,
                "message": f"HOMR processed {len(notes)} notes, {len(rests)} rests, {len(repeat_markers)} repeats in {processing_time:.1f}s",
            }

        finally:
            # Clean up temp files
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
