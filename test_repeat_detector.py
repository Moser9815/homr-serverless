"""
Test script for detect_repeats.py

Usage:
    # Test with a saved HOMR result (fast, no GPU needed):
    python test_repeat_detector.py --result saved_result.json --image cancan.jpeg

    # Test by calling the live HOMR endpoint (slower, needs API key):
    python test_repeat_detector.py --image cancan.jpeg --endpoint jeoecq5d89wutl

The first mode uses a previously saved HOMR result JSON (barline_info, staff_info).
The second mode calls RunPod, saves the result, then runs the classifier.
"""

import argparse
import base64
import json
import os
import sys
import time

import requests

from detect_repeats import detect_repeat_barlines, build_repeat_markers

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")


def call_homr_endpoint(image_path: str, endpoint_id: str, clef: str = "bass", tempo: int = 60, time_sig: str = "2/4") -> dict:
    """Call RunPod HOMR endpoint and return the result."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    run_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

    payload = {
        "input": {
            "image": image_b64,
            "clef": clef,
            "tempo": tempo,
            "time_signature": time_sig,
        }
    }

    print(f"Submitting to RunPod endpoint {endpoint_id}...")
    resp = requests.post(run_url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    job = resp.json()
    job_id = job["id"]
    print(f"Job ID: {job_id}")

    # Poll for completion
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    for _ in range(120):  # 10 min max
        time.sleep(5)
        resp = requests.get(status_url, headers=headers, timeout=30)
        data = resp.json()
        status = data.get("status", "UNKNOWN")
        print(f"  Status: {status}")
        if status == "COMPLETED":
            return data.get("output", {})
        elif status in ("FAILED", "CANCELLED"):
            print(f"  Error: {data}")
            sys.exit(1)

    print("Timeout!")
    sys.exit(1)


def test_from_result(image_path: str, result: dict):
    """Run the repeat detector on a saved result."""
    barline_info = result.get("barline_positions", [])
    staff_info = result.get("staff_positions", [])
    total_measures = result.get("metadata", {}).get("total_measures", 0)

    print(f"\nImage: {image_path}")
    print(f"Barlines: {len(barline_info)}, Staves: {len(staff_info)}, Measures: {total_measures}")

    # Show HOMR's repeat markers (from MusicXML)
    homr_markers = result.get("repeat_markers", [])
    print(f"\nHOMR's repeat markers (from MusicXML): {len(homr_markers)}")
    for rm in homr_markers:
        print(f"  m{rm['start_measure']}–m{rm['end_measure']} "
              f"voltas={rm.get('volta_endings')}")

    # Run our classifier
    print("\n--- Running dot-side classifier ---")
    classified = detect_repeat_barlines(
        image_path, barline_info, staff_info,
        total_measures=total_measures, debug=True,
    )

    # Show all classified barlines
    print(f"\nClassified barlines ({len(classified)} total):")
    for bl in classified:
        if bl["type"] != "normal":
            print(f"  [{bl['type'].upper():8s}] staff={bl['staff_idx']} "
                  f"x={bl['x']:.0f} w={bl['width']:.1f} "
                  f"L={bl['left_score']:.3f} R={bl['right_score']:.3f} "
                  f"m_before={bl.get('measure_before')} m_after={bl.get('measure_after')}")

    # Build repeat markers
    markers = build_repeat_markers(classified, debug=True)
    print(f"\nOur repeat markers: {len(markers)}")
    for rm in markers:
        print(f"  m{rm['start_measure']}–m{rm['end_measure']}")

    # Summary
    print(f"\n=== COMPARISON ===")
    print(f"HOMR transformer: {len(homr_markers)} repeat markers")
    print(f"Our detector:     {len(markers)} repeat markers")

    # Show all scores for tuning
    print(f"\n=== ALL BARLINE SCORES (for threshold tuning) ===")
    scores = [(bl["left_score"], bl["right_score"], bl["staff_idx"], bl["x"], bl["width"])
              for bl in classified]
    scores.sort(key=lambda s: max(s[0], s[1]), reverse=True)
    for ls, rs, si, x, w in scores[:20]:
        tag = ""
        if ls > 0.12 or rs > 0.12:
            tag = " <<<" if ls > 0.12 and rs > 0.12 else " <"
        print(f"  staff={si} x={x:6.0f} w={w:4.1f}  L={ls:.3f} R={rs:.3f}{tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test repeat barline detector")
    parser.add_argument("--image", required=True, help="Path to sheet music image")
    parser.add_argument("--result", help="Path to saved HOMR result JSON")
    parser.add_argument("--endpoint", default="jeoecq5d89wutl", help="RunPod endpoint ID")
    parser.add_argument("--clef", default="bass")
    parser.add_argument("--tempo", type=int, default=60)
    parser.add_argument("--time-sig", default="2/4")
    args = parser.parse_args()

    if args.result:
        # Load saved result
        with open(args.result) as f:
            result = json.load(f)
    else:
        # Call live endpoint
        result = call_homr_endpoint(
            args.image, args.endpoint,
            clef=args.clef, tempo=args.tempo, time_sig=args.time_sig,
        )
        # Save for future fast testing
        save_path = os.path.splitext(args.image)[0] + "_homr_result.json"
        with open(save_path, "w") as f:
            result_copy = {k: v for k, v in result.items() if k != "musicxml"}
            json.dump(result_copy, f, indent=2)
        print(f"Result saved to {save_path}")

    test_from_result(args.image, result)
