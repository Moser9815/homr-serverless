"""
Local test script for HOMR OMR handler.

Usage:
    python test_local.py <image_path> [--clef bass|treble] [--tempo 120] [--time-sig 4/4]

Tests the handler locally without RunPod. Useful for validating HOMR output
before deploying to serverless.
"""

import argparse
import base64
import json
import sys
import time


def test_handler(image_path: str, clef: str = "treble", tempo: int = 120, time_sig: str = "4/4"):
    """Test the handler with a local image file."""
    from handler import handler

    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Build mock RunPod event
    event = {
        "input": {
            "image": image_data,
            "clef": clef,
            "tempo": tempo,
            "time_signature": time_sig,
        }
    }

    print(f"Processing: {image_path}")
    print(f"  Clef: {clef}, Tempo: {tempo}, Time sig: {time_sig}")
    print()

    start = time.time()
    result = handler(event)
    elapsed = time.time() - start

    if result.get("success"):
        print(f"SUCCESS in {elapsed:.1f}s")
        print(f"  Notes: {result['note_count']}")
        print(f"  Rests: {result['rest_count']}")
        print(f"  Metadata: {json.dumps(result['metadata'], indent=2)}")
        print()

        # Show first few notes
        for i, note in enumerate(result["notes"][:10]):
            print(f"  Note {i+1}: {note['pitch_name']} "
                  f"(MIDI {note['pitch']}) "
                  f"m{note['measure']}b{note['beat']} "
                  f"{note['duration_type']} "
                  f"t={note['start_time']:.2f}-{note['end_time']:.2f}s")

        if result["note_count"] > 10:
            print(f"  ... and {result['note_count'] - 10} more notes")

        # Save full result
        output_path = image_path.rsplit(".", 1)[0] + "_homr_result.json"
        with open(output_path, "w") as f:
            # Don't save the musicxml in the test output (too large)
            result_copy = {k: v for k, v in result.items() if k != "musicxml"}
            json.dump(result_copy, f, indent=2)
        print(f"\n  Full result saved to: {output_path}")

        # Also save raw MusicXML
        if result.get("musicxml"):
            xml_path = image_path.rsplit(".", 1)[0] + "_homr.musicxml"
            with open(xml_path, "w") as f:
                f.write(result["musicxml"])
            print(f"  MusicXML saved to: {xml_path}")
    else:
        print(f"FAILED in {elapsed:.1f}s")
        print(f"  Error: {result.get('error')}")
        if result.get("traceback"):
            print(f"  Traceback:\n{result['traceback']}")

    return result


def test_parser_only(musicxml_path: str, clef: str = "treble", tempo: int = 120, time_sig: str = "4/4"):
    """Test just the MusicXML parser on an existing file."""
    from parse_musicxml import parse_musicxml_to_json

    with open(musicxml_path, "r") as f:
        content = f.read()

    result = parse_musicxml_to_json(content, clef, tempo, time_sig)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HOMR OMR locally")
    parser.add_argument("image", help="Path to sheet music image (or .musicxml for parser-only test)")
    parser.add_argument("--clef", default="treble", choices=["treble", "bass", "alto"])
    parser.add_argument("--tempo", type=int, default=120)
    parser.add_argument("--time-sig", default="4/4")

    args = parser.parse_args()

    if args.image.endswith((".musicxml", ".xml")):
        test_parser_only(args.image, args.clef, args.tempo, args.time_sig)
    else:
        test_handler(args.image, args.clef, args.tempo, args.time_sig)
