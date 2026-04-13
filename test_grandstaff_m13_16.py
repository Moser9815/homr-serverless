#!/usr/bin/env python3
"""
Test the grand staff split against ground truth for measures 13-16.

Runs both merged and split, generates MusicXML, parses notes, and
compares against the known ground truth for Drift Away m13-16.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from test_grandstaff_split import split_multi_staffs_for_transformer, parse_staffs_with_grandstaff_split


def run_pipeline_and_get_musicxml(image_path, split_grandstaves=False):
    """Run HOMR pipeline on image and return MusicXML content and result staffs."""
    from homr.main import (
        load_and_preprocess_predictions,
        predict_symbols,
        break_wide_fragments,
        combine_noteheads_with_stems,
        detect_bar_lines,
        detect_staff,
        detect_title,
        add_notes_to_staffs,
        prepare_brace_dot_image,
        create_rotated_bounding_boxes,
        find_braces_brackets_and_grand_staff_lines,
        parse_staffs,
        ProcessingConfig,
    )
    from homr.model import BarLine as BarLineModel
    from homr.transformer.configs import Config as TransformerConfig
    from homr.music_xml_generator import generate_xml, XmlGeneratorArguments

    config = ProcessingConfig(
        enable_debug=False, enable_cache=False,
        write_staff_positions=False, read_staff_positions=False,
        selected_staff=-1, use_gpu_inference=False,
    )

    predictions, debug = load_and_preprocess_predictions(
        image_path, config.enable_debug, config.enable_cache, config.use_gpu_inference
    )
    symbols = predict_symbols(debug, predictions)
    symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)

    noteheads_with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
    average_note_head_height = float(
        np.median([nh.notehead.size[1] for nh in noteheads_with_stems])
    )

    all_noteheads = [nh.notehead for nh in noteheads_with_stems]
    all_stems = [n.stem for n in noteheads_with_stems if n.stem is not None]
    bar_lines_or_rests = [
        line for line in symbols.bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]
    bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)

    staffs = detect_staff(
        debug, predictions.staff, symbols.staff_fragments,
        symbols.clefs_keys, bar_line_boxes
    )

    from homr.model import BarLine as BLM
    notes = add_notes_to_staffs(
        staffs, noteheads_with_stems, predictions.symbols, predictions.notehead
    )
    for blbox in bar_line_boxes:
        for staff in staffs:
            if staff.is_on_staff_zone(blbox):
                staff.add_symbol(BLM(blbox))
                break

    brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
    brace_dot = create_rotated_bounding_boxes(brace_dot_img, skip_merging=True, max_size=(100, -1))
    multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)

    transformer_config = TransformerConfig()
    transformer_config.use_gpu_inference = False

    t0 = time.time()
    if split_grandstaves:
        result_staffs = parse_staffs_with_grandstaff_split(
            debug, multi_staffs, predictions.preprocessed,
            config=transformer_config, selected_staff=-1,
        )
        mode = "SPLIT"
    else:
        result_staffs = parse_staffs(
            debug, multi_staffs, predictions.preprocessed,
            selected_staff=-1, config=transformer_config,
        )
        mode = "MERGED"
    t1 = time.time()

    title_future = detect_title(debug, staffs[0])
    title = ""
    try:
        title = title_future.result(10)
    except Exception:
        pass

    xml = generate_xml(XmlGeneratorArguments(), result_staffs, title)

    # Write to temp file and read back
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False, mode='w') as f:
        tmp_path = f.name

    xml.write(tmp_path)
    with open(tmp_path, 'r') as f:
        musicxml_content = f.read()
    os.unlink(tmp_path)

    print(f"[{mode}] Transformer: {t1-t0:.1f}s, "
          f"{len(result_staffs)} voices, "
          f"{sum(1 for v in result_staffs for s in v if s.rhythm.startswith('note_'))} notes")

    return musicxml_content


def parse_and_extract_m13_16(musicxml_content):
    """Parse MusicXML and extract notes from measures 13-16."""
    from parse_musicxml import parse_musicxml_to_json

    parsed = parse_musicxml_to_json(musicxml_content)
    notes = parsed.get("notes", [])
    rests = parsed.get("rests", [])

    # Filter to measures 13-16
    m13_16_notes = [n for n in notes if 13 <= n.get("measure", 0) <= 16]
    m13_16_rests = [r for r in rests if 13 <= r.get("measure", 0) <= 16]

    return m13_16_notes, m13_16_rests, parsed


# Ground truth for m13-16 (MIDI note numbers)
# From the task description
GROUND_TRUTH = {
    # (measure, staff, beat_approx): (pitch_name, midi)
    # Bar 13 top
    (13, 1, "rest_1"): ("1/8 rest", None),
    (13, 1, "G3_1"): ("G3", 55),
    (13, 1, "A3_16"): ("A3 16th", 57),
    (13, 1, "B3_16"): ("B3 16th", 59),
    (13, 1, "G3_8"): ("G3 8th", 55),
    (13, 1, "G3_q"): ("G3 quarter", 55),
    # Bar 13 bottom
    (13, 2, "G2"): ("G2", 43),
    (13, 2, "D3"): ("D3", 50),
    # Bar 14 top
    (14, 1, "G3_1"): ("G3", 55),
    (14, 1, "A3_1"): ("A3", 57),
    (14, 1, "B3_1"): ("B3", 59),
    (14, 1, "A3_dq"): ("dotted-quarter A3", 57),
    # Bar 14 bottom
    (14, 2, "F2"): ("F2", 41),
    (14, 2, "D3"): ("D3", 50),
    # Bar 15 top
    (15, 1, "rest_1"): ("rest", None),
    (15, 1, "G3_1"): ("G3", 55),
    (15, 1, "G3_t"): ("G3 triplet", 55),
    (15, 1, "A3_t"): ("A3 triplet", 57),
    (15, 1, "B3_t"): ("B3 triplet", 59),
    (15, 1, "A3_q"): ("quarter A3", 57),
    (15, 1, "G3_8"): ("eighth G3", 55),
    # Bar 15 bottom
    (15, 2, "E2"): ("E2", 40),
    (15, 2, "C3"): ("C3", 48),
    # Bar 16 top
    (16, 1, "Eb2"): ("half Eb2", 39),
    # Bar 16 bottom
    (16, 2, "Eb2_1"): ("Eb2", 39),
    (16, 2, "C3_1"): ("C3", 48),
    (16, 2, "Eb2_2"): ("Eb2", 39),
    (16, 2, "C3_2"): ("C3", 48),
    (16, 2, "Eb2_3"): ("Eb2", 39),
    (16, 2, "Cb_1"): ("Cb/B2", 47),  # "Cb" from task description
}

# Simplified: just count correct MIDI pitches by measure and staff
GT_NOTES_FLAT = [
    # (measure, staff, midi, description)
    # m13 top
    (13, 1, 55, "G3"),
    (13, 1, 57, "A3 16th"),
    (13, 1, 59, "B3 16th"),
    (13, 1, 55, "G3 8th"),
    (13, 1, 55, "G3 quarter"),
    # m13 bottom
    (13, 2, 43, "G2"),
    (13, 2, 50, "D3"),
    # m14 top
    (14, 1, 55, "G3"),
    (14, 1, 57, "A3"),
    (14, 1, 59, "B3"),
    (14, 1, 57, "A3 dotted-quarter"),
    # m14 bottom
    (14, 2, 41, "F2"),
    (14, 2, 50, "D3"),
    # m15 top
    (15, 1, 55, "G3"),
    (15, 1, 55, "G3 triplet"),
    (15, 1, 57, "A3 triplet"),
    (15, 1, 59, "B3 triplet"),
    (15, 1, 57, "A3 quarter"),
    (15, 1, 55, "G3 eighth"),
    # m15 bottom
    (15, 2, 40, "E2"),
    (15, 2, 48, "C3"),
    # m16 top
    (16, 1, 39, "Eb2"),
    # m16 bottom - dotted half chord + eighth chords
    (16, 2, 39, "Eb2"),
    (16, 2, 48, "C3"),
    (16, 2, 39, "Eb2"),
    (16, 2, 48, "C3"),
]

# Total: 27 notes


def note_name_to_midi(step, octave, alter=0):
    """Convert note name components to MIDI number."""
    step_to_semitone = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    return 12 * (octave + 1) + step_to_semitone[step] + alter


def compare_with_gt(notes, rests, label):
    """Compare parsed notes against ground truth for m13-16."""
    print(f"\n{'='*60}")
    print(f"  {label}: m13-16 comparison")
    print(f"{'='*60}")

    # Sort notes by measure, staff, beat
    sorted_notes = sorted(notes, key=lambda n: (
        n.get("measure", 0),
        n.get("staff", 1),
        n.get("beat", 0),
    ))

    # Print all notes in m13-16
    for m in [13, 14, 15, 16]:
        m_notes = [n for n in sorted_notes if n.get("measure") == m]
        m_rests = [r for r in rests if r.get("measure") == m]
        print(f"\n  Measure {m}:")
        for n in m_notes:
            staff = n.get("staff", 1)
            pitch = n.get("pitch", "?")
            midi = n.get("midi", "?")
            beat = n.get("beat", "?")
            dur = n.get("duration_type", "?")
            dur_beats = n.get("duration_beats", "?")
            dot = " (dotted)" if n.get("dotted") else ""
            triplet = " [triplet]" if n.get("is_tuplet") else ""
            print(f"    S{staff} beat {beat}: {pitch} (MIDI {midi}) {dur}{dot}{triplet} dur={dur_beats}")
        for r in m_rests:
            staff = r.get("staff", 1)
            beat = r.get("beat", "?")
            dur = r.get("duration_type", "?")
            print(f"    S{staff} beat {beat}: REST {dur}")

    # Count correct MIDI pitches
    # Group parsed notes by (measure, staff)
    parsed_by_ms = {}
    for n in sorted_notes:
        key = (n.get("measure"), n.get("staff", 1))
        parsed_by_ms.setdefault(key, []).append(n)

    # Group GT by (measure, staff)
    gt_by_ms = {}
    for m, s, midi, desc in GT_NOTES_FLAT:
        gt_by_ms.setdefault((m, s), []).append((midi, desc))

    correct = 0
    wrong = 0
    missing = 0
    extra = 0

    for key in sorted(set(list(gt_by_ms.keys()) + list(parsed_by_ms.keys()))):
        gt_list = gt_by_ms.get(key, [])
        parsed_list = parsed_by_ms.get(key, [])

        gt_midis = [m for m, d in gt_list]
        parsed_midis = [n.get("midi") for n in parsed_list]

        # Simple greedy matching
        gt_remaining = list(gt_midis)
        for pm in parsed_midis:
            if pm in gt_remaining:
                gt_remaining.remove(pm)
                correct += 1
            else:
                wrong += 1
        missing += len(gt_remaining)

    total_gt = len(GT_NOTES_FLAT)
    print(f"\n  SCORE: {correct}/{total_gt} correct, {wrong} wrong, {missing} missing")
    return correct, total_gt


def main():
    image_path = "/Users/moserrs/Studio/04_Apps/03_Personal/01_Music App/services/ocr/data/drift_away_clean.png"

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # Run merged
    print("\n" + "="*60)
    print("  Running MERGED (baseline)...")
    print("="*60)
    merged_xml = run_pipeline_and_get_musicxml(image_path, split_grandstaves=False)

    # Run split
    print("\n" + "="*60)
    print("  Running SPLIT...")
    print("="*60)
    split_xml = run_pipeline_and_get_musicxml(image_path, split_grandstaves=True)

    # Parse and compare
    merged_notes, merged_rests, merged_parsed = parse_and_extract_m13_16(merged_xml)
    split_notes, split_rests, split_parsed = parse_and_extract_m13_16(split_xml)

    merged_score = compare_with_gt(merged_notes, merged_rests, "MERGED")
    split_score = compare_with_gt(split_notes, split_rests, "SPLIT")

    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"  Merged: {merged_score[0]}/{merged_score[1]}")
    print(f"  Split:  {split_score[0]}/{split_score[1]}")
    print(f"{'='*60}")

    # Also save both XMLs for inspection
    out_dir = "/Users/moserrs/Studio/04_Apps/03_Personal/04_HOMR-OMR"
    with open(os.path.join(out_dir, "drift_away_merged.musicxml"), 'w') as f:
        f.write(merged_xml)
    with open(os.path.join(out_dir, "drift_away_split.musicxml"), 'w') as f:
        f.write(split_xml)
    print(f"\nSaved MusicXML files to {out_dir}/drift_away_*.musicxml")


if __name__ == "__main__":
    main()
