#!/usr/bin/env python3
"""
Test script: Grand Staff Split for Transformer Inference

Phase 1: Verify we can split a merged 10-line Staff back into two 5-line Staffs.
Phase 2: Run parse_staffs with split staves and compare note counts.
"""

import sys
import os
import time
import numpy as np

# Add the handler directory to path
sys.path.insert(0, os.path.dirname(__file__))

from homr.model import Staff, StaffPoint, MultiStaff, Note, BarLine, Rest, Clef, SymbolOnStaff
from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox


def split_grandstaff(staff: Staff) -> tuple[Staff, Staff]:
    """
    Split a merged 10-line grand Staff back into two 5-line sub-Staffs.

    The merged Staff has grid points with 10 y-values each (lines 0-4 = top,
    lines 5-9 = bottom). We split these into two separate grids.

    Symbols are partitioned by y-coordinate: above the midpoint between
    line 4 and line 5 goes to top staff, below goes to bottom staff.
    """
    if not staff.is_grandstaff:
        raise ValueError("Staff is not a grand staff — cannot split")

    if len(staff.grid) == 0:
        raise ValueError("Staff has empty grid")

    num_lines = len(staff.grid[0].y)
    if num_lines != 10:
        raise ValueError(f"Expected 10 grid lines for grand staff, got {num_lines}")

    # Build two grids: top = lines 0-4, bottom = lines 5-9
    top_grid = []
    bot_grid = []

    for point in staff.grid:
        top_ys = point.y[:5]
        bot_ys = point.y[5:]
        top_grid.append(StaffPoint(point.x, top_ys, point.angle))
        bot_grid.append(StaffPoint(point.x, bot_ys, point.angle))

    top_staff = Staff(top_grid)
    bot_staff = Staff(bot_grid)

    # Partition symbols by y-coordinate
    # Use the gap between line 4 (bottom of top staff) and line 5 (top of bottom staff)
    # at each x position to determine which staff a symbol belongs to.
    for symbol in staff.symbols:
        sx = symbol.center[0]
        sy = symbol.center[1]

        # Find the closest grid point to determine the split boundary
        closest = min(staff.grid, key=lambda p: abs(p.x - sx))
        mid_y = (closest.y[4] + closest.y[5]) / 2.0

        if sy < mid_y:
            top_staff.add_symbol(symbol)
        else:
            bot_staff.add_symbol(symbol)

    return top_staff, bot_staff


def split_multi_staffs_for_transformer(multi_staffs: list[MultiStaff]) -> list[MultiStaff]:
    """
    Process a list of MultiStaffs, splitting any merged grand staves
    back into two separate staves within their MultiStaff.

    After splitting:
    - Each MultiStaff that WAS a grand staff now has 2 staves (top, bottom)
    - Each MultiStaff that was a single staff still has 1 staff
    - _get_number_of_voices() will return 2 for grand staff pieces
    """
    result = []
    for ms in multi_staffs:
        if len(ms.staffs) == 1 and ms.staffs[0].is_grandstaff:
            merged = ms.staffs[0]
            top, bot = split_grandstaff(merged)
            # Create a new MultiStaff with both sub-staves
            result.append(MultiStaff([top, bot], ms.connections))
        else:
            result.append(ms)
    return result


def _create_tight_regions_for_split(multi_staffs, split_ms):
    """
    Create StaffRegions with phantom boundary staves at the midpoint between
    split grand staff sub-staves. This prevents the crop region from extending
    into the neighboring sub-staff.

    Without this, _calculate_region's 4*unit padding extends 30+ pixels past
    the gap midpoint, causing the transformer to see notes from both staves.
    """
    from homr.model import MultiStaff, StaffPoint

    # Build regions with extra phantom staves at gap midpoints
    # A phantom staff is placed at the midpoint between top and bottom sub-staves
    # It acts as a boundary to limit _calculate_region's padding
    phantom_ms = list(split_ms)
    phantoms_added = 0

    for ms in split_ms:
        if len(ms.staffs) == 2:
            top = ms.staffs[0]
            bot = ms.staffs[1]
            gap_midpoint = (top.max_y + bot.min_y) / 2.0

            # Create a thin phantom staff at the midpoint
            # StaffRegions only looks at min_y and max_y, so we just need
            # a Staff with those values near the midpoint
            phantom_grid = [
                StaffPoint(top.grid[0].x,
                          [gap_midpoint - 2, gap_midpoint - 1, gap_midpoint,
                           gap_midpoint + 1, gap_midpoint + 2],
                          0.0),
                StaffPoint(top.grid[-1].x,
                          [gap_midpoint - 2, gap_midpoint - 1, gap_midpoint,
                           gap_midpoint + 1, gap_midpoint + 2],
                          0.0),
            ]
            from homr.model import Staff as StaffClass
            phantom = StaffClass(phantom_grid)
            phantom_ms.append(MultiStaff([phantom], []))
            phantoms_added += 1

    if phantoms_added > 0:
        from homr.simple_logging import eprint
        eprint(f"[SPLIT] Added {phantoms_added} phantom boundary staves for tight crop regions")

    from homr.staff_regions import StaffRegions
    return StaffRegions(phantom_ms)


def parse_staffs_with_grandstaff_split(debug, multi_staffs, image, config, selected_staff=-1):
    """
    Custom parse_staffs that splits grand staves before transformer inference,
    then recombines the results into a single voice with upper/lower positions.

    This gives each 5-line sub-staff full 256px resolution in the transformer
    instead of cramming both into 256px (half resolution each).

    For non-grand-staff pieces, falls through to the standard parse_staffs.
    """
    import copy
    from homr.staff_parsing import (
        _ensure_same_number_of_staffs,
        parse_staff_image,
    )
    from homr.staff_regions import StaffRegions
    from homr.transformer.vocabulary import EncodedSymbol, remove_duplicated_symbols
    from homr.simple_logging import eprint

    # Check if any multi_staff has a grand staff
    has_grandstaff = any(
        len(ms.staffs) == 1 and ms.staffs[0].is_grandstaff
        for ms in multi_staffs
    )

    if not has_grandstaff:
        # No grand staves — use standard parse_staffs
        from homr.staff_parsing import parse_staffs
        eprint("[SPLIT] No grand staves detected — using standard parse_staffs")
        return parse_staffs(debug, multi_staffs, image, config=config, selected_staff=selected_staff)

    eprint("[SPLIT] Grand staff detected — splitting for full-resolution transformer inference")

    # Build the split staves
    split_ms = split_multi_staffs_for_transformer(multi_staffs)

    # Create tight regions with phantom boundary staves to prevent
    # crop overlap between split sub-staves
    regions = _create_tight_regions_for_split(multi_staffs, split_ms)

    # Process each system: split grand staff, run transformer on each sub-staff
    i = 0
    voices_treble = []
    voices_bass = []

    for sys_idx, ms in enumerate(multi_staffs):
        if len(ms.staffs) != 1 or not ms.staffs[0].is_grandstaff:
            # Not a grand staff — process normally
            staff = ms.staffs[0]
            if selected_staff >= 0 and sys_idx != selected_staff:
                eprint(f"[SPLIT] Ignoring staff {i} (selected_staff={selected_staff})")
                i += 1
                continue
            result = parse_staff_image(debug, i, staff, image, regions, config)
            if len(result) > 0:
                result.append(EncodedSymbol("newline"))
                voices_treble.extend(result)
            else:
                eprint(f"[SPLIT] Skipping empty staff {i}")
            i += 1
            continue

        # Grand staff — split and process each sub-staff independently
        merged = ms.staffs[0]
        top_staff, bot_staff = split_grandstaff(merged)

        eprint(f"[SPLIT] System {sys_idx}: splitting grand staff "
               f"(top: {top_staff.get_number_of_notes()} notes, "
               f"bot: {bot_staff.get_number_of_notes()} notes)")

        # Run transformer on top sub-staff (treble)
        top_result = parse_staff_image(debug, i, top_staff, image, regions, config)
        i += 1
        if len(top_result) > 0:
            top_result.append(EncodedSymbol("newline"))
            voices_treble.extend(top_result)

        # Run transformer on bottom sub-staff (bass)
        bot_result = parse_staff_image(debug, i, bot_staff, image, regions, config)
        i += 1
        if len(bot_result) > 0:
            bot_result.append(EncodedSymbol("newline"))
            voices_bass.extend(bot_result)

        eprint(f"[SPLIT] System {sys_idx}: top={len(top_result)} symbols, "
               f"bot={len(bot_result)} symbols")

    # Return 2 voices: treble (voice 0) and bass (voice 1)
    # This produces a 2-part MusicXML which the parser now handles
    result = [
        remove_duplicated_symbols(voices_treble),
        remove_duplicated_symbols(voices_bass),
    ]

    treble_notes = sum(1 for s in result[0] if s.rhythm.startswith("note_"))
    bass_notes = sum(1 for s in result[1] if s.rhythm.startswith("note_"))
    eprint(f"[SPLIT] Result: {treble_notes} treble notes + {bass_notes} bass notes = {treble_notes + bass_notes} total")

    return result


def test_split_basic():
    """Test that split produces valid Staff objects with correct structure."""
    # Create a simple merged grand staff
    grid = []
    for x in range(0, 100, 10):
        # Top staff lines: y = 100, 110, 120, 130, 140
        # Bottom staff lines: y = 200, 210, 220, 230, 240
        ys = [100, 110, 120, 130, 140, 200, 210, 220, 230, 240]
        grid.append(StaffPoint(float(x), ys, 0.0))

    staff = Staff(grid)
    staff.is_grandstaff = True

    # Add a note in the top staff area (y=125)
    # BoundingEllipse expects ((cx, cy), (w, h), angle) RotatedRect format
    top_note = Note(
        BoundingEllipse(((50, 125), (10, 8), 0.0), np.array([]), 0),
        position=3,
        stem=None,
        stem_direction=None,
    )
    staff.add_symbol(top_note)

    # Add a note in the bottom staff area (y=215)
    bot_note = Note(
        BoundingEllipse(((50, 215), (10, 8), 0.0), np.array([]), 0),
        position=-2,
        stem=None,
        stem_direction=None,
    )
    staff.add_symbol(bot_note)

    # Split
    top, bot = split_grandstaff(staff)

    # Verify grid structure
    assert len(top.grid[0].y) == 5, f"Top grid should have 5 lines, got {len(top.grid[0].y)}"
    assert len(bot.grid[0].y) == 5, f"Bot grid should have 5 lines, got {len(bot.grid[0].y)}"
    assert top.grid[0].y == [100, 110, 120, 130, 140], f"Top y-values wrong: {top.grid[0].y}"
    assert bot.grid[0].y == [200, 210, 220, 230, 240], f"Bot y-values wrong: {bot.grid[0].y}"

    # Verify symbols were partitioned correctly
    assert len(top.get_notes()) == 1, f"Top should have 1 note, got {len(top.get_notes())}"
    assert len(bot.get_notes()) == 1, f"Bot should have 1 note, got {len(bot.get_notes())}"
    assert top.get_notes()[0].center[1] == 125, "Top note should be at y=125"
    assert bot.get_notes()[0].center[1] == 215, "Bot note should be at y=215"

    # Verify is_grandstaff is NOT set on sub-staves
    assert not top.is_grandstaff, "Top sub-staff should not be grandstaff"
    assert not bot.is_grandstaff, "Bot sub-staff should not be grandstaff"

    # Verify min_y / max_y
    assert top.min_y == 100, f"Top min_y should be 100, got {top.min_y}"
    assert top.max_y == 140, f"Top max_y should be 140, got {top.max_y}"
    assert bot.min_y == 200, f"Bot min_y should be 200, got {bot.min_y}"
    assert bot.max_y == 240, f"Bot max_y should be 240, got {bot.max_y}"

    print("PASS: test_split_basic")


def test_multi_staff_split():
    """Test split_multi_staffs_for_transformer preserves non-grand-staves."""
    # Create a grand staff MultiStaff
    grid = []
    for x in range(0, 100, 10):
        ys = [100, 110, 120, 130, 140, 200, 210, 220, 230, 240]
        grid.append(StaffPoint(float(x), ys, 0.0))

    merged_staff = Staff(grid)
    merged_staff.is_grandstaff = True
    grand_ms = MultiStaff([merged_staff], [])

    # Create a single-staff MultiStaff
    single_grid = []
    for x in range(0, 100, 10):
        ys = [300, 310, 320, 330, 340]
        single_grid.append(StaffPoint(float(x), ys, 0.0))
    single_staff = Staff(single_grid)
    single_ms = MultiStaff([single_staff], [])

    # Split
    result = split_multi_staffs_for_transformer([grand_ms, single_ms])

    assert len(result) == 2, f"Should still have 2 MultiStaffs, got {len(result)}"
    assert len(result[0].staffs) == 2, f"Grand staff should split to 2, got {len(result[0].staffs)}"
    assert len(result[1].staffs) == 1, f"Single staff should stay 1, got {len(result[1].staffs)}"

    print("PASS: test_multi_staff_split")


def test_on_real_image():
    """Test on Drift Away image — the real test."""
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
    from homr.transformer.configs import Config as TransformerConfig
    from homr.debug import Debug

    image_path = "/Users/moserrs/Studio/04_Apps/03_Personal/01_Music App/services/ocr/data/drift_away_clean.png"

    if not os.path.exists(image_path):
        print(f"SKIP: {image_path} not found")
        return

    print(f"\n{'='*60}")
    print(f"Testing on: drift_away_clean.png")
    print(f"{'='*60}")

    config = ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        use_gpu_inference=False,
    )

    # Run detection pipeline
    print("\nStep 1: Running detection pipeline...")
    t0 = time.time()
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

    # Add notes and barlines to staves
    from homr.model import BarLine as BarLineModel
    notes = add_notes_to_staffs(
        staffs, noteheads_with_stems, predictions.symbols, predictions.notehead
    )
    for blbox in bar_line_boxes:
        for staff in staffs:
            if staff.is_on_staff_zone(blbox):
                staff.add_symbol(BarLineModel(blbox))
                break

    # Grand staff detection (merge)
    brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
    brace_dot = create_rotated_bounding_boxes(brace_dot_img, skip_merging=True, max_size=(100, -1))
    multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)

    t1 = time.time()
    print(f"Detection complete in {t1-t0:.1f}s")
    print(f"Found {len(multi_staffs)} multi-staffs, staves per system: {[len(ms.staffs) for ms in multi_staffs]}")

    # Check if any are grand staves
    for i, ms in enumerate(multi_staffs):
        for j, s in enumerate(ms.staffs):
            grid_lines = len(s.grid[0].y) if s.grid else 0
            print(f"  System {i}, staff {j}: {grid_lines} grid lines, "
                  f"is_grandstaff={s.is_grandstaff}, "
                  f"notes={s.get_number_of_notes()}, "
                  f"barlines={len(s.get_bar_lines())}")

    # === Run A: Merged (baseline) ===
    print(f"\n--- Run A: MERGED (baseline) ---")
    transformer_config = TransformerConfig()
    transformer_config.use_gpu_inference = False

    t2 = time.time()
    result_merged = parse_staffs(
        debug, multi_staffs, predictions.preprocessed,
        selected_staff=-1, config=transformer_config,
    )
    t3 = time.time()

    total_merged = sum(len(v) for v in result_merged)
    note_count_merged = sum(
        1 for v in result_merged for s in v
        if s.rhythm.startswith("note_")
    )
    print(f"Merged: {len(result_merged)} voices, {total_merged} total symbols, "
          f"{note_count_merged} notes, {t3-t2:.1f}s")

    # === Run B: Split (recombined into single voice) ===
    print(f"\n--- Run B: SPLIT (recombined) ---")
    t4 = time.time()
    result_split = parse_staffs_with_grandstaff_split(
        debug, multi_staffs, predictions.preprocessed,
        config=transformer_config, selected_staff=-1,
    )
    t5 = time.time()

    total_split = sum(len(v) for v in result_split)
    note_count_split = sum(
        1 for v in result_split for s in v
        if s.rhythm.startswith("note_")
    )
    print(f"Split: {len(result_split)} voices, {total_split} total symbols, "
          f"{note_count_split} notes, {t5-t4:.1f}s")

    # === Print comparison ===
    print(f"\n{'='*60}")
    print(f"COMPARISON:")
    print(f"  Merged: {len(result_merged)} voices, {note_count_merged} notes")
    print(f"  Split:  {len(result_split)} voices, {note_count_split} notes")
    print(f"{'='*60}")

    # Print the actual symbols for each voice to compare
    for voice_idx, voice in enumerate(result_merged):
        notes_in_voice = [s for s in voice if s.rhythm.startswith("note_")]
        print(f"\nMERGED voice {voice_idx}: {len(notes_in_voice)} notes")
        for s in voice[:30]:
            print(f"  {s}")

    for voice_idx, voice in enumerate(result_split):
        notes_in_voice = [s for s in voice if s.rhythm.startswith("note_")]
        print(f"\nSPLIT voice {voice_idx}: {len(notes_in_voice)} notes")
        for s in voice[:30]:
            print(f"  {s}")


if __name__ == "__main__":
    # Unit tests first
    test_split_basic()
    test_multi_staff_split()

    # Real image test
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        test_on_real_image()
    else:
        print("\nRun with --real to test on actual Drift Away image")
