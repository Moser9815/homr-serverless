"""
HOMR OMR — RunPod Serverless Handler

Calls HOMR's Python API directly to access internal staff, barline,
and note pixel positions from the segmentation pipeline.

This code is licensed under AGPL-3.0 to comply with HOMR's license.
"""

HANDLER_VERSION = "3.0-geometric-pitch"

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
from detect_repeats import detect_repeat_barlines, build_repeat_markers


def decode_image(base64_data: str) -> Image.Image:
    """Decode base64 image data to PIL Image, handling EXIF orientation."""
    image_bytes = base64.b64decode(base64_data)
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def run_homr_api(image_path: str, use_gpu: bool = True) -> tuple:
    """
    Run HOMR via Python API, capturing barline and note pixel positions
    that the standard pipeline discards.

    Returns:
    - musicxml_path: path to the output MusicXML file
    - staff_info: list of staff dicts with pixel positions
    - barline_info: list of barline dicts with pixel positions
    - note_info: list of note dicts with pixel positions
    """
    # Import HOMR internals
    from homr.main import (
        ProcessingConfig,
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
        prepare_bar_line_image,
        parse_staffs,
    )
    from homr.model import BarLine
    from homr.transformer.configs import Config as TransformerConfig
    from homr.simple_logging import eprint
    from homr.music_xml_generator import generate_xml, XmlGeneratorArguments
    import numpy as np

    print(f"[HOMR] Processing {image_path} via Python API (with barline capture)")
    start = time.time()

    config = ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        use_gpu_inference=use_gpu,
    )

    # === Run the pipeline manually to capture bar_line_boxes ===

    # Step 1: Load and preprocess
    predictions, debug = load_and_preprocess_predictions(
        image_path, config.enable_debug, config.enable_cache, config.use_gpu_inference
    )
    symbols = predict_symbols(debug, predictions)
    symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)

    # Step 2: Noteheads
    noteheads_with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
    if len(noteheads_with_stems) == 0:
        raise Exception("No noteheads found")

    average_note_head_height = float(
        np.median([nh.notehead.size[1] for nh in noteheads_with_stems])
    )

    # Step 3: Barlines — CAPTURE THESE (normally discarded)
    all_noteheads = [nh.notehead for nh in noteheads_with_stems]
    all_stems = [n.stem for n in noteheads_with_stems if n.stem is not None]
    bar_lines_or_rests = [
        line for line in symbols.bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]
    bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)
    print(f"[HOMR] Captured {len(bar_line_boxes)} barline boxes from segmentation")

    # Step 4: Staff detection
    staffs = detect_staff(
        debug, predictions.staff, symbols.staff_fragments,
        symbols.clefs_keys, bar_line_boxes
    )
    if len(staffs) == 0:
        raise Exception("No staffs found")

    title_future = detect_title(debug, staffs[0])

    # Step 5: Add notes to staves (gives notes pixel positions)
    notes = add_notes_to_staffs(
        staffs, noteheads_with_stems, predictions.symbols, predictions.notehead
    )

    # Step 6: ADD BARLINES TO STAVES (the missing step!)
    barlines_assigned = 0
    for blbox in bar_line_boxes:
        for staff in staffs:
            if staff.is_on_staff_zone(blbox):
                staff.add_symbol(BarLine(blbox))
                barlines_assigned += 1
                break
    print(f"[HOMR] Assigned {barlines_assigned} barlines to staves")

    # Step 7: Grand staff detection (merged — for structure & note positions)
    brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
    brace_dot = create_rotated_bounding_boxes(brace_dot_img, skip_merging=True, max_size=(100, -1))
    multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)

    elapsed_detect = time.time() - start
    print(f"[HOMR] Staff detection: {elapsed_detect:.1f}s, "
          f"{len(staffs)} staves, {barlines_assigned} barlines, {len(notes)} notes")

    # Step 7b: Dewarp staff 0 for clef classification (before transformer)
    dewarped_staff0 = None
    dewarp_error = None
    try:
        from homr.staff_parsing import prepare_staff_image
        from homr.staff_regions import StaffRegions
        regions = StaffRegions(multi_staffs)
        first_staff = multi_staffs[0].staffs[0]
        dewarped_staff0, _ = prepare_staff_image(debug, 0, first_staff, predictions.preprocessed, regions)
        print(f"[HOMR] Dewarped staff 0: {dewarped_staff0.shape}")
    except Exception as e:
        dewarp_error = str(e)
        print(f"[HOMR] Dewarp failed: {e}")
        traceback.print_exc()

    # Step 8: Transformer (symbol recognition → MusicXML)
    # For grand staves, HOMR normally merges two physical staves into one
    # and feeds the combined image to the transformer. This causes pitch
    # errors on the bottom staff because the transformer processes both
    # staves in a single 256px canvas and loses precision on the lower half.
    #
    # Fix: create UNMERGED MultiStaff objects that pair the two physical
    # staves without merging them. parse_staffs then processes each
    # physical staff independently through the transformer, with its own
    # dewarp. The result is more accurate pitches because each transformer
    # run sees a single 5-line staff — the same format HOMR handles well
    # for single-staff pieces.
    transformer_config = TransformerConfig()
    transformer_config.use_gpu_inference = use_gpu

    # Check if we have grand staves (merged). If so, create unmerged
    # versions for the transformer while keeping the merged multi_staffs
    # for note position extraction later.
    has_grand = any(
        len(ms.staffs) == 1 and getattr(ms.staffs[0], 'is_grandstaff', False)
        for ms in multi_staffs
    )

    if has_grand and len(staffs) >= 2:
        # Build unmerged MultiStaffs: pair consecutive individual staves
        # (from step 4, before merge). Staves are y-sorted, so pairs are
        # (0,1), (2,3), etc. — matching the grand-staff grouping.
        unmerged_multi = []
        i = 0
        for ms in multi_staffs:
            if len(ms.staffs) == 1 and getattr(ms.staffs[0], 'is_grandstaff', False):
                # This was a merged grand staff — use 2 individual staves
                if i + 1 < len(staffs):
                    unmerged_multi.append(MultiStaff([staffs[i], staffs[i + 1]], ms.connections))
                    i += 2
                else:
                    unmerged_multi.append(MultiStaff([staffs[i]], []))
                    i += 1
            else:
                # Single staff — keep as-is
                n_staves_in_ms = len(ms.staffs)
                unmerged_multi.append(ms)
                i += n_staves_in_ms

        print(f"[HOMR] Split grand staves: {len(multi_staffs)} merged → "
              f"{len(unmerged_multi)} unmerged ({sum(len(m.staffs) for m in unmerged_multi)} voices)")

        result_staffs = parse_staffs(
            debug, unmerged_multi, predictions.preprocessed,
            selected_staff=-1, config=transformer_config,
        )
    else:
        result_staffs = parse_staffs(
            debug, multi_staffs, predictions.preprocessed,
            selected_staff=-1, config=transformer_config,
        )

    title = ""
    try:
        title = title_future.result(30)
    except Exception:
        pass

    # Step 9: Generate MusicXML
    xml_generator_args = XmlGeneratorArguments()
    xml = generate_xml(xml_generator_args, result_staffs, title)

    musicxml_path = os.path.splitext(image_path)[0] + ".musicxml"
    xml.write(musicxml_path)

    elapsed_total = time.time() - start
    print(f"[HOMR] Total: {elapsed_total:.1f}s")

    # Step 10: Extract pixel data, scaled to original image coordinates
    import cv2 as _cv2
    original = _cv2.imread(image_path)
    orig_h, orig_w = original.shape[:2]
    proc_h, proc_w = predictions.preprocessed.shape[:2]
    scale_x = orig_w / proc_w
    scale_y = orig_h / proc_h

    from homr.model import Rest

    staff_info = []
    barline_info = []
    note_info = []
    rest_info = []

    # homr_buckets keys each parsed note's HOMR Note by (staff_number,
    # measure_number) where staff_number = 1 (top of grand staff) or 2
    # (bottom) to match MusicXML <staff>, and measure_number is the global
    # measure index across all systems. Each bucket's list is in x-order.
    # Used later for geometric pitch resolution with confidence.
    homr_buckets: dict = {}
    running_measure = 1

    for ms_idx, multi_staff in enumerate(multi_staffs):
        # Use the top staff's barlines for measure bucketing (grand-staff
        # barlines span both physical staves).
        top_staff_for_barlines = multi_staff.staffs[0]
        system_barline_xs = sorted(
            b.box.center[0] for b in top_staff_for_barlines.get_bar_lines()
        )
        num_measures_in_system = max(len(system_barline_xs), 1)

        for s_idx, staff in enumerate(multi_staff.staffs):
            staff_idx = len(staff_info)
            # For a merged grand staff (`is_grandstaff == True`), the Staff's
            # grid has 10 y-values per column: lines 0-4 = top physical
            # staff, 5-9 = bottom. Notes from BOTH physical staves live in
            # the merged staff's symbols. For a non-merged (single) staff,
            # `s_idx + 1` is the staff number directly.
            is_grand = getattr(staff, "is_grandstaff", False)

            # Bucket this staff's notes into global measures using the
            # system's barline x-positions AND classify each note as
            # belonging to the top or bottom physical staff by comparing
            # its y to the grid's line positions.
            for note in sorted(staff.get_notes(), key=lambda n: n.center[0]):
                nx_proc = note.center[0]
                ny_proc = note.center[1]

                # Determine staff_number_1or2 for this note.
                if is_grand:
                    grid_point = staff.get_at(nx_proc)
                    if grid_point is None and staff.grid:
                        grid_point = min(staff.grid, key=lambda p: abs(p.x - nx_proc))
                    if grid_point is not None and len(grid_point.y) >= 10:
                        # Midpoint between bottom line of top staff (idx 4)
                        # and top line of bottom staff (idx 5).
                        mid_y = (grid_point.y[4] + grid_point.y[5]) / 2.0
                        staff_number_1or2 = 1 if ny_proc < mid_y else 2
                    else:
                        # Fallback: single-staff or missing grid → default top
                        staff_number_1or2 = 1
                else:
                    staff_number_1or2 = s_idx + 1

                bucket_idx = 0
                for bx in system_barline_xs:
                    if nx_proc > bx:
                        bucket_idx += 1
                    else:
                        break
                global_measure = running_measure + bucket_idx
                homr_buckets.setdefault(
                    (staff_number_1or2, global_measure), []
                ).append({
                    "position": int(note.position),
                    "x": float(note.center[0] * scale_x),
                    "y": float(note.center[1] * scale_y),
                })

            # Actual fitted staff-line geometry — same data HOMR's own teaser
            # uses. Each StaffPoint has 5, 10, 15... y-values (multi-staff
            # grand staves are merged into a single Staff with 10+ lines).
            grid_points = []
            for point in staff.grid:
                if len(point.y) == 0:
                    continue
                grid_points.append({
                    "x": float(point.x * scale_x),
                    "ys": [float(y * scale_y) for y in point.y],
                })

            staff_info.append({
                "staff": staff_idx,
                "min_x": float(staff.min_x * scale_x),
                "max_x": float(staff.max_x * scale_x),
                "min_y": float(staff.min_y * scale_y),
                "max_y": float(staff.max_y * scale_y),
                "unit_size": float(staff.average_unit_size * scale_y),
                "line_grid": grid_points,
            })

            # Barlines on this staff
            for barline in staff.get_bar_lines():
                cx, cy = barline.box.center
                bw, bh = barline.box.size
                barline_info.append({
                    "staff_idx": staff_idx,
                    "x": float(cx * scale_x),
                    "y": float(cy * scale_y),
                    "width": float(bw * scale_x),
                    "height": float(bh * scale_y),
                })

            # Notes on this staff (include position for geometric pitch)
            for note in staff.get_notes():
                nx, ny = note.center
                note_info.append({
                    "staff_idx": staff_idx,
                    "x": float(nx * scale_x),
                    "y": float(ny * scale_y),
                    "width": float(note.box.size[0] * scale_x),
                    "height": float(note.box.size[1] * scale_y),
                    "position": note.position,
                })

            # Rests on this staff
            for symbol in staff.get_all_except_notes():
                if isinstance(symbol, Rest):
                    rx, ry = symbol.center
                    rw, rh = symbol.box.size
                    rest_info.append({
                        "staff_idx": staff_idx,
                        "x": float(rx * scale_x),
                        "y": float(ry * scale_y),
                        "width": float(rw * scale_x),
                        "height": float(rh * scale_y),
                    })

        # Advance global measure counter by the number of measures in this
        # system so the next system's buckets land on the right global #.
        # NOTE: this trusts HOMR's barline count. If HOMR drops a final
        # double-barline, the next system drifts. The downstream sanity
        # check in recompute_pitches_with_confidence will catch count
        # mismatches per staff and fall back to transformer pitch.
        running_measure += num_measures_in_system

    # Clef/key symbol bounding boxes (for clef classification)
    clef_key_info = []
    for ck_box in symbols.clefs_keys:
        cx, cy = ck_box.center
        cw, ch = ck_box.size
        clef_key_info.append({
            "x": float(cx * scale_x),
            "y": float(cy * scale_y),
            "width": float(cw * scale_x),
            "height": float(ch * scale_y),
        })

    print(f"[HOMR] Extracted: {len(staff_info)} staves, "
          f"{len(barline_info)} barlines, {len(note_info)} notes, "
          f"{len(rest_info)} rests, {len(clef_key_info)} clef/key symbols, "
          f"{len(homr_buckets)} measure-buckets for pitch resolution")

    return (musicxml_path, staff_info, barline_info, note_info, rest_info,
            clef_key_info, dewarped_staff0, dewarp_error, homr_buckets)


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
            # Run HOMR via Python API
            (musicxml_path, staff_info, barline_info, note_info, rest_info,
             clef_key_info, dewarped_staff0, dewarp_error,
             homr_buckets) = run_homr_api(tmp_path, use_gpu=True)

            with open(musicxml_path, "r", encoding="utf-8") as f:
                musicxml_content = f.read()

            parsed = parse_musicxml_to_json(
                musicxml_content,
                default_clef=clef,
                default_tempo=tempo,
                default_time_signature=time_signature,
            )

            # Targeted pitch correction: only override transformer pitches
            # that are wildly wrong (> 12 semitones from geometric position).
            # Uses global per-staff sequential zip — simpler and avoids the
            # system→measure mapping problem (HOMR overcounts barlines).
            correction_count = 0
            try:
                from pitch_from_position import position_to_midi, _effective_clef_at
                from collections import defaultdict
                clef_changes_list = parsed.get("metadata", {}).get("clef_changes") or []
                staff_clefs_map = parsed.get("metadata", {}).get("staff_clefs") or {}
                fifths_raw = parsed.get("metadata", {}).get("fifths", 0)
                fifths_val = int(fifths_raw) if fifths_raw else 0

                # Build global per-staff HOMR note lists sorted by x
                homr_by_staff: dict = defaultdict(list)
                for si, sp in enumerate(staff_info):
                    lg = sp.get("line_grid", [])
                    if not lg or len(lg) < 2:
                        continue
                    mid_g = lg[len(lg)//2]
                    ys_vals = mid_g.get("ys", [])
                    if len(ys_vals) < 10:
                        # Single staff — all notes are staff 1
                        for ni in sorted([n for n in note_info if n.get("staff_idx") == si],
                                         key=lambda n: (n["x"], n["y"])):
                            homr_by_staff[1].append(ni)
                        continue
                    mid_y_val = (ys_vals[4] + ys_vals[5]) / 2.0
                    for ni in sorted([n for n in note_info if n.get("staff_idx") == si],
                                     key=lambda n: (n["x"], n["y"])):
                        ps = 1 if ni["y"] < mid_y_val else 2
                        homr_by_staff[ps].append(ni)

                # Global per-staff zip
                for phys_staff in [1, 2]:
                    parsed_list = [n for n in parsed["notes"]
                                   if (n.get("staff") or 1) == phys_staff]
                    homr_list = homr_by_staff.get(phys_staff, [])
                    n = min(len(parsed_list), len(homr_list))
                    for i in range(n):
                        pn = parsed_list[i]
                        hn = homr_list[i]
                        pos = hn.get("position")
                        if pos is None:
                            continue
                        m = pn.get("measure", 1)
                        clef = _effective_clef_at(
                            clef_changes_list, phys_staff, m,
                            staff_clefs_map.get(str(phys_staff), "treble")
                        )
                        geo_midi, geo_name = position_to_midi(int(pos), clef, fifths_val)
                        t_midi = pn.get("pitch", 0)
                        if abs(t_midi - geo_midi) > 7:
                            pn["pitch"] = geo_midi
                            pn["pitch_name"] = geo_name
                            correction_count += 1

                parsed["metadata"]["pitch_corrections"] = correction_count
                print(f"[HOMR] Targeted pitch correction: {correction_count} notes "
                      f"(staff1: {len(homr_by_staff[1])}h/{sum(1 for n in parsed['notes'] if n.get('staff')==1)}p, "
                      f"staff2: {len(homr_by_staff[2])}h/{sum(1 for n in parsed['notes'] if n.get('staff')==2)}p)")
            except Exception as e:
                print(f"[HOMR] Targeted pitch correction failed: {e}")
                import traceback; traceback.print_exc()

            # Post-process step 0: geometric clef detection + pitch recomputation
            # HOMR's transformer sometimes gets the clef wrong, which shifts all pitches.
            # Use visual analysis of the clef region to validate/correct.
            # Only override when we have HIGH confidence — avoid false positives.
            geometric_clef_status = "skipped"
            try:
                from clef_classifier import find_clef_for_staff
                from pitch_from_position import recompute_pitches

                import cv2 as _cv2
                original_img = _cv2.imread(tmp_path)

                # Classify clef from dewarped staff 0 image
                visual_clef = None
                visual_confidence = 0.0
                if staff_info:
                    visual_clef, visual_confidence = find_clef_for_staff(
                        original_img, staff_info[0], clef_key_info,
                        dewarped_staff=dewarped_staff0,
                    )
                    print(f"[HOMR] Visual clef: {visual_clef} (confidence: {visual_confidence:.2f})")

                # Transformer's clef (from MusicXML parsing)
                transformer_clef = parsed.get("metadata", {}).get("clef", clef)
                print(f"[HOMR] Transformer clef: {transformer_clef}")

                # Decision logic:
                # - Only override transformer when visual confidence is HIGH (>0.7)
                #   AND visual disagrees with transformer.
                # - Low confidence = defer to transformer (avoid false positives).
                determined_clef = transformer_clef

                if visual_clef and visual_clef != transformer_clef and visual_confidence >= 0.7:
                    determined_clef = visual_clef
                    geometric_clef_status = f"override:{visual_clef}(conf={visual_confidence:.2f}),was:{transformer_clef}"
                elif visual_clef and visual_clef == transformer_clef:
                    geometric_clef_status = f"confirmed:{visual_clef}(conf={visual_confidence:.2f})"
                elif visual_clef and visual_confidence < 0.7:
                    geometric_clef_status = f"low_conf:{visual_clef}(conf={visual_confidence:.2f}),keeping:{transformer_clef}"
                else:
                    geometric_clef_status = f"no_visual,keeping:{transformer_clef}"

                # If overriding, recompute pitches from note positions
                if determined_clef != transformer_clef:
                    print(f"[HOMR] Clef override: {transformer_clef} → {determined_clef}")
                    fifths_str = parsed.get("metadata", {}).get("fifths", 0)
                    fifths = int(fifths_str) if fifths_str else 0
                    parsed["notes"] = recompute_pitches(
                        parsed["notes"], note_info, determined_clef, fifths
                    )
                    parsed["metadata"]["clef"] = determined_clef
                    parsed["metadata"]["clef_override"] = True
                    geometric_clef_status += f",recomputed"
                else:
                    print(f"[HOMR] Clef confirmed: {determined_clef}")

            except Exception as e:
                geometric_clef_status = f"error: {e}"
                traceback.print_exc()

            # Post-process step 1: classify barlines as repeat/normal
            # using image-based dot detection (replaces HOMR's unreliable transformer)
            total_m = parsed.get("metadata", {}).get("total_measures", 0)
            repeat_status = "skipped"
            repeat_markers = parsed.get("repeat_markers", [])
            homr_repeat_count = len(repeat_markers)
            try:
                notes_list = parsed.get("notes", [])
                rests_list = parsed.get("rests", [])
                classified_barlines = detect_repeat_barlines(
                    tmp_path, barline_info, staff_info,
                    total_measures=total_m,
                    notes=notes_list,
                    note_positions=note_info,
                    rests=rests_list,
                    rest_positions=rest_info,
                    debug=True,
                )
                custom_markers = build_repeat_markers(classified_barlines, debug=True)
                if custom_markers:
                    # Custom detection found repeats — use these instead of HOMR's
                    repeat_markers = custom_markers
                    repeat_status = f"custom:{len(custom_markers)} (homr:{homr_repeat_count})"
                else:
                    # No repeats found by custom detector — fall back to HOMR's
                    repeat_status = f"homr_fallback:{homr_repeat_count}"
            except Exception as e:
                repeat_status = f"error: {e}"
                traceback.print_exc()

            # Post-process step 2: detect voltas (augments repeat markers)
            volta_status = "skipped"
            try:
                repeat_markers = detect_voltas(
                    tmp_path, repeat_markers, total_m,
                    staff_info=staff_info, barline_info=barline_info,
                )
                has_voltas = any(rm.get("volta_endings") for rm in repeat_markers)
                volta_status = "detected" if has_voltas else "none_found"
            except Exception as e:
                volta_status = f"error: {e}"
                traceback.print_exc()

            processing_time = time.time() - start_time

            notes = parsed.get("notes", [])
            rests = parsed.get("rests", [])
            metadata = parsed.get("metadata", {})
            metadata["processing_time"] = round(processing_time, 2)
            metadata["detection_method"] = "homr"
            metadata["handler_version"] = HANDLER_VERSION
            metadata["repeat_detection"] = repeat_status
            metadata["volta_detection"] = volta_status
            metadata["staves_detected"] = len(staff_info)
            metadata["barlines_detected"] = len(barline_info)
            metadata["notes_with_positions"] = len(note_info)
            metadata["rests_with_positions"] = len(rest_info)
            metadata["geometric_clef"] = geometric_clef_status
            metadata["dewarped_available"] = dewarped_staff0 is not None
            if dewarp_error:
                metadata["dewarp_error"] = dewarp_error

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
                "note_positions": note_info,
                "rest_positions": rest_info,
                "musicxml": musicxml_content,
                "message": f"HOMR: {len(notes)} notes, {len(rests)} rests, "
                           f"{len(repeat_markers)} repeats, "
                           f"{len(note_info)} note positions, "
                           f"{len(barline_info)} barline positions",
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
