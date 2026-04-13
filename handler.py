"""
HOMR OMR — RunPod Serverless Handler

Calls HOMR's Python API directly to access internal staff, barline,
and note pixel positions from the segmentation pipeline.

This code is licensed under AGPL-3.0 to comply with HOMR's license.
"""

HANDLER_VERSION = "4.2-grandstaff-split"

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


# ---------------------------------------------------------------------------
# Monkey-patch: Capture per-note pitch/rhythm confidence from decoder logits
# ---------------------------------------------------------------------------
# The HOMR decoder outputs raw logits for 5 heads at each autoregressive
# step (decoder_inference.py line 83). The top-k filter (lines 96-100)
# then destroys the probability distribution. We intercept BEFORE top-k
# to compute softmax on the raw logits, giving true model confidence.
#
# Thread-local storage holds per-symbol confidence lists. The patched
# generate() populates them; the handler reads them after each call to
# parse_staffs(). Cleared between runs.
#
# This avoids modifying the installed HOMR package directly.
# ---------------------------------------------------------------------------

import threading

_confidence_store = threading.local()


def _get_confidence_store() -> list:
    """Get the per-symbol confidence list for the current thread."""
    if not hasattr(_confidence_store, "symbols"):
        _confidence_store.symbols = []
    return _confidence_store.symbols


def _clear_confidence_store():
    """Reset confidence store before a new run."""
    _confidence_store.symbols = []


def _install_confidence_patch():
    """Monkey-patch ScoreDecoder.generate to capture raw logit confidence.

    Must be called AFTER HOMR's imports so the class exists.
    Idempotent — safe to call multiple times.

    The original generate() uses self.net.run_with_iobinding() (NOT
    self.net.run), so we intercept run_with_iobinding to read the raw
    logits from the io_binding outputs after each decoder step.

    Each confidence entry is stored WITH the rhythm token string so that
    downstream code can filter to note-bearing entries only.
    """
    from homr.transformer.decoder_inference import ScoreDecoder
    from homr.transformer.utils import softmax as _softmax

    if getattr(ScoreDecoder, "_confidence_patched", False):
        return  # Already patched

    _original_generate = ScoreDecoder.generate

    def _patched_generate(self, start_tokens, nonote_tokens,
                          temperature=1.0, filter_thres=0.7, **kwargs):
        """Wraps the original generate() to capture raw logit confidence
        per symbol before top-k filtering distorts the distribution."""
        import numpy as np

        # The original generate() calls self.net.run_with_iobinding(),
        # then reads results via self.io_binding.get_outputs(). We
        # intercept run_with_iobinding to capture the raw logits from
        # the io_binding AFTER each inference step completes.
        _original_rwib = self.net.run_with_iobinding
        step_confidences = []

        def _capturing_rwib(**rwib_kwargs):
            _original_rwib(**rwib_kwargs)
            # After the original runs, outputs are in the io_binding.
            # Read them to extract raw logits for confidence scoring.
            try:
                outputs = self.io_binding.get_outputs()
                if len(outputs) >= 5:
                    rhythmsp = outputs[0].numpy()
                    pitchsp = outputs[1].numpy()
                    # Compute softmax on raw (pre-filter) logits
                    raw_pitch_probs = _softmax(pitchsp[:, -1, :], dim=-1)
                    raw_rhythm_probs = _softmax(rhythmsp[:, -1, :], dim=-1)
                    # Flatten to 1-D
                    pitch_flat = raw_pitch_probs.ravel()
                    rhythm_flat = raw_rhythm_probs.ravel()
                    # Top-3 pitch alternatives
                    top3_idx = np.argsort(pitch_flat)[-3:][::-1]
                    top3 = [(int(idx), float(pitch_flat[idx])) for idx in top3_idx]
                    step_confidences.append({
                        "pitch_confidence": float(pitch_flat.max()),
                        "rhythm_confidence": float(rhythm_flat.max()),
                        "pitch_top3": top3,
                    })
            except Exception:
                # Don't let confidence capture crash the decoder
                step_confidences.append({
                    "pitch_confidence": 0.5,
                    "rhythm_confidence": 0.5,
                    "pitch_top3": [],
                })

        self.net.run_with_iobinding = _capturing_rwib
        try:
            symbols = _original_generate(
                self, start_tokens, nonote_tokens,
                temperature=temperature, filter_thres=filter_thres,
                **kwargs,
            )
        finally:
            self.net.run_with_iobinding = _original_rwib

        # Store confidence alongside symbols in thread-local.
        # Include the rhythm token string so downstream code can filter
        # to note-bearing entries (note_*) vs structural symbols
        # (barlines, clefs, key/time signatures, chord markers, etc.).
        store = _get_confidence_store()
        for i, sym in enumerate(symbols):
            conf = step_confidences[i] if i < len(step_confidences) else {
                "pitch_confidence": 0.5, "rhythm_confidence": 0.5, "pitch_top3": [],
            }
            conf["rhythm_token"] = sym.rhythm
            store.append(conf)

        return symbols

    ScoreDecoder.generate = _patched_generate
    ScoreDecoder._confidence_patched = True
    print("[HOMR] Confidence monkey-patch installed on ScoreDecoder.generate")


def split_grandstaff(staff):
    """
    Split a merged 10-line grand Staff back into two 5-line sub-Staffs.

    The merged Staff has grid points with 10 y-values each (lines 0-4 = top,
    lines 5-9 = bottom). We split these into two separate grids.

    Symbols are partitioned by y-coordinate: above the midpoint between
    line 4 and line 5 goes to top staff, below goes to bottom staff.
    """
    from homr.model import Staff, StaffPoint

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
    for symbol in staff.symbols:
        sx = symbol.center[0]
        sy = symbol.center[1]
        closest = min(staff.grid, key=lambda p: abs(p.x - sx))
        mid_y = (closest.y[4] + closest.y[5]) / 2.0
        if sy < mid_y:
            top_staff.add_symbol(symbol)
        else:
            bot_staff.add_symbol(symbol)

    return top_staff, bot_staff


def parse_staffs_with_grandstaff_split(debug, multi_staffs, image, config, selected_staff=-1):
    """
    Custom parse_staffs that splits grand staves before transformer inference.

    Each 5-line sub-staff gets full 256px resolution in the transformer
    instead of both sharing 256px (half resolution each).

    Returns 2 voices: treble (voice 0) and bass (voice 1).
    For non-grand-staff pieces, falls through to standard parse_staffs.
    """
    from homr.model import Staff, StaffPoint, MultiStaff
    from homr.staff_parsing import parse_staff_image
    from homr.staff_regions import StaffRegions
    from homr.transformer.vocabulary import EncodedSymbol, remove_duplicated_symbols
    from homr.simple_logging import eprint

    # Check if any multi_staff has a grand staff
    has_grandstaff = any(
        len(ms.staffs) == 1 and ms.staffs[0].is_grandstaff
        for ms in multi_staffs
    )

    if not has_grandstaff:
        from homr.staff_parsing import parse_staffs
        eprint("[SPLIT] No grand staves detected — using standard parse_staffs")
        return parse_staffs(debug, multi_staffs, image, config=config, selected_staff=selected_staff)

    eprint("[SPLIT] Grand staff detected — splitting for full-resolution transformer inference")

    # Build split staves
    split_ms = []
    for ms in multi_staffs:
        if len(ms.staffs) == 1 and ms.staffs[0].is_grandstaff:
            top, bot = split_grandstaff(ms.staffs[0])
            split_ms.append(MultiStaff([top, bot], ms.connections))
        else:
            split_ms.append(ms)

    # Create tight regions with phantom boundary staves to prevent
    # crop overlap between split sub-staves
    phantom_ms = list(split_ms)
    for ms in split_ms:
        if len(ms.staffs) == 2:
            top = ms.staffs[0]
            bot = ms.staffs[1]
            gap_mid = (top.max_y + bot.min_y) / 2.0
            phantom_grid = [
                StaffPoint(top.grid[0].x,
                          [gap_mid - 2, gap_mid - 1, gap_mid, gap_mid + 1, gap_mid + 2],
                          0.0),
                StaffPoint(top.grid[-1].x,
                          [gap_mid - 2, gap_mid - 1, gap_mid, gap_mid + 1, gap_mid + 2],
                          0.0),
            ]
            phantom_ms.append(MultiStaff([Staff(phantom_grid)], []))

    regions = StaffRegions(phantom_ms)

    # Process each system
    i = 0
    voices_treble = []
    voices_bass = []

    for sys_idx, ms in enumerate(multi_staffs):
        if len(ms.staffs) != 1 or not ms.staffs[0].is_grandstaff:
            staff = ms.staffs[0]
            if selected_staff >= 0 and sys_idx != selected_staff:
                i += 1
                continue
            result = parse_staff_image(debug, i, staff, image, regions, config)
            if len(result) > 0:
                result.append(EncodedSymbol("newline"))
                voices_treble.extend(result)
            i += 1
            continue

        merged = ms.staffs[0]
        top_staff, bot_staff = split_grandstaff(merged)

        eprint(f"[SPLIT] System {sys_idx}: top={top_staff.get_number_of_notes()} notes, "
               f"bot={bot_staff.get_number_of_notes()} notes")

        top_result = parse_staff_image(debug, i, top_staff, image, regions, config)
        i += 1
        if len(top_result) > 0:
            top_result.append(EncodedSymbol("newline"))
            voices_treble.extend(top_result)

        bot_result = parse_staff_image(debug, i, bot_staff, image, regions, config)
        i += 1
        if len(bot_result) > 0:
            bot_result.append(EncodedSymbol("newline"))
            voices_bass.extend(bot_result)

        eprint(f"[SPLIT] System {sys_idx}: top={len(top_result)} symbols, "
               f"bot={len(bot_result)} symbols")

    result = [
        remove_duplicated_symbols(voices_treble),
        remove_duplicated_symbols(voices_bass),
    ]

    treble_n = sum(1 for s in result[0] if s.rhythm.startswith("note_"))
    bass_n = sum(1 for s in result[1] if s.rhythm.startswith("note_"))
    eprint(f"[SPLIT] Result: {treble_n} treble + {bass_n} bass = {treble_n + bass_n} total")

    return result


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
    # Install confidence extraction monkey-patch (idempotent)
    _install_confidence_patch()
    _clear_confidence_store()

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
    from homr.model import BarLine, MultiStaff
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
    # Feature flag: split grand staves before transformer inference.
    # When enabled, each 5-line sub-staff gets full 256px resolution instead
    # of both sharing 256px. Produces 2-part MusicXML for grand staff pieces.
    # The parser now supports multi-part MusicXML (part 2 → staff 2).
    grandstaff_split_enabled = os.environ.get("GRANDSTAFF_SPLIT_ENABLED", "false").lower() == "true"
    transformer_config = TransformerConfig()
    transformer_config.use_gpu_inference = use_gpu

    if grandstaff_split_enabled:
        result_staffs = parse_staffs_with_grandstaff_split(
            debug, multi_staffs, predictions.preprocessed,
            config=transformer_config, selected_staff=-1,
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
            # For 20-ys staves (2 grand staves merged), compute sub_system
            # to split notes back into separate rows of music.
            grid_ys_count = len(staff.grid[0].y) if staff.grid else 5
            has_sub_systems = grid_ys_count >= 20

            for note in staff.get_notes():
                nx, ny = note.center
                sub_system = 0
                note_staff = 1  # default: single staff or treble

                if is_grand and staff.grid:
                    gp = staff.get_at(nx)
                    if gp is None:
                        gp = min(staff.grid, key=lambda p: abs(p.x - nx))

                    if gp and has_sub_systems and len(gp.y) >= 20:
                        # 20-ys = 2 grand staves merged. Split into sub-systems
                        # using the gap between lines 9 and 10 (between the two
                        # grand staves), then classify treble/bass within each.
                        gap_y = (gp.y[9] + gp.y[10]) / 2.0
                        if ny < gap_y:
                            sub_system = 0
                            mid_y = (gp.y[4] + gp.y[5]) / 2.0
                        else:
                            sub_system = 1
                            mid_y = (gp.y[14] + gp.y[15]) / 2.0
                        note_staff = 1 if ny < mid_y else 2
                    elif gp and len(gp.y) >= 10:
                        # Normal 10-ys grand staff
                        mid_y = (gp.y[4] + gp.y[5]) / 2.0
                        note_staff = 1 if ny < mid_y else 2

                note_info.append({
                    "staff_idx": staff_idx,
                    "system_idx": ms_idx,
                    "sub_system": sub_system,
                    "staff_number": note_staff,
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

            # --- Step 0: Apply captured confidence data to parsed notes ---
            # Replace hardcoded confidence: 0.9 with actual pitch confidence
            # from the decoder's raw logits (captured by monkey-patch).
            # This MUST run first so downstream steps can use pitch_confidence.
            #
            # The confidence store has one entry per decoder symbol across
            # ALL staves — including barlines, clefs, key/time signatures,
            # chord markers, volta markers, and tieSlur tokens. Parsed notes
            # only include note_* symbols. We filter the store to note-bearing
            # entries (rhythm_token starts with "note_") before mapping by
            # index to avoid the off-by-N drift that previously corrupted
            # every confidence value after the first non-note symbol.
            confidence_data = _get_confidence_store()
            conf_applied = 0
            if confidence_data:
                parsed_notes = parsed.get("notes", [])
                # Filter to only note-bearing decoder symbols
                note_confidence = [
                    cd for cd in confidence_data
                    if cd.get("rhythm_token", "").startswith("note_")
                ]
                total_symbols = len(confidence_data)
                filtered_count = len(note_confidence)
                print(f"[HOMR] Confidence store: {total_symbols} total symbols, "
                      f"{filtered_count} note-bearing (filtered from "
                      f"{total_symbols - filtered_count} non-note symbols)")
                for i, note in enumerate(parsed_notes):
                    if i < len(note_confidence):
                        cd = note_confidence[i]
                        note["confidence"] = cd["pitch_confidence"]
                        note["pitch_confidence"] = cd["pitch_confidence"]
                        note["rhythm_confidence"] = cd["rhythm_confidence"]
                        note["pitch_alternatives"] = cd.get("pitch_top3", [])
                        conf_applied += 1
                print(f"[HOMR] Applied confidence data to {conf_applied}/{len(parsed_notes)} notes "
                      f"(note-bearing entries: {filtered_count})")
                if filtered_count != len(parsed_notes):
                    print(f"[HOMR] WARNING: note confidence count ({filtered_count}) != "
                          f"parsed note count ({len(parsed_notes)})")
            else:
                print("[HOMR] No confidence data captured (monkey-patch may not have fired)")

            # --- Step 0b: geometric clef detection + pitch recomputation ---
            # HOMR's transformer sometimes gets the clef wrong, which shifts all pitches.
            # Use visual analysis of the clef region to validate/correct.
            # Only override when we have HIGH confidence — avoid false positives.
            geometric_clef_status = "skipped"
            try:
                from clef_classifier import find_clef_for_staff
                from pitch_from_position import recompute_pitches_with_confidence

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

                # If overriding, inject the clef override into metadata so
                # Step 1's pitch recomputation uses the corrected clef.
                # Do NOT recompute pitches here — single pass in Step 1
                # avoids the double-pass overwrite bug (expert review #14).
                if determined_clef != transformer_clef:
                    print(f"[HOMR] Clef override: {transformer_clef} → {determined_clef}")
                    parsed["metadata"]["clef"] = determined_clef
                    parsed["metadata"]["clef_override"] = True
                    # Inject override into clef_changes so Step 1 picks it up
                    existing_changes = parsed.get("metadata", {}).get("clef_changes") or []
                    existing_changes.insert(0, {"staff": 1, "measure": 1, "clef": determined_clef})
                    parsed["metadata"]["clef_changes"] = existing_changes
                    # Also update staff_clefs default
                    staff_clefs = parsed.get("metadata", {}).get("staff_clefs") or {}
                    staff_clefs["1"] = determined_clef
                    parsed["metadata"]["staff_clefs"] = staff_clefs
                    geometric_clef_status += f",injected_for_step1"
                else:
                    print(f"[HOMR] Clef confirmed: {determined_clef}")

            except Exception as e:
                geometric_clef_status = f"error: {e}"
                traceback.print_exc()

            # --- Step 1: Geometric pitch cross-check ---
            # Selective override: only correct transformer pitch when
            # geometric position disagrees by >= 7 semitones (obvious error)
            # or when disagreement < 7 AND pitch_confidence < 0.5.
            # Feature-flagged for safe rollout. Runs AFTER clef override
            # so it benefits from corrected clef when applicable.
            geometric_pitch_enabled = os.environ.get("GEOMETRIC_PITCH_ENABLED", "true").lower() == "true"
            pitch_correction_status = "disabled"
            if geometric_pitch_enabled and note_info:
                try:
                    from spatial_pitch_alignment import apply_geometric_pitch
                    clef_changes_list = parsed.get("metadata", {}).get("clef_changes", []) or []
                    fifths_val = int(parsed.get("metadata", {}).get("fifths", 0) or 0)
                    staff_clefs_default = {}
                    raw_staff_clefs = parsed.get("metadata", {}).get("staff_clefs")
                    if raw_staff_clefs:
                        staff_clefs_default = {int(k): v for k, v in raw_staff_clefs.items()}

                    parsed["notes"] = apply_geometric_pitch(
                        parsed["notes"],
                        note_info,
                        clef_changes=clef_changes_list,
                        staff_clefs_default=staff_clefs_default,
                        fifths=fifths_val,
                    )
                    pitch_correction_status = "spatial_align"
                except Exception as e:
                    pitch_correction_status = f"error: {e}"
                    traceback.print_exc()
            parsed["metadata"]["pitch_corrections"] = pitch_correction_status

            # Post-process step 2a: classify barlines as repeat/normal
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

            # Post-process step 2b: detect voltas (augments repeat markers)
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
            metadata["grandstaff_split"] = os.environ.get("GRANDSTAFF_SPLIT_ENABLED", "false").lower() == "true"
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
            base = os.path.splitext(tmp_path)[0]
            for path in [tmp_path, base + ".musicxml", base + ".xml"]:
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
