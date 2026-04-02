"""
MusicXML Parser — Converts HOMR's MusicXML output to structured note/rest JSON.

Extracts notes, rests, repeat markers, volta brackets, articulations, and
other musical elements that HOMR's transformer can detect.

Produces a format compatible with the Music Learning App's iOS models.

This code is licensed under AGPL-3.0 to comply with HOMR's license.
"""

import xml.etree.ElementTree as ET
from typing import Any

# MusicXML duration type mapping
MUSICXML_TYPE_TO_DURATION = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "16th": 0.25,
    "32nd": 0.125,
    "64th": 0.0625,
}

STEP_TO_PITCH_CLASS = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

FIFTHS_TO_KEY = {
    -7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
}


def midi_pitch(step: str, octave: int, alter: int = 0) -> int:
    return (octave + 1) * 12 + STEP_TO_PITCH_CLASS[step] + alter


def pitch_name_from_midi(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"


def duration_type_name(type_text: str, dots: int = 0) -> str:
    base_map = {
        "whole": "whole",
        "half": "half",
        "quarter": "quarter",
        "eighth": "eighth",
        "16th": "sixteenth",
        "32nd": "thirtySecond",
    }
    name = base_map.get(type_text, type_text)
    if dots >= 1:
        name = "dotted-" + name
    return name


def duration_beats(type_text: str, dots: int = 0) -> float:
    base = MUSICXML_TYPE_TO_DURATION.get(type_text, 1.0)
    if dots == 1:
        base *= 1.5
    elif dots == 2:
        base *= 1.75
    return base


def _extract_articulations(note_el, ns: str) -> list[str]:
    """Extract articulation names from a note element's notations."""
    articulations = []

    def find(el, tag):
        return el.find(f"{ns}{tag}")

    def findall(el, tag):
        return el.findall(f"{ns}{tag}")

    notations = find(note_el, "notations")
    if notations is None:
        return articulations

    # Direct children of notations
    fermata = find(notations, "fermata")
    if fermata is not None:
        articulations.append("fermata")

    arpeggiate = find(notations, "arpeggiate")
    if arpeggiate is not None:
        articulations.append("arpeggiate")

    # Articulations container
    arts_el = find(notations, "articulations")
    if arts_el is not None:
        for child in arts_el:
            tag = child.tag
            if "}" in tag:
                tag = tag.split("}")[-1]
            articulations.append(tag)

    # Ornaments container
    ornaments = find(notations, "ornaments")
    if ornaments is not None:
        for child in ornaments:
            tag = child.tag
            if "}" in tag:
                tag = tag.split("}")[-1]
            articulations.append(tag)

    return articulations


def _extract_repeats_and_voltas(measures, ns: str) -> list[dict]:
    """
    Extract repeat markers and volta brackets from barline elements.

    MusicXML encodes repeats as:
    - <barline><repeat direction="forward"/></barline> = start of repeat
    - <barline><repeat direction="backward"/></barline> = end of repeat
    - <barline><ending number="1" type="start"/></barline> = volta start
    - <barline><ending number="1" type="stop"/></barline> = volta end

    Returns list matching the iOS RepeatMarker format:
    [{
        "start_measure": int,
        "end_measure": int,
        "repeat_count": 1,
        "volta_endings": {"5": [1], "6": [2]} or null
    }]
    """
    def find(el, tag):
        return el.find(f"{ns}{tag}")

    def findall(el, tag):
        return el.findall(f"{ns}{tag}")

    # Collect all repeat barlines
    forward_starts = []  # Measures with forward repeat
    backward_ends = []   # Measures with backward repeat

    # Collect volta/ending info
    volta_info = []  # (measure_number, ending_number, type)

    for measure in measures:
        measure_num = int(measure.get("number", "1"))

        for barline in findall(measure, "barline"):
            repeat = find(barline, "repeat")
            ending = find(barline, "ending")

            if repeat is not None:
                direction = repeat.get("direction", "")
                if direction == "forward":
                    forward_starts.append(measure_num)
                elif direction == "backward":
                    backward_ends.append(measure_num)

            if ending is not None:
                ending_num = ending.get("number", "1")
                ending_type = ending.get("type", "")
                volta_info.append((measure_num, ending_num, ending_type))

    # Pair forward/backward into repeat markers
    repeat_markers = []

    # Sort for pairing
    forward_starts.sort()
    backward_ends.sort()

    for end_measure in backward_ends:
        # Find the most recent forward start before this end
        start_measure = 1  # Default: repeat from measure 1
        for fs in reversed(forward_starts):
            if fs <= end_measure:
                start_measure = fs
                break

        # Remove used forward start
        if start_measure in forward_starts:
            forward_starts.remove(start_measure)

        # Check for volta endings within this repeat range
        volta_endings = None
        voltas_in_range = [
            (m, num, typ) for m, num, typ in volta_info
            if start_measure <= m <= end_measure and typ == "start"
        ]

        if voltas_in_range:
            # Build volta map: measure -> [iterations]
            volta_endings = {}
            for m, num, _ in voltas_in_range:
                try:
                    iterations = [int(n) for n in num.split(",")]
                except ValueError:
                    iterations = [int(num)]
                volta_endings[str(m)] = iterations

        repeat_markers.append({
            "start_measure": start_measure,
            "end_measure": end_measure,
            "repeat_count": 1,
            "volta_endings": volta_endings,
        })

    # Handle any remaining forward starts without a backward end
    # (open-ended repeats — less common but valid)
    # We skip these since they can't form a complete repeat marker

    return repeat_markers


def parse_musicxml_to_json(
    musicxml_content: str,
    default_clef: str = "treble",
    default_tempo: int = 120,
    default_time_signature: str = "4/4",
) -> dict[str, Any]:
    """
    Parse MusicXML content into structured JSON.

    Returns:
    {
        "notes": [...],
        "rests": [...],
        "repeat_markers": [...],
        "metadata": { ... }
    }
    """
    root = ET.fromstring(musicxml_content)

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    def find(element, tag):
        return element.find(f"{ns}{tag}")

    def findall(element, tag):
        return element.findall(f"{ns}{tag}")

    def find_text(element, tag, default=""):
        el = find(element, tag)
        return el.text if el is not None and el.text else default

    parts = findall(root, "part")
    if not parts:
        score = find(root, "score-partwise")
        if score is not None:
            parts = findall(score, "part")

    notes_out = []
    rests_out = []
    repeat_markers = []

    detected_tempo = default_tempo
    detected_time_sig = default_time_signature
    detected_key = "C"
    detected_clef = default_clef
    tempo_detected = False
    time_sig_detected = False
    total_measures = 0

    ts_parts = default_time_signature.split("/")
    beats_per_measure = int(ts_parts[0])
    beat_unit = int(ts_parts[1])

    # Track articulation counts for metadata
    articulation_counts = {}

    for part in parts:
        measures = findall(part, "measure")

        # Extract repeats and voltas from barlines
        repeat_markers = _extract_repeats_and_voltas(measures, ns)

        current_time = 0.0
        current_beat = 1.0
        divisions = 1

        for measure in measures:
            measure_number = int(measure.get("number", "1"))
            total_measures = max(total_measures, measure_number)
            current_beat = 1.0

            attributes = find(measure, "attributes")
            if attributes is not None:
                div_el = find(attributes, "divisions")
                if div_el is not None and div_el.text:
                    divisions = int(div_el.text)

                time_el = find(attributes, "time")
                if time_el is not None:
                    beat_text = find_text(time_el, "beats")
                    beat_type_text = find_text(time_el, "beat-type")
                    if beat_text and beat_type_text:
                        beats_per_measure = int(beat_text)
                        beat_unit = int(beat_type_text)
                        detected_time_sig = f"{beats_per_measure}/{beat_unit}"
                        time_sig_detected = True

                key_el = find(attributes, "key")
                if key_el is not None:
                    fifths_text = find_text(key_el, "fifths")
                    if fifths_text:
                        fifths = int(fifths_text)
                        detected_key = FIFTHS_TO_KEY.get(fifths, "C")

                clef_el = find(attributes, "clef")
                if clef_el is not None:
                    sign = find_text(clef_el, "sign", "G")
                    if sign == "G":
                        detected_clef = "treble"
                    elif sign == "F":
                        detected_clef = "bass"
                    elif sign == "C":
                        detected_clef = "alto"

            for direction in findall(measure, "direction"):
                sound = find(direction, "sound")
                if sound is not None and sound.get("tempo"):
                    detected_tempo = int(float(sound.get("tempo")))
                    tempo_detected = True

            seconds_per_beat = 60.0 / detected_tempo

            for note_el in findall(measure, "note"):
                is_chord = find(note_el, "chord") is not None

                type_el = find(note_el, "type")
                type_text = type_el.text if type_el is not None and type_el.text else "quarter"

                dots = len(findall(note_el, "dot"))

                dur_beats = duration_beats(type_text, dots)
                dur_seconds = dur_beats * seconds_per_beat

                # Extract articulations
                arts = _extract_articulations(note_el, ns)
                for a in arts:
                    articulation_counts[a] = articulation_counts.get(a, 0) + 1

                is_rest = find(note_el, "rest") is not None

                if is_rest:
                    rest_entry = {
                        "duration_type": duration_type_name(type_text, dots),
                        "duration_beats": dur_beats,
                        "measure": measure_number,
                        "beat": round(current_beat, 4),
                        "start_time": round(current_time, 4),
                        "end_time": round(current_time + dur_seconds, 4),
                        "duration": round(dur_seconds, 4),
                    }
                    # Include articulations on rests too (e.g., fermata on rest)
                    if arts:
                        rest_entry["articulations"] = arts
                    rests_out.append(rest_entry)
                else:
                    pitch_el = find(note_el, "pitch")
                    if pitch_el is not None:
                        step = find_text(pitch_el, "step", "C")
                        octave = int(find_text(pitch_el, "octave", "4"))
                        alter_text = find_text(pitch_el, "alter", "0")
                        alter = int(float(alter_text)) if alter_text else 0

                        midi = midi_pitch(step, octave, alter)
                        p_name = pitch_name_from_midi(midi)

                        # Check for grace note
                        is_grace = find(note_el, "grace") is not None

                        # Grace notes get near-zero duration so they don't
                        # overlap the principal note in the playback schedule
                        if is_grace:
                            grace_dur = 0.05
                            note_entry = {
                                "pitch": midi,
                                "pitch_name": p_name,
                                "start_time": round(current_time, 4),
                                "end_time": round(current_time + grace_dur, 4),
                                "duration": grace_dur,
                                "duration_type": duration_type_name(type_text, dots),
                                "beat": round(current_beat, 4),
                                "measure": measure_number,
                                "confidence": 0.9,
                                "is_grace": True,
                            }
                        else:
                            note_entry = {
                                "pitch": midi,
                                "pitch_name": p_name,
                                "start_time": round(current_time, 4),
                                "end_time": round(current_time + dur_seconds, 4),
                                "duration": round(dur_seconds, 4),
                                "duration_type": duration_type_name(type_text, dots),
                                "beat": round(current_beat, 4),
                                "measure": measure_number,
                                "confidence": 0.9,
                            }
                        if arts:
                            note_entry["articulations"] = arts

                        notes_out.append(note_entry)

                if not is_chord:
                    # Grace notes don't advance time
                    if find(note_el, "grace") is None:
                        current_time += dur_seconds
                        current_beat += dur_beats

            for forward in findall(measure, "forward"):
                dur_el = find(forward, "duration")
                if dur_el is not None and dur_el.text:
                    fwd_beats = int(dur_el.text) / divisions
                    current_time += fwd_beats * seconds_per_beat
                    current_beat += fwd_beats

            for backup in findall(measure, "backup"):
                dur_el = find(backup, "duration")
                if dur_el is not None and dur_el.text:
                    bk_beats = int(dur_el.text) / divisions
                    current_time -= bk_beats * seconds_per_beat
                    current_beat -= bk_beats

        # Only process first part
        break

    key_display = detected_key
    if detected_key in FIFTHS_TO_KEY.values():
        key_display = f"{detected_key} major"

    return {
        "notes": notes_out,
        "rests": rests_out,
        "repeat_markers": repeat_markers,
        "metadata": {
            "tempo": detected_tempo,
            "tempo_detected": tempo_detected,
            "time_signature": detected_time_sig,
            "time_signature_detected": time_sig_detected,
            "clef": detected_clef,
            "detected_key": detected_key,
            "key_display": key_display,
            "total_measures": total_measures,
            "articulations_detected": articulation_counts if articulation_counts else None,
        },
    }
