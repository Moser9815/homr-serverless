"""
MusicXML Parser — Converts HOMR's MusicXML output to structured note/rest JSON.

Produces the same format as the oemer endpoint so the iOS app can use either
backend with zero changes.

This code is licensed under AGPL-3.0 to comply with HOMR's license.
"""

import xml.etree.ElementTree as ET
from typing import Any

# MusicXML duration type mapping (divisions-relative → named duration)
# Standard MusicXML type names
MUSICXML_TYPE_TO_DURATION = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "16th": 0.25,
    "32nd": 0.125,
    "64th": 0.0625,
}

# Pitch name mapping
STEP_TO_PITCH_CLASS = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Key signature: fifths value → key name
FIFTHS_TO_KEY = {
    -7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
}


def midi_pitch(step: str, octave: int, alter: int = 0) -> int:
    """Convert MusicXML pitch (step, octave, alter) to MIDI number."""
    return (octave + 1) * 12 + STEP_TO_PITCH_CLASS[step] + alter


def pitch_name_from_midi(midi: int) -> str:
    """Convert MIDI number to human-readable pitch name (e.g., 'C4', 'F#3')."""
    note = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1
    return f"{note}{octave}"


def duration_type_name(type_text: str, dots: int = 0) -> str:
    """Convert MusicXML type text + dots to our duration type name."""
    base_map = {
        "whole": "whole",
        "half": "half",
        "quarter": "quarter",
        "eighth": "eighth",
        "16th": "sixteenth",
        "32nd": "thirtySecond",
        "64th": "sixtyFourth",
    }
    name = base_map.get(type_text, type_text)
    if dots == 1:
        name = "dotted" + name[0].upper() + name[1:]
    return name


def duration_beats(type_text: str, dots: int = 0) -> float:
    """Calculate duration in beats from MusicXML type + dots."""
    base = MUSICXML_TYPE_TO_DURATION.get(type_text, 1.0)
    if dots == 1:
        base *= 1.5
    elif dots == 2:
        base *= 1.75
    return base


def parse_musicxml_to_json(
    musicxml_content: str,
    default_clef: str = "treble",
    default_tempo: int = 120,
    default_time_signature: str = "4/4",
) -> dict[str, Any]:
    """
    Parse MusicXML content into structured note/rest JSON.

    Returns the same format as the oemer endpoint:
    {
        "notes": [...],
        "rests": [...],
        "metadata": {
            "tempo", "time_signature", "clef", "detected_key",
            "total_measures", ...
        }
    }
    """
    root = ET.fromstring(musicxml_content)

    # MusicXML namespace handling — some files use a namespace, some don't
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

    # Extract parts
    parts = findall(root, "part")
    if not parts:
        # Try score-partwise > part
        score = find(root, "score-partwise")
        if score is not None:
            parts = findall(score, "part")

    notes_out = []
    rests_out = []

    detected_tempo = default_tempo
    detected_time_sig = default_time_signature
    detected_key = "C"
    detected_clef = default_clef
    tempo_detected = False
    time_sig_detected = False
    total_measures = 0

    # Parse time signature components
    ts_parts = default_time_signature.split("/")
    beats_per_measure = int(ts_parts[0])
    beat_unit = int(ts_parts[1])

    for part in parts:
        measures = findall(part, "measure")
        current_time = 0.0  # Running time in seconds
        current_beat = 1.0  # Current beat within measure
        divisions = 1  # MusicXML divisions per quarter note

        for measure in measures:
            measure_number = int(measure.get("number", "1"))
            total_measures = max(total_measures, measure_number)
            current_beat = 1.0

            # Check for attributes (time signature, key, clef, divisions)
            attributes = find(measure, "attributes")
            if attributes is not None:
                div_el = find(attributes, "divisions")
                if div_el is not None and div_el.text:
                    divisions = int(div_el.text)

                # Time signature
                time_el = find(attributes, "time")
                if time_el is not None:
                    beat_text = find_text(time_el, "beats")
                    beat_type_text = find_text(time_el, "beat-type")
                    if beat_text and beat_type_text:
                        beats_per_measure = int(beat_text)
                        beat_unit = int(beat_type_text)
                        detected_time_sig = f"{beats_per_measure}/{beat_unit}"
                        time_sig_detected = True

                # Key signature
                key_el = find(attributes, "key")
                if key_el is not None:
                    fifths_text = find_text(key_el, "fifths")
                    if fifths_text:
                        fifths = int(fifths_text)
                        detected_key = FIFTHS_TO_KEY.get(fifths, "C")

                # Clef
                clef_el = find(attributes, "clef")
                if clef_el is not None:
                    sign = find_text(clef_el, "sign", "G")
                    line = find_text(clef_el, "line", "2")
                    if sign == "G":
                        detected_clef = "treble"
                    elif sign == "F":
                        detected_clef = "bass"
                    elif sign == "C":
                        detected_clef = "alto"

            # Check for tempo (direction > sound)
            for direction in findall(measure, "direction"):
                sound = find(direction, "sound")
                if sound is not None and sound.get("tempo"):
                    detected_tempo = int(float(sound.get("tempo")))
                    tempo_detected = True

            # Calculate seconds per beat based on current tempo
            seconds_per_beat = 60.0 / detected_tempo

            # Process notes and rests
            for note_el in findall(measure, "note"):
                # Check for chord (simultaneous notes) — don't advance time
                is_chord = find(note_el, "chord") is not None

                # Get duration in divisions
                dur_div_el = find(note_el, "duration")
                dur_divisions = int(dur_div_el.text) if dur_div_el is not None and dur_div_el.text else divisions

                # Get note type (quarter, eighth, etc.)
                type_el = find(note_el, "type")
                type_text = type_el.text if type_el is not None and type_el.text else "quarter"

                # Count dots
                dots = len(findall(note_el, "dot"))

                # Duration in beats
                dur_beats = duration_beats(type_text, dots)
                dur_seconds = dur_beats * seconds_per_beat

                # Check if rest
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
                    rests_out.append(rest_entry)
                else:
                    # Extract pitch
                    pitch_el = find(note_el, "pitch")
                    if pitch_el is not None:
                        step = find_text(pitch_el, "step", "C")
                        octave = int(find_text(pitch_el, "octave", "4"))
                        alter_text = find_text(pitch_el, "alter", "0")
                        alter = int(float(alter_text)) if alter_text else 0

                        midi = midi_pitch(step, octave, alter)
                        p_name = pitch_name_from_midi(midi)

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
                        notes_out.append(note_entry)

                # Advance time (unless chord)
                if not is_chord:
                    current_time += dur_seconds
                    current_beat += dur_beats

            # Handle forward/backup elements (voice changes, etc.)
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

        # Only process first part for now (single-staff sheets)
        break

    # Build key display name
    key_display = detected_key
    if detected_key in FIFTHS_TO_KEY.values():
        key_display = f"{detected_key} major"

    return {
        "notes": notes_out,
        "rests": rests_out,
        "metadata": {
            "tempo": detected_tempo,
            "tempo_detected": tempo_detected,
            "time_signature": detected_time_sig,
            "time_signature_detected": time_sig_detected,
            "clef": detected_clef,
            "detected_key": detected_key,
            "key_display": key_display,
            "total_measures": total_measures,
        },
    }
