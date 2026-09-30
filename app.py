import streamlit as st
import librosa
import tempfile
import subprocess
import math
import re
import numpy as np
import pandas as pd
import imageio_ffmpeg
from pathlib import Path


FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="DetectTheBeat",
    page_icon="🎵",
    layout="centered",
)

st.title("🎵 DetectTheBeat")
st.write("Turn music into useful editing information.")


# =========================================================
# GLOBAL SETTINGS
# =========================================================

MAX_AUDIO_DURATION = 6 * 60
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
INTERNAL_WIDTH = 64
INTERNAL_HEIGHT = 36
HOP_LENGTH = 512
PHRASE_BEATS = 16
SWITCH_MARGIN = 0.10

FPS_OPTIONS = {
    "25 fps": {"value": 25.0, "ffmpeg": "25"},
    "23.976 fps": {"value": 24000 / 1001, "ffmpeg": "24000/1001"},
}

BEAT_INTERVALS = {
    "Every beat": 1,
    "Every 2 beats": 2,
    "Every 4 beats": 4,
}


# =========================================================
# GENERAL HELPERS
# =========================================================


def run_command(command, cwd=None):
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{result.stderr[-5000:]}")
    return result


def get_audio_duration(audio_path):
    return float(librosa.get_duration(path=audio_path))


def beat_time_to_frame(time_seconds, fps):
    return round(time_seconds * fps)


def format_time(seconds):
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes}:{remaining:05.2f}"


def save_uploaded_audio(uploaded_file, work_dir):
    extension = Path(uploaded_file.name).suffix.lower()
    if extension not in [".mp3", ".wav", ".m4a"]:
        extension = ".mp3"

    audio_path = work_dir / f"input{extension}"
    with open(audio_path, "wb") as file:
        file.write(uploaded_file.getvalue())
    return audio_path


def create_analysis_wav(audio_path, work_dir):
    analysis_path = work_dir / "analysis.wav"
    run_command(
        [
            FFMPEG_EXE,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "22050",
            str(analysis_path),
        ]
    )
    return analysis_path


# =========================================================
# FILENAME
# =========================================================


def make_output_filename(original_name, beat_choice):
    song_name = Path(original_name).stem
    song_name = re.sub(r'[<>:"/\\|?*]', "", song_name).strip()
    if not song_name:
        song_name = "Song"

    beat_labels = {
        "Every beat": "EveryBeat",
        "Every 2 beats": "Every2Beats",
        "Every 4 beats": "Every4Beats",
    }

    return f"{song_name}_DetectTheBeat_{beat_labels[beat_choice]}.mp4"


# =========================================================
# CHECKERBOARD VIDEO
# =========================================================


def create_solid_frame(width, height, color):
    return bytes(color) * (width * height)


def create_checkerboard_frame(width, height, inverted=False):
    block_size = 4
    pixels = bytearray()

    for y in range(height):
        for x in range(width):
            checker = ((x // block_size) + (y // block_size)) % 2
            if inverted:
                checker = 1 - checker

            if checker == 0:
                pixels.extend((0, 0, 0))
            else:
                pixels.extend((255, 255, 255))

    return bytes(pixels)


BLACK_FRAME = create_solid_frame(
    INTERNAL_WIDTH,
    INTERNAL_HEIGHT,
    (0, 0, 0),
)

PATTERN_A_FRAME = create_checkerboard_frame(
    INTERNAL_WIDTH,
    INTERNAL_HEIGHT,
    inverted=False,
)

PATTERN_B_FRAME = create_checkerboard_frame(
    INTERNAL_WIDTH,
    INTERNAL_HEIGHT,
    inverted=True,
)


# =========================================================
# FEATURE HELPERS
# =========================================================


def normalize_feature(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values

    low = np.percentile(values, 10)
    high = np.percentile(values, 90)

    if high <= low:
        return np.zeros_like(values)

    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def match_feature_length(values, target_length):
    values = np.asarray(values, dtype=float)

    if len(values) == target_length:
        return values
    if len(values) > target_length:
        return values[:target_length]
    if len(values) == 0:
        return np.zeros(target_length)

    padding = np.full(target_length - len(values), values[-1])
    return np.concatenate([values, padding])


def sample_curve_at_time(curve, time_seconds, sr, radius=1):
    frame = int(round(time_seconds * sr / HOP_LENGTH))
    start = max(0, frame - radius)
    end = min(len(curve), frame + radius + 1)

    if end <= start:
        return 0.0

    return float(np.max(curve[start:end]))


def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()

    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def rising_trend(values, window_frames):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values

    half = max(2, window_frames // 2)
    smoothed = moving_average(values, max(2, half // 4))
    recent = moving_average(smoothed, half)
    previous = np.roll(recent, half)

    trend = recent - previous
    trend[:half] = 0
    trend = np.maximum(trend, 0)
    return normalize_feature(trend)


# =========================================================
# BASIC BEAT V1
# =========================================================


def build_accent_curve(y, sr, onset_envelope):
    onset_curve = normalize_feature(onset_envelope)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=HOP_LENGTH,
        n_mels=32,
        fmin=30,
        fmax=1000,
        power=1.0,
    )

    mel_frequencies = librosa.mel_frequencies(
        n_mels=32,
        fmin=30,
        fmax=1000,
    )

    bass_mask = mel_frequencies <= 180

    if np.any(bass_mask):
        bass_curve = np.mean(mel[bass_mask, :], axis=0)
    else:
        bass_curve = np.zeros(mel.shape[1])

    del mel

    bass_curve = normalize_feature(bass_curve)

    rms_curve = librosa.feature.rms(
        y=y,
        frame_length=1024,
        hop_length=HOP_LENGTH,
    )[0]
    rms_curve = normalize_feature(rms_curve)

    target_length = len(onset_curve)
    bass_curve = match_feature_length(bass_curve, target_length)
    rms_curve = match_feature_length(rms_curve, target_length)

    accent_curve = (
        0.55 * onset_curve
        + 0.30 * bass_curve
        + 0.15 * rms_curve
    )

    return normalize_feature(accent_curve)


def get_median_beat_period(beat_times):
    if len(beat_times) < 2:
        return 0.5

    differences = np.diff(beat_times)
    differences = differences[differences > 0]

    if len(differences) == 0:
        return 0.5

    return float(np.median(differences))


def create_offset_candidates(beat_period, sr):
    analysis_step = HOP_LENGTH / sr
    maximum_shift = beat_period * 0.55

    offsets = np.arange(
        -maximum_shift,
        maximum_shift + analysis_step / 2,
        analysis_step,
    )
    offsets = np.append(offsets, 0.0)
    return np.unique(np.round(offsets, 6))


def get_phrase_selected_indices(block_start, block_end, phase, interval):
    selected = []

    for beat_index in range(block_start, block_end):
        local_index = beat_index - block_start
        if local_index % interval == phase:
            selected.append(beat_index)

    return selected


def score_phrase_state(
    beat_times,
    accent_curve,
    sr,
    block_start,
    block_end,
    interval,
    phase,
    offset,
    ignore_first_beats=0,
):
    selected_indices = get_phrase_selected_indices(
        block_start,
        block_end,
        phase,
        interval,
    )

    strengths = []

    for beat_index in selected_indices:
        if beat_index < block_start + ignore_first_beats:
            continue

        shifted_time = float(beat_times[beat_index]) + offset
        if shifted_time < 0:
            continue

        strength = sample_curve_at_time(
            accent_curve,
            shifted_time,
            sr,
            radius=1,
        )
        strengths.append(strength)

    if len(strengths) == 0:
        return -999.0

    strengths = np.asarray(strengths)
    median_strength = float(np.median(strengths))
    mean_strength = float(np.mean(strengths))
    lower_strength = float(np.percentile(strengths, 25))

    score = (
        0.50 * median_strength
        + 0.25 * mean_strength
        + 0.25 * lower_strength
    )

    beat_period = get_median_beat_period(beat_times)
    if beat_period > 0:
        score -= 0.025 * abs(offset) / beat_period

    return score


def find_best_phrase_state(
    beat_times,
    accent_curve,
    sr,
    block_start,
    block_end,
    interval,
    offset_candidates,
    ignore_first_beats=0,
):
    best_phase = 0
    best_offset = 0.0
    best_score = -999.0

    for phase in range(interval):
        for offset in offset_candidates:
            score = score_phrase_state(
                beat_times=beat_times,
                accent_curve=accent_curve,
                sr=sr,
                block_start=block_start,
                block_end=block_end,
                interval=interval,
                phase=phase,
                offset=float(offset),
                ignore_first_beats=ignore_first_beats,
            )

            if score > best_score:
                best_score = score
                best_phase = phase
                best_offset = float(offset)

    return best_phase, best_offset, best_score


def select_phrase_locked_beats(beat_times, accent_curve, sr, interval):
    beat_count = len(beat_times)

    if beat_count == 0:
        return [], []

    if interval == 1:
        return [float(x) for x in beat_times], []

    beat_period = get_median_beat_period(beat_times)
    offset_candidates = create_offset_candidates(beat_period, sr)

    selected_records = []
    phrase_states = []

    previous_phase = None
    previous_offset = None
    block_number = 0

    for block_start in range(0, beat_count, PHRASE_BEATS):
        block_end = min(beat_count, block_start + PHRASE_BEATS)

        if block_number == 0:
            ignore_first_beats = min(
                2,
                max(0, block_end - block_start - 1),
            )

            chosen_phase, chosen_offset, chosen_score = find_best_phrase_state(
                beat_times=beat_times,
                accent_curve=accent_curve,
                sr=sr,
                block_start=block_start,
                block_end=block_end,
                interval=interval,
                offset_candidates=offset_candidates,
                ignore_first_beats=ignore_first_beats,
            )

        else:
            best_phase, best_offset, best_score = find_best_phrase_state(
                beat_times=beat_times,
                accent_curve=accent_curve,
                sr=sr,
                block_start=block_start,
                block_end=block_end,
                interval=interval,
                offset_candidates=offset_candidates,
                ignore_first_beats=0,
            )

            previous_score = score_phrase_state(
                beat_times=beat_times,
                accent_curve=accent_curve,
                sr=sr,
                block_start=block_start,
                block_end=block_end,
                interval=interval,
                phase=previous_phase,
                offset=previous_offset,
                ignore_first_beats=0,
            )

            offset_change = abs(best_offset - previous_offset)
            offset_change_penalty = (
                0.05 * offset_change / max(beat_period, 0.001)
            )

            phase_change_penalty = 0.0
            if best_phase != previous_phase:
                phase_change_penalty = 0.04

            required_improvement = (
                SWITCH_MARGIN
                + offset_change_penalty
                + phase_change_penalty
            )

            if best_score > previous_score + required_improvement:
                chosen_phase = best_phase
                chosen_offset = best_offset
                chosen_score = best_score
            else:
                chosen_phase = previous_phase
                chosen_offset = previous_offset
                chosen_score = previous_score

        phrase_indices = get_phrase_selected_indices(
            block_start,
            block_end,
            chosen_phase,
            interval,
        )

        for beat_index in phrase_indices:
            shifted_time = float(beat_times[beat_index]) + chosen_offset
            if shifted_time <= 0:
                continue

            strength = sample_curve_at_time(
                accent_curve,
                shifted_time,
                sr,
                radius=1,
            )

            selected_records.append(
                {
                    "time": shifted_time,
                    "strength": strength,
                    "beat_index": beat_index,
                    "block": block_number,
                }
            )

        phrase_states.append(
            {
                "block": block_number,
                "start_beat": block_start,
                "end_beat": block_end,
                "phase": chosen_phase,
                "offset": chosen_offset,
                "score": chosen_score,
            }
        )

        previous_phase = chosen_phase
        previous_offset = chosen_offset
        block_number += 1

    selected_records = sorted(
        selected_records,
        key=lambda item: item["time"],
    )

    if len(selected_records) == 0:
        return [], phrase_states

    target_gap = beat_period * interval
    minimum_gap = target_gap * 0.55
    cleaned = []

    for record in selected_records:
        if not cleaned:
            cleaned.append(record)
            continue

        previous = cleaned[-1]
        gap = record["time"] - previous["time"]

        if gap < minimum_gap:
            if record["strength"] > previous["strength"]:
                cleaned[-1] = record
        else:
            cleaned.append(record)

    selected_times = [item["time"] for item in cleaned]
    return selected_times, phrase_states


# =========================================================
# SONG ANALYZER 0.1
# =========================================================


def calculate_band_activity(y, sr):
    """
    Low-frequency change: kick/bass/low percussion.
    High-frequency change: piano, guitar, claps, cymbals, bright synths.
    """

    n_fft = 2048
    magnitude = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=HOP_LENGTH,
        )
    )

    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    low_mask = (frequencies >= 30) & (frequencies <= 220)
    high_mask = (frequencies >= 600) & (frequencies <= 8000)

    low_energy = np.mean(magnitude[low_mask, :], axis=0)
    high_energy = np.mean(magnitude[high_mask, :], axis=0)

    low_log = np.log1p(low_energy)
    high_log = np.log1p(high_energy)

    low_change = np.maximum(
        np.diff(low_log, prepend=low_log[0]),
        0,
    )
    high_change = np.maximum(
        np.diff(high_log, prepend=high_log[0]),
        0,
    )

    return (
        normalize_feature(low_change),
        normalize_feature(high_change),
    )


def build_song_analysis(y, sr):
    """
    Build independent musical signals.
    This describes the song; it does not choose Smart Edit cuts.
    """

    onset_envelope = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    onset = normalize_feature(onset_envelope)
    target_length = len(onset)

    rms = librosa.feature.rms(
        y=y,
        frame_length=2048,
        hop_length=HOP_LENGTH,
    )[0]
    rms = normalize_feature(
        match_feature_length(rms, target_length)
    )

    brightness = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=HOP_LENGTH,
    )[0]
    brightness = normalize_feature(
        match_feature_length(brightness, target_length)
    )

    bass_hit, high_hit = calculate_band_activity(y, sr)
    bass_hit = match_feature_length(bass_hit, target_length)
    high_hit = match_feature_length(high_hit, target_length)

    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    beat_frames = np.asarray(beat_frames, dtype=int)

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    beat_pulse = np.zeros(target_length)
    for frame in beat_frames:
        if 0 <= frame < target_length:
            beat_pulse[frame] = 1.0

    frames_per_second = sr / HOP_LENGTH

    density_window = max(
        2,
        int(round(2.0 * frames_per_second)),
    )
    onset_density = normalize_feature(
        moving_average(onset, density_window)
    )

    build_window = max(
        4,
        int(round(4.0 * frames_per_second)),
    )

    energy_rise = rising_trend(rms, build_window)
    brightness_rise = rising_trend(brightness, build_window)
    density_rise = rising_trend(onset_density, build_window)

    percussive_activity = moving_average(
        np.maximum(onset, bass_hit),
        max(2, int(frames_per_second)),
    )
    percussive_activity = normalize_feature(percussive_activity)

    build_score = (
        0.40 * energy_rise
        + 0.22 * brightness_rise
        + 0.25 * density_rise
        + 0.13 * percussive_activity
    )
    build_score = normalize_feature(build_score)

    times = librosa.frames_to_time(
        np.arange(target_length),
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    return {
        "times": times,
        "onset": onset,
        "onset_raw": onset_envelope,
        "rms": rms,
        "brightness": brightness,
        "bass_hit": bass_hit,
        "high_hit": high_hit,
        "onset_density": onset_density,
        "build_score": build_score,
        "beat_times": beat_times,
        "beat_pulse": beat_pulse,
    }


def calculate_beat_alignment(event_time, beat_times, beat_period):
    if len(beat_times) == 0:
        return 0.0

    distance = float(
        np.min(np.abs(beat_times - event_time))
    )

    tolerance = max(0.08, beat_period * 0.45)
    alignment = 1.0 - distance / tolerance

    return float(np.clip(alignment, 0.0, 1.0))


def calculate_pause_before_hit(rms, frame, sr):
    """
    High value means there was a noticeable energy drop before the hit.
    """

    fps = sr / HOP_LENGTH
    short_window = int(0.70 * fps)
    context_window = int(2.50 * fps)

    pause_start = max(0, frame - short_window)
    context_start = max(0, pause_start - context_window)

    pre_values = rms[pause_start:max(pause_start + 1, frame)]
    pre_energy = float(np.mean(pre_values)) if len(pre_values) else 0.0

    context_values = rms[
        context_start:max(context_start + 1, pause_start)
    ]

    if len(context_values) == 0:
        return 0.0

    context_energy = float(np.mean(context_values))
    energy_drop = context_energy - pre_energy
    pause_score = energy_drop * 3.0

    if pre_energy < 0.20:
        pause_score += 0.20

    return float(np.clip(pause_score, 0.0, 1.0))


def calculate_build_context(event_time, build_regions):
    """
    Return a contextual build score only when a hit occurs
    at the end of, or shortly after, a detected build region.

    This replaces the old behaviour where any strong build
    somewhere in the previous six seconds could make almost
    every later hit look like a post-build hit.
    """

    best_score = 0.0

    for build in build_regions:
        # Allow a tiny amount before the detected end because
        # our frame-based build boundary may land just after
        # the musical release.
        delta = event_time - build["End"]

        if -0.12 <= delta <= 0.80:
            if delta <= 0:
                proximity = 1.0
            else:
                proximity = 1.0 - (delta / 0.80)

            contextual_score = (
                build["Score"]
                * max(0.0, proximity)
            )

            best_score = max(
                best_score,
                contextual_score
            )

    return float(
        np.clip(
            best_score,
            0.0,
            1.0
        )
    )


def get_event_tier(opportunity):
    if opportunity >= 0.85:
        return "PRIMARY"

    if opportunity >= 0.70:
        return "STRONG"

    if opportunity >= 0.50:
        return "SECONDARY"

    return "LOW"


def calculate_event_opportunity(event):
    """
    Recalculate the analyzer/debugging opportunity score from
    the event's current components.

    Distinctive tonal attacks are protected from dropping to LOW
    only because they sit between beats. This is deliberately a
    minimum floor, not an automatic PRIMARY/STRONG promotion.
    """

    distinctive_score = max(
        event["Bass"],
        event["High / tonal"],
        event["Onset"],
    )

    context_score = max(
        event["Pause before"],
        event["Build before"],
    )

    opportunity = (
        0.40 * event["Hit strength"]
        +
        0.20 * event["Beat alignment"]
        +
        0.25 * context_score
        +
        0.15 * distinctive_score
    )

    # Protect clearly distinctive upper-frequency attacks such as
    # piano/guitar/synth stabs. They can be excellent edit points
    # even when they are intentionally off the main beat grid.
    if (
        event["High / tonal"] >= 0.75
        and
        event["Hit strength"] >= 0.45
    ):
        opportunity = max(
            opportunity,
            0.50
        )

    # Also protect genuinely loud/strong standalone attacks.
    # A hit should not fall all the way to LOW just because it is
    # between beats. This is the rule that keeps moments like the
    # useful ~16.32 s hit in Heron Vale alive.
    if (
        event["Onset"] >= 0.78
        and
        event["Hit strength"] >= 0.65
    ):
        opportunity = max(
            opportunity,
            0.50
        )

    return float(
        np.clip(
            opportunity,
            0.0,
            1.0
        )
    )


def build_event_label(event):
    labels = []

    if (
        event["Pause before"] >= 0.45
        and
        event["Onset"] >= 0.35
    ):
        labels.append(
            "Post-pause hit"
        )

    if (
        event["Build before"] >= 0.55
        and
        event["Onset"] >= 0.35
    ):
        labels.append(
            "Post-build hit"
        )

    if event["Onset"] >= 0.78:
        labels.append(
            "Very strong hit"
        )

    elif event["Onset"] >= 0.48:
        labels.append(
            "Strong hit"
        )

    else:
        labels.append(
            "Accent"
        )

    if (
        event["High / tonal"] >= 0.55
        and
        event["High / tonal"]
        >
        event["Bass"] + 0.08
    ):
        labels.append(
            "High / tonal"
        )

    if (
        event["Bass"] >= 0.55
        and
        event["Bass"]
        >
        event["High / tonal"] + 0.05
    ):
        labels.append(
            "Bass / kick"
        )

    return " + ".join(
        labels
    )


def refresh_event_ranking(event):
    """
    Recalculate score, tier and labels whenever context changes.
    """

    event["Opportunity"] = (
        calculate_event_opportunity(
            event
        )
    )

    event["Tier"] = (
        get_event_tier(
            event["Opportunity"]
        )
    )

    event["Event"] = (
        build_event_label(
            event
        )
    )

    return event


def cluster_micro_hits(events, cluster_seconds=0.12):
    """
    Collapse only peaks that fit inside one small TOTAL time window.

    Important: clustering is measured from the first event in the
    cluster, not from the previous event. This prevents chain merging:

        16.10 -> 16.20 -> 16.30 -> 16.40

    With neighbour-to-neighbour clustering that entire sequence could
    collapse even though it spans 0.30 s. With total-width clustering,
    a cluster can never become wider than cluster_seconds.
    """

    if not events:
        return [], 0

    events = sorted(
        events,
        key=lambda item: item["Time"]
    )

    clusters = []
    current_cluster = [
        events[0]
    ]
    cluster_start_time = events[0]["Time"]

    for event in events[1:]:
        total_cluster_width = (
            event["Time"]
            -
            cluster_start_time
        )

        if total_cluster_width <= cluster_seconds:
            current_cluster.append(
                event
            )

        else:
            clusters.append(
                current_cluster
            )
            current_cluster = [
                event
            ]
            cluster_start_time = (
                event["Time"]
            )

    clusters.append(
        current_cluster
    )

    merged_events = []

    for cluster in clusters:
        # Anchor timing to the strongest/most useful physical attack.
        anchor = max(
            cluster,
            key=lambda item: (
                item["Hit strength"],
                item["Opportunity"]
            )
        )

        merged = dict(
            anchor
        )

        # Preserve useful characteristics seen by neighbouring analysis
        # frames that still belong to this same very short attack.
        for key in [
            "Hit strength",
            "Beat alignment",
            "Pause before",
            "Build before",
            "Bass",
            "High / tonal",
            "Onset",
        ]:
            merged[key] = max(
                item[key]
                for item
                in cluster
            )

        refresh_event_ranking(
            merged
        )

        merged_events.append(
            merged
        )

    removed_count = (
        len(events)
        -
        len(merged_events)
    )

    return (
        merged_events,
        removed_count
    )


def keep_first_post_pause_hit(events, suppression_seconds=1.75):
    """
    A pause is most editorially useful at the FIRST meaningful hit
    after the quiet moment.

    The pause detector can naturally remain elevated for several
    following attacks because its look-back window still contains the
    same silence. Treat those as one pause episode and keep the pause
    bonus only on its first meaningful hit.

    Raw hit/onset/bass/high information is untouched; only the repeated
    pause context is removed, then the ranking is recalculated.
    """

    if not events:
        return events, 0

    events = sorted(
        events,
        key=lambda item: item["Time"]
    )

    last_kept_pause_time = None
    suppressed_count = 0

    for event in events:
        is_pause_candidate = (
            event["Pause before"] >= 0.45
            and
            event["Onset"] >= 0.35
        )

        if not is_pause_candidate:
            continue

        if last_kept_pause_time is None:
            last_kept_pause_time = (
                event["Time"]
            )
            continue

        gap = (
            event["Time"]
            -
            last_kept_pause_time
        )

        if gap <= suppression_seconds:
            event["Pause before"] = 0.0
            refresh_event_ranking(
                event
            )
            suppressed_count += 1

        else:
            last_kept_pause_time = (
                event["Time"]
            )

    return (
        events,
        suppressed_count
    )


def calculate_local_prominence(
    events,
    window_seconds=0.65
):
    """
    Measure how prominent each hit is compared with the nearby
    musical activity.

    A value near 1.0 means this event is one of the strongest hits
    in its local neighbourhood. This lets us keep a useful off-grid
    hit without treating every transient as equally important.
    """

    if not events:
        return []

    times = np.asarray(
        [
            event["Time"]
            for event in events
        ],
        dtype=float
    )

    hit_strengths = np.asarray(
        [
            event["Hit strength"]
            for event in events
        ],
        dtype=float
    )

    prominence = []

    for index, event_time in enumerate(times):
        local_mask = (
            np.abs(
                times - event_time
            )
            <=
            window_seconds
        )

        local_max = float(
            np.max(
                hit_strengths[local_mask]
            )
        )

        if local_max <= 0:
            value = 0.0

        else:
            value = float(
                hit_strengths[index]
                /
                local_max
            )

        prominence.append(
            float(
                np.clip(
                    value,
                    0.0,
                    1.0
                )
            )
        )

    return prominence


def build_editorial_candidates(events):
    """
    Turn the sensitive raw event map into a cleaner set of moments
    that an editor is realistically likely to care about.

    Nothing is deleted from the raw analyzer data. This is a second
    layer on top of it. Smart Edit will later make the actual cut
    decisions from these candidates.
    """

    if not events:
        return []

    events = sorted(
        events,
        key=lambda item: item["Time"]
    )

    local_prominence = (
        calculate_local_prominence(
            events,
            window_seconds=0.65
        )
    )

    candidates = []

    for event, prominence in zip(
        events,
        local_prominence
    ):
        context_score = max(
            event["Pause before"],
            event["Build before"]
        )

        distinctive_score = max(
            event["Bass"],
            event["High / tonal"]
        )

        editorial_score = (
            0.42 * event["Hit strength"]
            +
            0.18 * event["Beat alignment"]
            +
            0.18 * prominence
            +
            0.14 * context_score
            +
            0.08 * distinctive_score
        )

        editorial_score = float(
            np.clip(
                editorial_score,
                0.0,
                1.0
            )
        )

        reasons = []

        # First meaningful hit after silence/quiet is a high-value
        # editorial opportunity.
        if (
            event["Pause before"] >= 0.45
            and
            event["Hit strength"] >= 0.40
        ):
            reasons.append(
                "Post-pause"
            )

        # A release at the end of a real sustained build is similarly
        # important.
        if (
            event["Build before"] >= 0.55
            and
            event["Hit strength"] >= 0.40
        ):
            reasons.append(
                "Post-build"
            )

        # Protect distinctive upper-frequency musical notes even when
        # they are not sitting directly on the main beat grid.
        if (
            event["High / tonal"] >= 0.75
            and
            event["Hit strength"] >= 0.50
        ):
            reasons.append(
                "Distinct tonal hit"
            )

        # Big bass/kick events are commonly useful cut points.
        if (
            event["Bass"] >= 0.85
            and
            event["Hit strength"] >= 0.60
        ):
            reasons.append(
                "Strong bass/kick"
            )

        # Loud standalone hits are kept when they are also prominent
        # compared with the surrounding material. This prevents the
        # filter from keeping every single loud transient.
        if (
            event["Onset"] >= 0.78
            and
            event["Hit strength"] >= 0.65
            and
            prominence >= 0.80
        ):
            reasons.append(
                "Prominent strong hit"
            )

        # General editorial score catches well-balanced events that
        # may not trigger one of the special rules above.
        if editorial_score >= 0.64:
            reasons.append(
                "Strong overall candidate"
            )

        if not reasons:
            continue

        candidate = dict(
            event
        )

        candidate["Local prominence"] = (
            prominence
        )

        candidate["Editorial score"] = (
            editorial_score
        )

        candidate["Candidate reason"] = (
            " + ".join(
                reasons
            )
        )

        candidates.append(
            candidate
        )

    return candidates


def detect_song_events(
    analysis,
    sr,
    build_regions
):
    onset = analysis["onset"]
    bass_hit = analysis["bass_hit"]
    high_hit = analysis["high_hit"]
    rms = analysis["rms"]
    beat_times = analysis["beat_times"]

    beat_period = (
        get_median_beat_period(
            beat_times
        )
    )

    peak_frames = (
        librosa.util.peak_pick(
            onset,
            pre_max=2,
            post_max=2,
            pre_avg=5,
            post_avg=5,
            delta=0.035,
            wait=2,
        )
    )

    raw_events = []

    for frame in peak_frames:
        if (
            frame < 0
            or
            frame >= len(onset)
        ):
            continue

        event_time = float(
            librosa.frames_to_time(
                frame,
                sr=sr,
                hop_length=HOP_LENGTH,
            )
        )

        onset_strength = float(
            onset[frame]
        )

        bass_strength = float(
            bass_hit[frame]
        )

        high_strength = float(
            high_hit[frame]
        )

        energy_strength = float(
            rms[frame]
        )

        beat_alignment = (
            calculate_beat_alignment(
                event_time,
                beat_times,
                beat_period,
            )
        )

        pause_score = (
            calculate_pause_before_hit(
                rms,
                frame,
                sr,
            )
        )

        build_context = (
            calculate_build_context(
                event_time,
                build_regions
            )
        )

        distinctive_score = max(
            bass_strength,
            high_strength,
            onset_strength,
        )

        # Loud / strong attacks deliberately retain a large
        # influence. Context can elevate them, but does not
        # replace raw hit strength.
        hit_strength = (
            0.60 * onset_strength
            +
            0.15 * bass_strength
            +
            0.15 * high_strength
            +
            0.10 * energy_strength
        )

        # Keep weaker real candidates for now. They can still
        # become useful in energetic passages later.
        if (
            onset_strength < 0.20
            and
            distinctive_score < 0.25
        ):
            continue

        event = {
            "Time": event_time,
            "Hit strength": hit_strength,
            "Beat alignment": beat_alignment,
            "Pause before": pause_score,
            "Build before": build_context,
            "Bass": bass_strength,
            "High / tonal": high_strength,
            "Onset": onset_strength,
        }

        refresh_event_ranking(
            event
        )

        raw_events.append(
            event
        )

    # Only merge peaks that fit inside one genuinely tiny total
    # window. This prevents chain-merging separate attacks.
    events, merged_count = (
        cluster_micro_hits(
            raw_events,
            cluster_seconds=0.12
        )
    )

    # The look-back pause window can remain elevated for several
    # following attacks. Preserve the pause bonus only on the first
    # meaningful hit in each short pause episode.
    events, suppressed_pause_count = (
        keep_first_post_pause_hit(
            events,
            suppression_seconds=1.75
        )
    )

    return (
        sorted(
            events,
            key=lambda item: item["Time"]
        ),
        len(raw_events),
        merged_count,
        suppressed_pause_count,
    )


def detect_build_regions(analysis, sr):
    """
    Detect sustained build regions rather than treating any
    brief rise in the build curve as a build.

    A region must:
    - remain elevated for long enough
    - show a genuine increase in at least two musical signals
    - reach a reasonably high build score
    """

    build = analysis["build_score"]
    times = analysis["times"]
    rms = analysis["rms"]
    onset_density = analysis["onset_density"]
    brightness = analysis["brightness"]

    frames_per_second = (
        sr
        /
        HOP_LENGTH
    )

    threshold = 0.68
    mask = (
        build >= threshold
    )

    # Fill very short holes so a build is not split by one or
    # two frames dipping under the threshold.
    max_gap_frames = max(
        1,
        int(
            round(
                0.30
                *
                frames_per_second
            )
        )
    )

    active_indices = np.flatnonzero(
        mask
    )

    candidate_regions = []

    if len(active_indices) > 0:
        start = int(
            active_indices[0]
        )
        previous = start

        for index in active_indices[1:]:
            index = int(index)

            if (
                index
                -
                previous
                <=
                max_gap_frames
                +
                1
            ):
                previous = index

            else:
                candidate_regions.append(
                    (
                        start,
                        previous + 1
                    )
                )
                start = index
                previous = index

        candidate_regions.append(
            (
                start,
                previous + 1
            )
        )

    minimum_duration_seconds = 1.50
    minimum_frames = max(
        2,
        int(
            round(
                minimum_duration_seconds
                *
                frames_per_second
            )
        )
    )

    regions = []

    for start, end in candidate_regions:
        if (
            end - start
            <
            minimum_frames
        ):
            continue

        region_length = (
            end - start
        )

        quarter = max(
            2,
            region_length // 4
        )

        early_slice = slice(
            start,
            min(
                start + quarter,
                end
            )
        )

        late_slice = slice(
            max(
                start,
                end - quarter
            ),
            end
        )

        energy_rise = (
            float(
                np.mean(
                    rms[late_slice]
                )
            )
            -
            float(
                np.mean(
                    rms[early_slice]
                )
            )
        )

        density_rise = (
            float(
                np.mean(
                    onset_density[late_slice]
                )
            )
            -
            float(
                np.mean(
                    onset_density[early_slice]
                )
            )
        )

        brightness_rise = (
            float(
                np.mean(
                    brightness[late_slice]
                )
            )
            -
            float(
                np.mean(
                    brightness[early_slice]
                )
            )
        )

        rising_signals = 0

        if energy_rise >= 0.08:
            rising_signals += 1

        if density_rise >= 0.08:
            rising_signals += 1

        if brightness_rise >= 0.06:
            rising_signals += 1

        if rising_signals < 2:
            continue

        segment = build[
            start:end
        ]

        peak_local = int(
            np.argmax(
                segment
            )
        )

        peak_index = (
            start
            +
            peak_local
        )

        # Region score combines the original build peak with
        # evidence that the region genuinely increased.
        rise_bonus = np.clip(
            (
                max(0.0, energy_rise)
                +
                max(0.0, density_rise)
                +
                max(0.0, brightness_rise)
            )
            /
            0.60,
            0.0,
            1.0
        )

        region_score = float(
            np.clip(
                0.75 * build[peak_index]
                +
                0.25 * rise_bonus,
                0.0,
                1.0
            )
        )

        regions.append(
            {
                "Start": float(
                    times[start]
                ),
                "End": float(
                    times[
                        min(
                            end - 1,
                            len(times) - 1
                        )
                    ]
                ),
                "Peak": float(
                    times[peak_index]
                ),
                "Score": region_score,
                "Energy rise": energy_rise,
                "Density rise": density_rise,
                "Brightness rise": brightness_rise,
            }
        )

    return regions


def sanitize_song_name(original_name):
    song_name = Path(
        original_name
    ).stem

    song_name = re.sub(
        r'[<>:"/\\|?*]',
        "",
        song_name
    ).strip()

    if not song_name:
        song_name = "Song"

    return song_name


def show_song_analyzer(
    y,
    sr,
    original_name
):
    analysis = (
        build_song_analysis(
            y,
            sr
        )
    )

    # Builds are identified first. Events are then only labelled
    # post-build if they actually occur at the release/end of one
    # of these regions.
    builds = (
        detect_build_regions(
            analysis,
            sr
        )
    )

    (
        events,
        raw_event_count,
        merged_micro_hits,
        suppressed_pause_hits,
    ) = (
        detect_song_events(
            analysis,
            sr,
            builds
        )
    )

    editorial_candidates = (
        build_editorial_candidates(
            events
        )
    )

    beat_times = analysis[
        "beat_times"
    ]

    if len(beat_times) > 1:
        beat_period = (
            get_median_beat_period(
                beat_times
            )
        )
        tempo = (
            60.0
            /
            beat_period
        )

    else:
        tempo = 0.0

    song_name = (
        sanitize_song_name(
            original_name
        )
    )

    st.success(
        "Song analysis complete."
    )

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )

    with metric1:
        st.metric(
            "Estimated tempo",
            (
                f"{tempo:.1f} BPM"
                if tempo > 0
                else "—"
            ),
        )

    with metric2:
        st.metric(
            "Detected beats",
            len(beat_times)
        )

    with metric3:
        st.metric(
            "Raw events",
            len(events)
        )

    with metric4:
        st.metric(
            "Editorial candidates",
            len(editorial_candidates)
        )

    if merged_micro_hits > 0:
        st.caption(
            f"Cleaned {merged_micro_hits} duplicate micro-hits "
            f"from {raw_event_count} raw onset candidates."
        )

    if suppressed_pause_hits > 0:
        st.caption(
            f"Kept post-pause context on the first meaningful hit and "
            f"removed it from {suppressed_pause_hits} following hits."
        )

    # -----------------------------------------------------
    # ENERGY
    # -----------------------------------------------------

    st.subheader(
        "Energy"
    )

    st.caption(
        "Useful for seeing quiet sections, pauses, "
        "large energy changes and drops."
    )

    energy_dataframe = (
        pd.DataFrame(
            {
                "Time": analysis[
                    "times"
                ],
                "Energy": analysis[
                    "rms"
                ],
            }
        )
        .set_index(
            "Time"
        )
    )

    st.line_chart(
        energy_dataframe,
        height=220
    )

    # -----------------------------------------------------
    # HITS
    # -----------------------------------------------------

    st.subheader(
        "Hits and musical accents"
    )

    st.caption(
        "Overall onset = general attacks. Bass = kick/low-frequency attacks. "
        "High/Tonal = piano, guitar, cymbal, synth and other brighter attacks. "
        "Beat spikes show the detected beat grid."
    )

    hit_dataframe = (
        pd.DataFrame(
            {
                "Time": analysis[
                    "times"
                ],
                "Overall onset": analysis[
                    "onset"
                ],
                "Bass hit": analysis[
                    "bass_hit"
                ],
                "High / tonal hit": analysis[
                    "high_hit"
                ],
                "Beat": analysis[
                    "beat_pulse"
                ],
            }
        )
        .set_index(
            "Time"
        )
    )

    st.line_chart(
        hit_dataframe,
        height=300
    )

    # -----------------------------------------------------
    # BUILD ANALYSIS
    # -----------------------------------------------------

    st.subheader(
        "Build-up analysis"
    )

    st.caption(
        "The curve shows possible rising musical activity. "
        "A region is only labelled as a build when the rise is "
        "sustained and several signals increase together."
    )

    build_dataframe = (
        pd.DataFrame(
            {
                "Time": analysis[
                    "times"
                ],
                "Build score": analysis[
                    "build_score"
                ],
                "Onset density": analysis[
                    "onset_density"
                ],
                "Brightness": analysis[
                    "brightness"
                ],
            }
        )
        .set_index(
            "Time"
        )
    )

    st.line_chart(
        build_dataframe,
        height=260
    )

    # -----------------------------------------------------
    # POSSIBLE BUILDS
    # -----------------------------------------------------

    st.subheader(
        "Possible builds"
    )

    if len(builds) == 0:
        st.write(
            "No clear sustained build regions detected."
        )

    else:
        build_rows = []

        for build in builds:
            build_rows.append(
                {
                    "Start": format_time(
                        build["Start"]
                    ),
                    "End": format_time(
                        build["End"]
                    ),
                    "Peak": format_time(
                        build["Peak"]
                    ),
                    "Score": round(
                        build["Score"],
                        2
                    ),
                    "Energy rise": round(
                        build["Energy rise"],
                        2
                    ),
                    "Activity rise": round(
                        build["Density rise"],
                        2
                    ),
                    "Brightness rise": round(
                        build["Brightness rise"],
                        2
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(
                build_rows
            ),
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # EDITORIAL CANDIDATES
    # -----------------------------------------------------

    st.subheader(
        "Editorial candidates"
    )

    st.caption(
        "The raw analyzer remains sensitive, but this second layer keeps the moments "
        "most likely to matter in an edit: strong/prominent hits, post-pause and post-build "
        "events, strong bass/kick hits, distinctive tonal notes and other well-balanced candidates. "
        "These are still opportunities, not automatic Smart Edit cuts."
    )

    candidate_rows = []

    for event in editorial_candidates:
        candidate_rows.append(
            {
                "Song": song_name,
                "Time": format_time(
                    event["Time"]
                ),
                "Seconds": round(
                    event["Time"],
                    3
                ),
                "Tier": event[
                    "Tier"
                ],
                "Event": event[
                    "Event"
                ],
                "Candidate reason": event[
                    "Candidate reason"
                ],
                "Editorial score": round(
                    event["Editorial score"],
                    2
                ),
                "Opportunity": round(
                    event["Opportunity"],
                    2
                ),
                "Hit": round(
                    event["Hit strength"],
                    2
                ),
                "Prominence": round(
                    event["Local prominence"],
                    2
                ),
                "Beat": round(
                    event["Beat alignment"],
                    2
                ),
                "Pause": round(
                    event["Pause before"],
                    2
                ),
                "Build": round(
                    event["Build before"],
                    2
                ),
                "Bass": round(
                    event["Bass"],
                    2
                ),
                "High": round(
                    event["High / tonal"],
                    2
                ),
            }
        )

    candidate_dataframe = (
        pd.DataFrame(
            candidate_rows
        )
    )

    st.dataframe(
        candidate_dataframe,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    if not candidate_dataframe.empty:
        candidate_csv_bytes = (
            candidate_dataframe
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )

        candidate_csv_filename = (
            f"{song_name}_"
            f"DetectTheBeat_"
            f"SongAnalysis.csv"
        )

        st.download_button(
            "Download analysis CSV",
            data=candidate_csv_bytes,
            file_name=(
                candidate_csv_filename
            ),
            mime="text/csv",
        )

    # -----------------------------------------------------
    # RAW EVENT MAP
    # -----------------------------------------------------

    with st.expander(
        f"Raw event map ({len(events)} events)"
    ):
        st.caption(
            "This is the sensitive underlying event map. Nothing here is deleted from the "
            "analysis; the Editorial Candidate layer simply selects the more useful subset."
        )

        raw_rows = []

        for event in events:
            raw_rows.append(
                {
                    "Song": song_name,
                    "Time": format_time(
                        event["Time"]
                    ),
                    "Seconds": round(
                        event["Time"],
                        3
                    ),
                    "Tier": event[
                        "Tier"
                    ],
                    "Event": event[
                        "Event"
                    ],
                    "Opportunity": round(
                        event["Opportunity"],
                        2
                    ),
                    "Hit": round(
                        event["Hit strength"],
                        2
                    ),
                    "Beat": round(
                        event["Beat alignment"],
                        2
                    ),
                    "Pause": round(
                        event["Pause before"],
                        2
                    ),
                    "Build": round(
                        event["Build before"],
                        2
                    ),
                    "Bass": round(
                        event["Bass"],
                        2
                    ),
                    "High": round(
                        event["High / tonal"],
                        2
                    ),
                }
            )

        raw_dataframe = (
            pd.DataFrame(
                raw_rows
            )
        )

        st.dataframe(
            raw_dataframe,
            use_container_width=True,
            hide_index=True,
            height=400,
        )

        if not raw_dataframe.empty:
            raw_csv_bytes = (
                raw_dataframe
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            )

            raw_csv_filename = (
                f"{song_name}_"
                f"DetectTheBeat_"
                f"RawEvents.csv"
            )

            st.download_button(
                "Download raw events CSV",
                data=raw_csv_bytes,
                file_name=(
                    raw_csv_filename
                ),
                mime="text/csv",
            )


# =========================================================
# MODE
# =========================================================

mode = st.radio(
    "Mode",
    ["Beat", "Song Analyzer"],
    horizontal=True,
)


# =========================================================
# UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload audio",
    type=["mp3", "wav", "m4a"],
)

st.caption("MP3, WAV or M4A · Maximum length: 6 minutes")

if uploaded_file is not None:
    st.audio(uploaded_file)


# =========================================================
# SONG ANALYZER MODE
# =========================================================

if mode == "Song Analyzer":
    st.write(
        "Song Analyzer finds beats, hits, pauses, build-ups and distinctive musical accents. "
        "It does not choose Smart Edit cuts yet."
    )

    if uploaded_file is not None:
        if st.button("🔍 Analyze Song", type="primary"):
            status = None

            try:
                status = st.status(
                    "Analyzing song...",
                    expanded=True,
                )

                with tempfile.TemporaryDirectory() as work_dir:
                    work_dir = Path(work_dir)

                    status.write("1/4 · Preparing audio")
                    audio_path = save_uploaded_audio(
                        uploaded_file,
                        work_dir,
                    )

                    duration = get_audio_duration(str(audio_path))

                    if duration > MAX_AUDIO_DURATION:
                        status.update(
                            label="Audio is too long",
                            state="error",
                        )
                        st.error(
                            "Please upload a track of 6 minutes or less."
                        )
                        st.stop()

                    analysis_path = create_analysis_wav(
                        audio_path,
                        work_dir,
                    )

                    status.write("2/4 · Reading waveform")
                    y, sr = librosa.load(
                        str(analysis_path),
                        sr=None,
                        mono=True,
                    )

                    status.write("3/4 · Detecting musical structure")
                    status.write("4/4 · Finding editing opportunities")

                    # Run the analysis while the temporary audio still exists.
                    analysis_output = (y, sr)

                status.update(
                    label="Analysis ready",
                    state="complete",
                    expanded=False,
                )

                show_song_analyzer(
                    analysis_output[0],
                    analysis_output[1],
                    uploaded_file.name,
                )

            except Exception as error:
                if status is not None:
                    try:
                        status.update(
                            label="Song analysis failed",
                            state="error",
                        )
                    except Exception:
                        pass

                st.error("Song analysis failed.")
                st.code(str(error))


# =========================================================
# BASIC BEAT MODE
# =========================================================

else:
    st.write(
        "Beat mode creates a checkerboard reference video for Scene Edit Detection."
    )

    st.subheader("Video settings")
    st.caption("Output: 16:9 · 1920×1080")

    fps_choice = st.radio(
        "Frame rate",
        options=["25 fps", "23.976 fps"],
        horizontal=True,
    )

    beat_choice = st.radio(
        "Scene change",
        options=[
            "Every beat",
            "Every 2 beats",
            "Every 4 beats",
        ],
        index=2,
    )

    FPS_VALUE = FPS_OPTIONS[fps_choice]["value"]
    FPS_FFMPEG = FPS_OPTIONS[fps_choice]["ffmpeg"]
    BEAT_INTERVAL = BEAT_INTERVALS[beat_choice]

    if uploaded_file is not None:
        download_filename = make_output_filename(
            uploaded_file.name,
            beat_choice,
        )

        if st.button("🚀 Generate Video", type="primary"):
            status = None

            try:
                status = st.status(
                    "Preparing audio...",
                    expanded=True,
                )

                with tempfile.TemporaryDirectory() as work_dir:
                    work_dir = Path(work_dir)

                    # ==========================================
                    # STEP 1
                    # ==========================================

                    status.write("1/5 · Checking audio")

                    audio_path = save_uploaded_audio(
                        uploaded_file,
                        work_dir,
                    )

                    total_duration = get_audio_duration(
                        str(audio_path)
                    )

                    if total_duration > MAX_AUDIO_DURATION:
                        status.update(
                            label="Audio is too long",
                            state="error",
                        )
                        st.error(
                            "Please upload a track of 6 minutes or less."
                        )
                        st.stop()

                    minutes = int(total_duration // 60)
                    seconds = int(total_duration % 60)

                    status.write(
                        f"Audio length: {minutes}:{seconds:02d}"
                    )

                    # ==========================================
                    # STEP 2
                    # ==========================================

                    status.write(
                        "2/5 · Preparing audio for beat detection"
                    )

                    analysis_path = create_analysis_wav(
                        audio_path,
                        work_dir,
                    )

                    # ==========================================
                    # STEP 3
                    # ==========================================

                    status.write(
                        "3/5 · Detecting rhythm and strong musical accents"
                    )

                    y, sr = librosa.load(
                        str(analysis_path),
                        sr=None,
                        mono=True,
                    )

                    onset_envelope = librosa.onset.onset_strength(
                        y=y,
                        sr=sr,
                        hop_length=HOP_LENGTH,
                    )

                    _, beat_frames = librosa.beat.beat_track(
                        onset_envelope=onset_envelope,
                        sr=sr,
                        hop_length=HOP_LENGTH,
                    )

                    beat_times = librosa.frames_to_time(
                        beat_frames,
                        sr=sr,
                        hop_length=HOP_LENGTH,
                    )

                    if len(beat_times) == 0:
                        raise RuntimeError(
                            "No reliable beats were detected in this track."
                        )

                    accent_curve = build_accent_curve(
                        y=y,
                        sr=sr,
                        onset_envelope=onset_envelope,
                    )

                    selected_beats, phrase_states = select_phrase_locked_beats(
                        beat_times=beat_times,
                        accent_curve=accent_curve,
                        sr=sr,
                        interval=BEAT_INTERVAL,
                    )

                    status.write(
                        f"Detected {len(beat_times)} base beats"
                    )

                    if BEAT_INTERVAL == 1:
                        status.write("Using every detected beat")
                    else:
                        status.write(
                            f"Selected {len(selected_beats)} phrase-locked edit points"
                        )

                        if len(phrase_states) > 0:
                            initial_offset = phrase_states[0]["offset"]
                            offset_ms = int(round(initial_offset * 1000))

                            status.write(
                                f"Opening rhythm alignment: {offset_ms:+d} ms"
                            )

                            number_of_changes = 0

                            for state_index in range(
                                1,
                                len(phrase_states),
                            ):
                                current_state = phrase_states[state_index]
                                previous_state = phrase_states[state_index - 1]

                                if (
                                    current_state["phase"]
                                    != previous_state["phase"]
                                    or abs(
                                        current_state["offset"]
                                        - previous_state["offset"]
                                    )
                                    > 0.01
                                ):
                                    number_of_changes += 1

                            status.write(
                                f"Rhythm alignment changes: {number_of_changes}"
                            )

                    del y
                    del onset_envelope
                    del accent_curve

                    # ==========================================
                    # FRAME CONVERSION
                    # ==========================================

                    total_video_frames = math.ceil(
                        total_duration * FPS_VALUE
                    )

                    scene_change_frames = []

                    for beat in selected_beats:
                        frame_number = beat_time_to_frame(
                            float(beat),
                            FPS_VALUE,
                        )

                        if 0 < frame_number < total_video_frames:
                            scene_change_frames.append(frame_number)

                    scene_change_frames = sorted(
                        set(scene_change_frames)
                    )

                    status.write(
                        f"Creating {len(scene_change_frames)} scene changes"
                    )

                    # ==========================================
                    # STEP 4
                    # ==========================================

                    status.write(
                        "4/5 · Building frame-accurate video"
                    )

                    raw_video_path = work_dir / "reference.rgb"

                    change_index = 0
                    pattern_index = -1

                    with open(raw_video_path, "wb") as raw_video:
                        for frame_number in range(total_video_frames):
                            while (
                                change_index < len(scene_change_frames)
                                and frame_number
                                >= scene_change_frames[change_index]
                            ):
                                pattern_index += 1
                                change_index += 1

                            if pattern_index < 0:
                                frame_data = BLACK_FRAME
                            elif pattern_index % 2 == 0:
                                frame_data = PATTERN_A_FRAME
                            else:
                                frame_data = PATTERN_B_FRAME

                            raw_video.write(frame_data)

                    # ==========================================
                    # STEP 5
                    # ==========================================

                    status.write("5/5 · Rendering video")

                    output_path = (
                        work_dir / "detectthebeat_video.mp4"
                    )

                    run_command(
                        [
                            FFMPEG_EXE,
                            "-y",
                            "-loglevel",
                            "error",
                            "-f",
                            "rawvideo",
                            "-pix_fmt",
                            "rgb24",
                            "-s:v",
                            f"{INTERNAL_WIDTH}x{INTERNAL_HEIGHT}",
                            "-r",
                            FPS_FFMPEG,
                            "-i",
                            str(raw_video_path),
                            "-i",
                            str(audio_path),
                            "-map",
                            "0:v:0",
                            "-map",
                            "1:a:0",
                            "-vf",
                            (
                                f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:"
                                "flags=neighbor,format=yuv420p"
                            ),
                            "-c:v",
                            "libx264",
                            "-preset",
                            "ultrafast",
                            "-crf",
                            "18",
                            "-c:a",
                            "aac",
                            "-b:a",
                            "192k",
                            "-shortest",
                            "-movflags",
                            "+faststart",
                            str(output_path),
                        ]
                    )

                    video_bytes = output_path.read_bytes()

                status.update(
                    label="Video ready!",
                    state="complete",
                    expanded=False,
                )

                st.success(
                    f"Created {len(scene_change_frames)} scene changes."
                )

                st.video(video_bytes)

                st.download_button(
                    label="📥 Download Video",
                    data=video_bytes,
                    file_name=download_filename,
                    mime="video/mp4",
                )

                st.info(
                    "Import the MP4 into your editing software and use Scene Edit Detection "
                    "to create cuts at the checkerboard changes. Use your original audio file "
                    "for the final edit."
                )

            except Exception as error:
                if status is not None:
                    try:
                        status.update(
                            label="Something went wrong",
                            state="error",
                        )
                    except Exception:
                        pass

                st.error("Video generation failed.")
                st.code(str(error))
