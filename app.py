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

st.set_page_config(
    page_title="DetectTheBeat",
    page_icon="🎵",
    layout="centered",
)

st.title("🎵 DetectTheBeat")
st.write("Turn music into useful editing information.")

MAX_AUDIO_DURATION = 6 * 60
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
INTERNAL_WIDTH = 64
INTERNAL_HEIGHT = 36
HOP_LENGTH = 512
PHRASE_BEATS = 16
SWITCH_MARGIN = 0.1

FPS_OPTIONS = {
    "25 fps": {
        "value": 25.0,
        "ffmpeg": "25",
    },
    "23.976 fps": {
        "value": 24000 / 1001,
        "ffmpeg": "24000/1001",
    },
}

BEAT_INTERVALS = {
    "Every beat": 1,
    "Every 2 beats": 2,
    "Every 4 beats": 4,
}


def run_command(
    command,
    cwd=None,
):
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg error:\n"
            f"{result.stderr[-5000:]}"
        )

    return result


def get_audio_duration(
    audio_path,
):
    return float(
        librosa.get_duration(
            path=audio_path,
        )
    )


def beat_time_to_frame(
    time_seconds,
    fps,
):
    return round(
        time_seconds * fps
    )


def format_time(
    seconds,
):
    minutes = int(
        seconds // 60
    )

    remaining = (
        seconds
        - minutes * 60
    )

    return (
        f"{minutes}:"
        f"{remaining:05.2f}"
    )


def save_uploaded_audio(
    uploaded_file,
    work_dir,
):
    extension = Path(
        uploaded_file.name
    ).suffix.lower()

    if extension not in [
        ".mp3",
        ".wav",
        ".m4a",
    ]:
        extension = ".mp3"

    audio_path = (
        work_dir
        / f"input{extension}"
    )

    with open(
        audio_path,
        "wb",
    ) as file:
        file.write(
            uploaded_file.getvalue()
        )

    return audio_path


def create_analysis_wav(
    audio_path,
    work_dir,
):
    analysis_path = (
        work_dir
        / "analysis.wav"
    )

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


def make_output_filename(
    original_name,
    beat_choice,
):
    song_name = Path(
        original_name
    ).stem

    song_name = re.sub(
        r'[<>:"/\\|?*]',
        "",
        song_name,
    ).strip()

    if not song_name:
        song_name = "Song"

    beat_labels = {
        "Every beat": "EveryBeat",
        "Every 2 beats": "Every2Beats",
        "Every 4 beats": "Every4Beats",
    }

    return (
        f"{song_name}_"
        f"DetectTheBeat_"
        f"{beat_labels[beat_choice]}.mp4"
    )


def create_solid_frame(
    width,
    height,
    color,
):
    return (
        bytes(color)
        * (
            width
            * height
        )
    )


def create_checkerboard_frame(
    width,
    height,
    inverted=False,
):
    block_size = 4
    pixels = bytearray()

    for y in range(height):
        for x in range(width):
            checker = (
                (
                    x // block_size
                )
                + (
                    y // block_size
                )
            ) % 2

            if inverted:
                checker = (
                    1 - checker
                )

            if checker == 0:
                pixels.extend(
                    (
                        0,
                        0,
                        0,
                    )
                )
            else:
                pixels.extend(
                    (
                        255,
                        255,
                        255,
                    )
                )

    return bytes(
        pixels
    )


BLACK_FRAME = (
    create_solid_frame(
        INTERNAL_WIDTH,
        INTERNAL_HEIGHT,
        (
            0,
            0,
            0,
        ),
    )
)

PATTERN_A_FRAME = (
    create_checkerboard_frame(
        INTERNAL_WIDTH,
        INTERNAL_HEIGHT,
        inverted=False,
    )
)

PATTERN_B_FRAME = (
    create_checkerboard_frame(
        INTERNAL_WIDTH,
        INTERNAL_HEIGHT,
        inverted=True,
    )
)


def normalize_feature(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return values

    low = np.percentile(
        values,
        10,
    )

    high = np.percentile(
        values,
        90,
    )

    if high <= low:
        return np.zeros_like(
            values
        )

    normalized = (
        values - low
    ) / (
        high - low
    )

    return np.clip(
        normalized,
        0.0,
        1.0,
    )


def match_feature_length(
    values,
    target_length,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == target_length:
        return values

    if len(values) > target_length:
        return values[
            :target_length
        ]

    if len(values) == 0:
        return np.zeros(
            target_length
        )

    padding = np.full(
        target_length
        - len(values),
        values[-1],
    )

    return np.concatenate(
        [
            values,
            padding,
        ]
    )


def sample_curve_at_time(
    curve,
    time_seconds,
    sr,
    radius=1,
):
    frame = int(
        round(
            time_seconds
            * sr
            / HOP_LENGTH
        )
    )

    start = max(
        0,
        frame - radius,
    )

    end = min(
        len(curve),
        frame
        + radius
        + 1,
    )

    if end <= start:
        return 0.0

    return float(
        np.max(
            curve[
                start:end
            ]
        )
    )


def moving_average(
    values,
    window,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    if window <= 1:
        return values.copy()

    kernel = (
        np.ones(window)
        / window
    )

    return np.convolve(
        values,
        kernel,
        mode="same",
    )


def rising_trend(
    values,
    window_frames,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return values

    half = max(
        2,
        window_frames // 2,
    )

    smoothed = moving_average(
        values,
        max(
            2,
            half // 4,
        ),
    )

    recent = moving_average(
        smoothed,
        half,
    )

    previous = np.roll(
        recent,
        half,
    )

    trend = (
        recent - previous
    )

    trend[:half] = 0

    trend = np.maximum(
        trend,
        0,
    )

    return normalize_feature(
        trend
    )


def build_accent_curve(
    y,
    sr,
    onset_envelope,
):
    onset_curve = normalize_feature(
        onset_envelope
    )

    mel = (
        librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=1024,
            hop_length=HOP_LENGTH,
            n_mels=32,
            fmin=30,
            fmax=1000,
            power=1.0,
        )
    )

    mel_frequencies = (
        librosa.mel_frequencies(
            n_mels=32,
            fmin=30,
            fmax=1000,
        )
    )

    bass_mask = (
        mel_frequencies
        <= 180
    )

    if np.any(
        bass_mask
    ):
        bass_curve = np.mean(
            mel[
                bass_mask,
                :
            ],
            axis=0,
        )
    else:
        bass_curve = np.zeros(
            mel.shape[1]
        )

    del mel

    bass_curve = normalize_feature(
        bass_curve
    )

    rms_curve = (
        librosa.feature.rms(
            y=y,
            frame_length=1024,
            hop_length=HOP_LENGTH,
        )[0]
    )

    rms_curve = normalize_feature(
        rms_curve
    )

    target_length = len(
        onset_curve
    )

    bass_curve = (
        match_feature_length(
            bass_curve,
            target_length,
        )
    )

    rms_curve = (
        match_feature_length(
            rms_curve,
            target_length,
        )
    )

    accent_curve = (
        0.55 * onset_curve
        + 0.30 * bass_curve
        + 0.15 * rms_curve
    )

    return normalize_feature(
        accent_curve
    )


def get_median_beat_period(
    beat_times,
):
    if len(beat_times) < 2:
        return 0.5

    differences = np.diff(
        beat_times
    )

    differences = (
        differences[
            differences > 0
        ]
    )

    if len(differences) == 0:
        return 0.5

    return float(
        np.median(
            differences
        )
    )


def create_offset_candidates(
    beat_period,
    sr,
):
    analysis_step = (
        HOP_LENGTH / sr
    )

    maximum_shift = (
        beat_period * 0.55
    )

    offsets = np.arange(
        -maximum_shift,
        maximum_shift
        + analysis_step / 2,
        analysis_step,
    )

    offsets = np.append(
        offsets,
        0.0,
    )

    return np.unique(
        np.round(
            offsets,
            6,
        )
    )


def get_phrase_selected_indices(
    block_start,
    block_end,
    phase,
    interval,
):
    selected = []

    for beat_index in range(
        block_start,
        block_end,
    ):
        local_index = (
            beat_index
            - block_start
        )

        if (
            local_index
            % interval
            == phase
        ):
            selected.append(
                beat_index
            )

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
    selected_indices = (
        get_phrase_selected_indices(
            block_start,
            block_end,
            phase,
            interval,
        )
    )

    strengths = []

    for beat_index in selected_indices:
        if (
            beat_index
            < block_start
            + ignore_first_beats
        ):
            continue

        shifted_time = (
            float(
                beat_times[
                    beat_index
                ]
            )
            + offset
        )

        if shifted_time < 0:
            continue

        strength = (
            sample_curve_at_time(
                accent_curve,
                shifted_time,
                sr,
                radius=1,
            )
        )

        strengths.append(
            strength
        )

    if len(strengths) == 0:
        return -999.0

    strengths = np.asarray(
        strengths
    )

    median_strength = float(
        np.median(
            strengths
        )
    )

    mean_strength = float(
        np.mean(
            strengths
        )
    )

    lower_strength = float(
        np.percentile(
            strengths,
            25,
        )
    )

    score = (
        0.50 * median_strength
        + 0.25 * mean_strength
        + 0.25 * lower_strength
    )

    beat_period = (
        get_median_beat_period(
            beat_times
        )
    )

    if beat_period > 0:
        score -= (
            0.025
            * abs(offset)
            / beat_period
        )

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

    for phase in range(
        interval
    ):
        for offset in (
            offset_candidates
        ):
            score = (
                score_phrase_state(
                    beat_times=beat_times,
                    accent_curve=accent_curve,
                    sr=sr,
                    block_start=block_start,
                    block_end=block_end,
                    interval=interval,
                    phase=phase,
                    offset=float(
                        offset
                    ),
                    ignore_first_beats=ignore_first_beats,
                )
            )

            if score > best_score:
                best_score = score
                best_phase = phase
                best_offset = float(
                    offset
                )

    return (
        best_phase,
        best_offset,
        best_score,
    )


def select_phrase_locked_beats(
    beat_times,
    accent_curve,
    sr,
    interval,
):
    beat_count = len(
        beat_times
    )

    if beat_count == 0:
        return (
            [],
            [],
        )

    if interval == 1:
        return (
            [
                float(x)
                for x
                in beat_times
            ],
            [],
        )

    beat_period = (
        get_median_beat_period(
            beat_times
        )
    )

    offset_candidates = (
        create_offset_candidates(
            beat_period,
            sr,
        )
    )

    selected_records = []
    phrase_states = []

    previous_phase = None
    previous_offset = None
    block_number = 0

    for block_start in range(
        0,
        beat_count,
        PHRASE_BEATS,
    ):
        block_end = min(
            beat_count,
            block_start
            + PHRASE_BEATS,
        )

        if block_number == 0:
            ignore_first_beats = min(
                2,
                max(
                    0,
                    block_end
                    - block_start
                    - 1,
                ),
            )

            (
                chosen_phase,
                chosen_offset,
                chosen_score,
            ) = find_best_phrase_state(
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
            (
                best_phase,
                best_offset,
                best_score,
            ) = find_best_phrase_state(
                beat_times=beat_times,
                accent_curve=accent_curve,
                sr=sr,
                block_start=block_start,
                block_end=block_end,
                interval=interval,
                offset_candidates=offset_candidates,
                ignore_first_beats=0,
            )

            previous_score = (
                score_phrase_state(
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
            )

            offset_change = abs(
                best_offset
                - previous_offset
            )

            offset_change_penalty = (
                0.05
                * offset_change
                / max(
                    beat_period,
                    0.001,
                )
            )

            phase_change_penalty = 0.0

            if (
                best_phase
                != previous_phase
            ):
                phase_change_penalty = 0.04

            required_improvement = (
                SWITCH_MARGIN
                + offset_change_penalty
                + phase_change_penalty
            )

            if (
                best_score
                > previous_score
                + required_improvement
            ):
                chosen_phase = best_phase
                chosen_offset = best_offset
                chosen_score = best_score

            else:
                chosen_phase = previous_phase
                chosen_offset = previous_offset
                chosen_score = previous_score

        phrase_indices = (
            get_phrase_selected_indices(
                block_start,
                block_end,
                chosen_phase,
                interval,
            )
        )

        for beat_index in phrase_indices:
            shifted_time = (
                float(
                    beat_times[
                        beat_index
                    ]
                )
                + chosen_offset
            )

            if shifted_time <= 0:
                continue

            strength = (
                sample_curve_at_time(
                    accent_curve,
                    shifted_time,
                    sr,
                    radius=1,
                )
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
        key=lambda item: item[
            "time"
        ],
    )

    if len(selected_records) == 0:
        return (
            [],
            phrase_states,
        )

    target_gap = (
        beat_period
        * interval
    )

    minimum_gap = (
        target_gap
        * 0.55
    )

    cleaned = []

    for record in selected_records:
        if not cleaned:
            cleaned.append(
                record
            )
            continue

        previous = cleaned[-1]

        gap = (
            record["time"]
            - previous["time"]
        )

        if gap < minimum_gap:
            if (
                record["strength"]
                > previous["strength"]
            ):
                cleaned[-1] = record
        else:
            cleaned.append(
                record
            )

    selected_times = [
        item["time"]
        for item in cleaned
    ]

    return (
        selected_times,
        phrase_states,
    )


def calculate_band_activity(
    y,
    sr,
):
    n_fft = 2048

    magnitude = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=HOP_LENGTH,
        )
    )

    frequencies = (
        librosa.fft_frequencies(
            sr=sr,
            n_fft=n_fft,
        )
    )

    low_mask = (
        (
            frequencies >= 30
        )
        & (
            frequencies <= 220
        )
    )

    high_mask = (
        (
            frequencies >= 600
        )
        & (
            frequencies <= 8000
        )
    )

    low_energy = np.mean(
        magnitude[
            low_mask,
            :
        ],
        axis=0,
    )

    high_energy = np.mean(
        magnitude[
            high_mask,
            :
        ],
        axis=0,
    )

    low_log = np.log1p(
        low_energy
    )

    high_log = np.log1p(
        high_energy
    )

    low_change = np.maximum(
        np.diff(
            low_log,
            prepend=low_log[0],
        ),
        0,
    )

    high_change = np.maximum(
        np.diff(
            high_log,
            prepend=high_log[0],
        ),
        0,
    )

    return (
        normalize_feature(
            low_change
        ),
        normalize_feature(
            high_change
        ),
    )


def robust_standardize_matrix(
    matrix,
):
    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    if (
        matrix.ndim != 2
        or matrix.size == 0
    ):
        return matrix

    center = np.median(
        matrix,
        axis=1,
        keepdims=True,
    )

    low = np.percentile(
        matrix,
        10,
        axis=1,
        keepdims=True,
    )

    high = np.percentile(
        matrix,
        90,
        axis=1,
        keepdims=True,
    )

    scale = np.maximum(
        high - low,
        1e-6,
    )

    standardized = (
        matrix - center
    ) / scale

    return np.clip(
        standardized,
        -3.0,
        3.0,
    )


def build_song_analysis(
    y,
    sr,
):
    onset_envelope = (
        librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=HOP_LENGTH,
        )
    )

    onset = normalize_feature(
        onset_envelope
    )

    target_length = len(
        onset
    )

    rms_raw = (
        librosa.feature.rms(
            y=y,
            frame_length=2048,
            hop_length=HOP_LENGTH,
        )[0]
    )

    rms_raw = (
        match_feature_length(
            rms_raw,
            target_length,
        )
    )

    rms = normalize_feature(
        rms_raw
    )

    brightness_raw = (
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=HOP_LENGTH,
        )[0]
    )

    brightness_raw = (
        match_feature_length(
            brightness_raw,
            target_length,
        )
    )

    brightness = normalize_feature(
        brightness_raw
    )

    (
        bass_hit,
        high_hit,
    ) = calculate_band_activity(
        y,
        sr,
    )

    bass_hit = (
        match_feature_length(
            bass_hit,
            target_length,
        )
    )

    high_hit = (
        match_feature_length(
            high_hit,
            target_length,
        )
    )

    chroma = (
        librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=HOP_LENGTH,
        )
    )

    if chroma.shape[1] != target_length:
        if chroma.shape[1] > target_length:
            chroma = (
                chroma[
                    :,
                    :target_length
                ]
            )
        else:
            pad = (
                target_length
                - chroma.shape[1]
            )

            chroma = np.pad(
                chroma,
                (
                    (
                        0,
                        0,
                    ),
                    (
                        0,
                        pad,
                    ),
                ),
                mode="edge",
            )

    mfcc = (
        librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13,
            n_fft=2048,
            hop_length=HOP_LENGTH,
        )
    )

    if mfcc.shape[1] != target_length:
        if mfcc.shape[1] > target_length:
            mfcc = (
                mfcc[
                    :,
                    :target_length
                ]
            )
        else:
            pad = (
                target_length
                - mfcc.shape[1]
            )

            mfcc = np.pad(
                mfcc,
                (
                    (
                        0,
                        0,
                    ),
                    (
                        0,
                        pad,
                    ),
                ),
                mode="edge",
            )

    mfcc = (
        robust_standardize_matrix(
            mfcc
        )
    )

    (
        _,
        beat_frames,
    ) = (
        librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sr,
            hop_length=HOP_LENGTH,
        )
    )

    beat_frames = np.asarray(
        beat_frames,
        dtype=int,
    )

    beat_times = (
        librosa.frames_to_time(
            beat_frames,
            sr=sr,
            hop_length=HOP_LENGTH,
        )
    )

    beat_pulse = np.zeros(
        target_length
    )

    for frame in beat_frames:
        if (
            0
            <= frame
            < target_length
        ):
            beat_pulse[
                frame
            ] = 1.0

    frames_per_second = (
        sr / HOP_LENGTH
    )

    density_window = max(
        2,
        int(
            round(
                2.0
                * frames_per_second
            )
        ),
    )

    onset_density = (
        normalize_feature(
            moving_average(
                onset,
                density_window,
            )
        )
    )

    build_window = max(
        4,
        int(
            round(
                4.0
                * frames_per_second
            )
        ),
    )

    energy_rise = (
        rising_trend(
            rms,
            build_window,
        )
    )

    brightness_rise = (
        rising_trend(
            brightness,
            build_window,
        )
    )

    density_rise = (
        rising_trend(
            onset_density,
            build_window,
        )
    )

    percussive_activity = (
        moving_average(
            np.maximum(
                onset,
                bass_hit,
            ),
            max(
                2,
                int(
                    frames_per_second
                ),
            ),
        )
    )

    percussive_activity = (
        normalize_feature(
            percussive_activity
        )
    )

    build_score = (
        0.40
        * energy_rise
        + 0.22
        * brightness_rise
        + 0.25
        * density_rise
        + 0.13
        * percussive_activity
    )

    build_score = (
        normalize_feature(
            build_score
        )
    )

    times = (
        librosa.frames_to_time(
            np.arange(
                target_length
            ),
            sr=sr,
            hop_length=HOP_LENGTH,
        )
    )

    return {
        "times": times,
        "onset": onset,
        "onset_raw": onset_envelope,
        "rms": rms,
        "rms_raw": rms_raw,
        "brightness": brightness,
        "brightness_raw": brightness_raw,
        "bass_hit": bass_hit,
        "high_hit": high_hit,
        "onset_density": onset_density,
        "build_score": build_score,
        "beat_times": beat_times,
        "beat_pulse": beat_pulse,
        "chroma": chroma,
        "mfcc": mfcc,
    }


def calculate_beat_alignment(
    event_time,
    beat_times,
    beat_period,
):
    if len(beat_times) == 0:
        return 0.0

    distance = float(
        np.min(
            np.abs(
                beat_times
                - event_time
            )
        )
    )

    tolerance = max(
        0.08,
        beat_period * 0.45,
    )

    alignment = (
        1.0
        - distance
        / tolerance
    )

    return float(
        np.clip(
            alignment,
            0.0,
            1.0,
        )
    )


def calculate_pause_before_hit(
    rms,
    frame,
    sr,
):
    fps = (
        sr / HOP_LENGTH
    )

    short_window = int(
        0.7 * fps
    )

    context_window = int(
        2.5 * fps
    )

    pause_start = max(
        0,
        frame
        - short_window,
    )

    context_start = max(
        0,
        pause_start
        - context_window,
    )

    pre_values = rms[
        pause_start:
        max(
            pause_start + 1,
            frame,
        )
    ]

    pre_energy = (
        float(
            np.mean(
                pre_values
            )
        )
        if len(pre_values)
        else 0.0
    )

    context_values = rms[
        context_start:
        max(
            context_start + 1,
            pause_start,
        )
    ]

    if len(context_values) == 0:
        return 0.0

    context_energy = float(
        np.mean(
            context_values
        )
    )

    energy_drop = (
        context_energy
        - pre_energy
    )

    pause_score = (
        energy_drop
        * 3.0
    )

    if pre_energy < 0.20:
        pause_score += 0.20

    return float(
        np.clip(
            pause_score,
            0.0,
            1.0,
        )
    )


def calculate_build_context(
    event_time,
    build_regions,
):
    best_score = 0.0

    for build in (
        build_regions
    ):
        delta = (
            event_time
            - build["End"]
        )

        if (
            -0.12
            <= delta
            <= 0.80
        ):
            if delta <= 0:
                proximity = 1.0
            else:
                proximity = (
                    1.0
                    - delta
                    / 0.80
                )

            contextual_score = (
                build["Score"]
                * max(
                    0.0,
                    proximity,
                )
            )

            best_score = max(
                best_score,
                contextual_score,
            )

    return float(
        np.clip(
            best_score,
            0.0,
            1.0,
        )
    )


def get_event_tier(
    opportunity,
):
    if opportunity >= 0.85:
        return "PRIMARY"

    if opportunity >= 0.70:
        return "STRONG"

    if opportunity >= 0.50:
        return "SECONDARY"

    return "LOW"


def calculate_event_opportunity(
    event,
):
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
        0.40
        * event[
            "Hit strength"
        ]
        + 0.20
        * event[
            "Beat alignment"
        ]
        + 0.25
        * context_score
        + 0.15
        * distinctive_score
    )

    if (
        event["High / tonal"]
        >= 0.75
        and event[
            "Hit strength"
        ]
        >= 0.45
    ):
        opportunity = max(
            opportunity,
            0.50,
        )

    if (
        event["Onset"]
        >= 0.78
        and event[
            "Hit strength"
        ]
        >= 0.65
    ):
        opportunity = max(
            opportunity,
            0.50,
        )

    return float(
        np.clip(
            opportunity,
            0.0,
            1.0,
        )
    )


def build_event_label(
    event,
):
    labels = []

    if (
        event["Pause before"]
        >= 0.45
        and event["Onset"]
        >= 0.35
    ):
        labels.append(
            "Post-pause hit"
        )

    if (
        event["Build before"]
        >= 0.55
        and event["Onset"]
        >= 0.35
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
        event["High / tonal"]
        >= 0.55
        and event["High / tonal"]
        > event["Bass"]
        + 0.08
    ):
        labels.append(
            "High / tonal"
        )

    if (
        event["Bass"]
        >= 0.55
        and event["Bass"]
        > event["High / tonal"]
        + 0.05
    ):
        labels.append(
            "Bass / kick"
        )

    return " + ".join(
        labels
    )


def refresh_event_ranking(
    event,
):
    event["Opportunity"] = (
        calculate_event_opportunity(
            event
        )
    )

    event["Tier"] = (
        get_event_tier(
            event[
                "Opportunity"
            ]
        )
    )

    event["Event"] = (
        build_event_label(
            event
        )
    )

    return event


def cluster_micro_hits(
    events,
    cluster_seconds=0.12,
):
    if not events:
        return (
            [],
            0,
        )

    events = sorted(
        events,
        key=lambda item: item[
            "Time"
        ],
    )

    clusters = []
    current_cluster = [
        events[0]
    ]

    cluster_start_time = (
        events[0]["Time"]
    )

    for event in events[1:]:
        total_cluster_width = (
            event["Time"]
            - cluster_start_time
        )

        if (
            total_cluster_width
            <= cluster_seconds
        ):
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
        anchor = max(
            cluster,
            key=lambda item: (
                item[
                    "Hit strength"
                ],
                item[
                    "Opportunity"
                ],
            ),
        )

        merged = dict(
            anchor
        )

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
        - len(
            merged_events
        )
    )

    return (
        merged_events,
        removed_count,
    )


def keep_first_post_pause_hit(
    events,
    suppression_seconds=1.75,
):
    if not events:
        return (
            events,
            0,
        )

    events = sorted(
        events,
        key=lambda item: item[
            "Time"
        ],
    )

    last_kept_pause_time = None
    suppressed_count = 0

    for event in events:
        is_pause_candidate = (
            event[
                "Pause before"
            ]
            >= 0.45
            and event[
                "Onset"
            ]
            >= 0.35
        )

        if not is_pause_candidate:
            continue

        if (
            last_kept_pause_time
            is None
        ):
            last_kept_pause_time = (
                event["Time"]
            )
            continue

        gap = (
            event["Time"]
            - last_kept_pause_time
        )

        if (
            gap
            <= suppression_seconds
        ):
            event[
                "Pause before"
            ] = 0.0

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
        suppressed_count,
    )


def calculate_local_prominence(
    events,
    window_seconds=0.65,
):
    if not events:
        return []

    times = np.asarray(
        [
            event["Time"]
            for event
            in events
        ],
        dtype=float,
    )

    hit_strengths = np.asarray(
        [
            event[
                "Hit strength"
            ]
            for event
            in events
        ],
        dtype=float,
    )

    prominence = []

    for (
        index,
        event_time,
    ) in enumerate(
        times
    ):
        local_mask = (
            np.abs(
                times
                - event_time
            )
            <= window_seconds
        )

        local_max = float(
            np.max(
                hit_strengths[
                    local_mask
                ]
            )
        )

        if local_max <= 0:
            value = 0.0
        else:
            value = float(
                hit_strengths[
                    index
                ]
                / local_max
            )

        prominence.append(
            float(
                np.clip(
                    value,
                    0.0,
                    1.0,
                )
            )
        )

    return prominence


def build_editorial_candidates(
    events,
):
    if not events:
        return []

    events = sorted(
        events,
        key=lambda item: item[
            "Time"
        ],
    )

    local_prominence = (
        calculate_local_prominence(
            events,
            window_seconds=0.65,
        )
    )

    candidates = []

    for (
        event,
        prominence,
    ) in zip(
        events,
        local_prominence,
    ):
        context_score = max(
            event[
                "Pause before"
            ],
            event[
                "Build before"
            ],
        )

        distinctive_score = max(
            event["Bass"],
            event[
                "High / tonal"
            ],
        )

        editorial_score = (
            0.42
            * event[
                "Hit strength"
            ]
            + 0.18
            * event[
                "Beat alignment"
            ]
            + 0.18
            * prominence
            + 0.14
            * context_score
            + 0.08
            * distinctive_score
        )

        editorial_score = float(
            np.clip(
                editorial_score,
                0.0,
                1.0,
            )
        )

        reasons = []

        if (
            event["Pause before"]
            >= 0.45
            and event[
                "Hit strength"
            ]
            >= 0.40
        ):
            reasons.append(
                "Post-pause"
            )

        if (
            event["Build before"]
            >= 0.55
            and event[
                "Hit strength"
            ]
            >= 0.40
        ):
            reasons.append(
                "Post-build"
            )

        if (
            event[
                "High / tonal"
            ]
            >= 0.75
            and event[
                "Hit strength"
            ]
            >= 0.50
        ):
            reasons.append(
                "Distinct tonal hit"
            )

        if (
            event["Bass"]
            >= 0.85
            and event[
                "Hit strength"
            ]
            >= 0.60
        ):
            reasons.append(
                "Strong bass/kick"
            )

        if (
            event["Onset"]
            >= 0.78
            and event[
                "Hit strength"
            ]
            >= 0.64
            and prominence
            >= 0.80
        ):
            reasons.append(
                "Prominent strong hit"
            )

        if editorial_score >= 0.64:
            reasons.append(
                "Strong overall candidate"
            )

        if not reasons:
            continue

        candidate = dict(
            event
        )

        candidate[
            "Local prominence"
        ] = prominence

        candidate[
            "Editorial score"
        ] = editorial_score

        candidate[
            "Candidate reason"
        ] = " + ".join(
            reasons
        )

        candidates.append(
            candidate
        )

    return candidates


def detect_song_events(
    analysis,
    sr,
    build_regions,
):
    onset = analysis[
        "onset"
    ]

    bass_hit = analysis[
        "bass_hit"
    ]

    high_hit = analysis[
        "high_hit"
    ]

    rms = analysis[
        "rms"
    ]

    beat_times = analysis[
        "beat_times"
    ]

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
            or frame
            >= len(onset)
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
            onset[
                frame
            ]
        )

        bass_strength = float(
            bass_hit[
                frame
            ]
        )

        high_strength = float(
            high_hit[
                frame
            ]
        )

        energy_strength = float(
            rms[
                frame
            ]
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
                build_regions,
            )
        )

        distinctive_score = max(
            bass_strength,
            high_strength,
            onset_strength,
        )

        hit_strength = (
            0.60
            * onset_strength
            + 0.15
            * bass_strength
            + 0.15
            * high_strength
            + 0.10
            * energy_strength
        )

        if (
            onset_strength < 0.20
            and distinctive_score
            < 0.25
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

    (
        events,
        merged_count,
    ) = cluster_micro_hits(
        raw_events,
        cluster_seconds=0.12,
    )

    (
        events,
        suppressed_pause_count,
    ) = keep_first_post_pause_hit(
        events,
        suppression_seconds=1.75,
    )

    return (
        sorted(
            events,
            key=lambda item: item[
                "Time"
            ],
        ),
        len(raw_events),
        merged_count,
        suppressed_pause_count,
    )


def detect_build_regions(
    analysis,
    sr,
):
    build = analysis[
        "build_score"
    ]

    times = analysis[
        "times"
    ]

    rms = analysis[
        "rms"
    ]

    onset_density = analysis[
        "onset_density"
    ]

    brightness = analysis[
        "brightness"
    ]

    frames_per_second = (
        sr / HOP_LENGTH
    )

    threshold = 0.68

    mask = (
        build
        >= threshold
    )

    max_gap_frames = max(
        1,
        int(
            round(
                0.30
                * frames_per_second
            )
        ),
    )

    active_indices = (
        np.flatnonzero(
            mask
        )
    )

    candidate_regions = []

    if len(active_indices) > 0:
        start = int(
            active_indices[0]
        )

        previous = start

        for index in (
            active_indices[1:]
        ):
            index = int(
                index
            )

            if (
                index
                - previous
                <= max_gap_frames
                + 1
            ):
                previous = index

            else:
                candidate_regions.append(
                    (
                        start,
                        previous + 1,
                    )
                )

                start = index
                previous = index

        candidate_regions.append(
            (
                start,
                previous + 1,
            )
        )

    minimum_duration_seconds = 1.50

    minimum_frames = max(
        2,
        int(
            round(
                minimum_duration_seconds
                * frames_per_second
            )
        ),
    )

    regions = []

    for (
        start,
        end,
    ) in candidate_regions:
        if (
            end - start
            < minimum_frames
        ):
            continue

        region_length = (
            end - start
        )

        quarter = max(
            2,
            region_length // 4,
        )

        early_slice = slice(
            start,
            min(
                start + quarter,
                end,
            ),
        )

        late_slice = slice(
            max(
                start,
                end - quarter,
            ),
            end,
        )

        energy_rise = (
            float(
                np.mean(
                    rms[
                        late_slice
                    ]
                )
            )
            - float(
                np.mean(
                    rms[
                        early_slice
                    ]
                )
            )
        )

        density_rise = (
            float(
                np.mean(
                    onset_density[
                        late_slice
                    ]
                )
            )
            - float(
                np.mean(
                    onset_density[
                        early_slice
                    ]
                )
            )
        )

        brightness_rise = (
            float(
                np.mean(
                    brightness[
                        late_slice
                    ]
                )
            )
            - float(
                np.mean(
                    brightness[
                        early_slice
                    ]
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

        segment = (
            build[
                start:end
            ]
        )

        peak_local = int(
            np.argmax(
                segment
            )
        )

        peak_index = (
            start + peak_local
        )

        rise_bonus = np.clip(
            (
                max(
                    0.0,
                    energy_rise,
                )
                + max(
                    0.0,
                    density_rise,
                )
                + max(
                    0.0,
                    brightness_rise,
                )
            )
            / 0.60,
            0.0,
            1.0,
        )

        region_score = float(
            np.clip(
                0.75
                * build[
                    peak_index
                ]
                + 0.25
                * rise_bonus,
                0.0,
                1.0,
            )
        )

        regions.append(
            {
                "Start": float(
                    times[
                        start
                    ]
                ),
                "End": float(
                    times[
                        min(
                            end - 1,
                            len(times) - 1,
                        )
                    ]
                ),
                "Peak": float(
                    times[
                        peak_index
                    ]
                ),
                "Score": region_score,
                "Energy rise": energy_rise,
                "Density rise": density_rise,
                "Brightness rise": brightness_rise,
            }
        )

    return regions


def sanitize_song_name(
    original_name,
):
    song_name = Path(
        original_name
    ).stem

    song_name = re.sub(
        r'[<>:"/\\|?*]',
        "",
        song_name,
    ).strip()

    if not song_name:
        song_name = "Song"

    return song_name


def show_song_analyzer(
    y,
    sr,
    original_name,
):
    analysis = (
        build_song_analysis(
            y,
            sr,
        )
    )

    builds = (
        detect_build_regions(
            analysis,
            sr,
        )
    )

    (
        events,
        raw_event_count,
        merged_micro_hits,
        suppressed_pause_hits,
    ) = detect_song_events(
        analysis,
        sr,
        builds,
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
            / beat_period
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

    (
        metric1,
        metric2,
        metric3,
        metric4,
    ) = st.columns(4)

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
            len(beat_times),
        )

    with metric3:
        st.metric(
            "Raw events",
            len(events),
        )

    with metric4:
        st.metric(
            "Editorial candidates",
            len(
                editorial_candidates
            ),
        )

    if merged_micro_hits > 0:
        st.caption(
            f"Cleaned "
            f"{merged_micro_hits} "
            f"duplicate micro-hits "
            f"from "
            f"{raw_event_count} "
            f"raw onset candidates."
        )

    if suppressed_pause_hits > 0:
        st.caption(
            "Kept post-pause context on "
            "the first meaningful hit and "
            f"removed it from "
            f"{suppressed_pause_hits} "
            "following hits."
        )

    st.subheader(
        "Energy"
    )

    st.caption(
        "Useful for seeing quiet sections, "
        "pauses, large energy changes and drops."
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
        height=220,
    )

    st.subheader(
        "Hits and musical accents"
    )

    st.caption(
        "Overall onset = general attacks. "
        "Bass = kick/low-frequency attacks. "
        "High/Tonal = piano, guitar, cymbal, synth "
        "and other brighter attacks. "
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
        height=300,
    )

    st.subheader(
        "Build-up analysis"
    )

    st.caption(
        "The curve shows possible rising musical activity. "
        "A region is only labelled as a build when the rise "
        "is sustained and several signals increase together."
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
        height=260,
    )

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
                        2,
                    ),
                    "Energy rise": round(
                        build[
                            "Energy rise"
                        ],
                        2,
                    ),
                    "Activity rise": round(
                        build[
                            "Density rise"
                        ],
                        2,
                    ),
                    "Brightness rise": round(
                        build[
                            "Brightness rise"
                        ],
                        2,
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

    st.subheader(
        "Editorial candidates"
    )

    candidate_rows = []

    for event in (
        editorial_candidates
    ):
        candidate_rows.append(
            {
                "Song": song_name,
                "Time": format_time(
                    event["Time"]
                ),
                "Seconds": round(
                    event["Time"],
                    3,
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
                    event[
                        "Editorial score"
                    ],
                    2,
                ),
                "Opportunity": round(
                    event[
                        "Opportunity"
                    ],
                    2,
                ),
                "Hit": round(
                    event[
                        "Hit strength"
                    ],
                    2,
                ),
                "Prominence": round(
                    event[
                        "Local prominence"
                    ],
                    2,
                ),
                "Beat": round(
                    event[
                        "Beat alignment"
                    ],
                    2,
                ),
                "Pause": round(
                    event[
                        "Pause before"
                    ],
                    2,
                ),
                "Build": round(
                    event[
                        "Build before"
                    ],
                    2,
                ),
                "Bass": round(
                    event[
                        "Bass"
                    ],
                    2,
                ),
                "High": round(
                    event[
                        "High / tonal"
                    ],
                    2,
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
            on_click="ignore",
        )


def safe_cosine_distance(
    vector_a,
    vector_b,
):
    vector_a = np.asarray(
        vector_a,
        dtype=float,
    )

    vector_b = np.asarray(
        vector_b,
        dtype=float,
    )

    if (
        vector_a.size == 0
        or vector_b.size == 0
    ):
        return 0.0

    denominator = (
        np.linalg.norm(
            vector_a
        )
        * np.linalg.norm(
            vector_b
        )
    )

    if denominator <= 1e-9:
        return 0.0

    similarity = float(
        np.dot(
            vector_a,
            vector_b,
        )
        / denominator
    )

    return float(
        np.clip(
            1.0 - similarity,
            0.0,
            1.0,
        )
    )


def robust_relative_normalize(
    values,
    low_percentile=20,
    high_percentile=90,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return values

    low = float(
        np.percentile(
            values,
            low_percentile,
        )
    )

    high = float(
        np.percentile(
            values,
            high_percentile,
        )
    )

    if high <= low + 1e-9:
        return np.zeros_like(
            values
        )

    normalized = (
        values - low
    ) / (
        high - low
    )

    return np.clip(
        normalized,
        0.0,
        1.0,
    )


def mean_window(
    values,
    start,
    end,
):
    start = max(
        0,
        int(start),
    )

    end = min(
        len(values),
        int(end),
    )

    if end <= start:
        return 0.0

    return float(
        np.mean(
            values[
                start:end
            ]
        )
    )


def feature_mean_window(
    matrix,
    start,
    end,
):
    start = max(
        0,
        int(start),
    )

    end = min(
        matrix.shape[1],
        int(end),
    )

    if end <= start:
        return np.zeros(
            matrix.shape[0],
            dtype=float,
        )

    return np.mean(
        matrix[
            :,
            start:end
        ],
        axis=1,
    )


def autocorrelation_at_lag(
    values,
    lag,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    lag = int(
        round(
            lag
        )
    )

    if (
        lag <= 0
        or len(values)
        <= lag + 2
    ):
        return 0.0

    left = values[
        :-lag
    ]

    right = values[
        lag:
    ]

    left = (
        left
        - np.mean(left)
    )

    right = (
        right
        - np.mean(right)
    )

    denominator = (
        np.linalg.norm(left)
        * np.linalg.norm(right)
    )

    if denominator <= 1e-9:
        return 0.0

    return float(
        np.clip(
            np.dot(
                left,
                right,
            )
            / denominator,
            -1.0,
            1.0,
        )
    )


def rhythm_signature(
    values,
    beat_period_frames,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) < 8:
        return np.zeros(
            3,
            dtype=float,
        )

    lags = [
        max(
            1,
            beat_period_frames
            * 0.5,
        ),
        max(
            1,
            beat_period_frames,
        ),
        max(
            1,
            beat_period_frames
            * 2.0,
        ),
    ]

    return np.asarray(
        [
            autocorrelation_at_lag(
                values,
                lag,
            )
            for lag in lags
        ],
        dtype=float,
    )


def calculate_structural_raw_features(
    event_time,
    analysis,
    sr,
    beat_period,
):
    frames_per_second = (
        sr / HOP_LENGTH
    )

    frame = int(
        round(
            event_time
            * frames_per_second
        )
    )

    compare_window = max(
        6,
        int(
            round(
                1.35
                * frames_per_second
            )
        ),
    )

    event_margin = max(
        2,
        int(
            round(
                0.12
                * frames_per_second
            )
        ),
    )

    pre_start = max(
        0,
        frame
        - event_margin
        - compare_window,
    )

    pre_end = max(
        0,
        frame
        - event_margin,
    )

    post_start = min(
        len(
            analysis[
                "onset"
            ]
        ),
        frame
        + event_margin,
    )

    post_end = min(
        len(
            analysis[
                "onset"
            ]
        ),
        frame
        + event_margin
        + compare_window,
    )

    minimum_frames = max(
        4,
        int(
            round(
                0.45
                * frames_per_second
            )
        ),
    )

    if (
        pre_end - pre_start
        < minimum_frames
        or post_end - post_start
        < minimum_frames
    ):
        return {
            "Tonal change raw": 0.0,
            "Timbre change raw": 0.0,
            "Rhythm change raw": 0.0,
            "Dynamics change raw": 0.0,
        }

    pre_chroma = (
        feature_mean_window(
            analysis[
                "chroma"
            ],
            pre_start,
            pre_end,
        )
    )

    post_chroma = (
        feature_mean_window(
            analysis[
                "chroma"
            ],
            post_start,
            post_end,
        )
    )

    tonal_change = (
        safe_cosine_distance(
            pre_chroma,
            post_chroma,
        )
    )

    pre_mfcc = (
        feature_mean_window(
            analysis[
                "mfcc"
            ],
            pre_start,
            pre_end,
        )
    )

    post_mfcc = (
        feature_mean_window(
            analysis[
                "mfcc"
            ],
            post_start,
            post_end,
        )
    )

    timbre_change = float(
        np.clip(
            np.mean(
                np.abs(
                    post_mfcc
                    - pre_mfcc
                )
            )
            / 1.5,
            0.0,
            1.0,
        )
    )

    beat_period_frames = max(
        2,
        beat_period
        * frames_per_second,
    )

    pre_onset = analysis[
        "onset"
    ][
        pre_start:pre_end
    ]

    post_onset = analysis[
        "onset"
    ][
        post_start:post_end
    ]

    pre_signature = (
        rhythm_signature(
            pre_onset,
            beat_period_frames,
        )
    )

    post_signature = (
        rhythm_signature(
            post_onset,
            beat_period_frames,
        )
    )

    autocorrelation_change = float(
        np.mean(
            np.abs(
                post_signature
                - pre_signature
            )
        )
        / 2.0
    )

    density_change = abs(
        mean_window(
            analysis[
                "onset_density"
            ],
            post_start,
            post_end,
        )
        - mean_window(
            analysis[
                "onset_density"
            ],
            pre_start,
            pre_end,
        )
    )

    rhythm_change = float(
        np.clip(
            0.70
            * autocorrelation_change
            + 0.30
            * density_change,
            0.0,
            1.0,
        )
    )

    energy_change = abs(
        mean_window(
            analysis[
                "rms"
            ],
            post_start,
            post_end,
        )
        - mean_window(
            analysis[
                "rms"
            ],
            pre_start,
            pre_end,
        )
    )

    brightness_change = abs(
        mean_window(
            analysis[
                "brightness"
            ],
            post_start,
            post_end,
        )
        - mean_window(
            analysis[
                "brightness"
            ],
            pre_start,
            pre_end,
        )
    )

    dynamics_change = float(
        np.clip(
            0.60
            * energy_change
            + 0.40
            * brightness_change,
            0.0,
            1.0,
        )
    )

    return {
        "Tonal change raw": tonal_change,
        "Timbre change raw": timbre_change,
        "Rhythm change raw": rhythm_change,
        "Dynamics change raw": dynamics_change,
    }


def calculate_long_quiet_before(
    analysis,
    event_time,
    sr,
):
    frames_per_second = (
        sr / HOP_LENGTH
    )

    frame = int(
        round(
            event_time
            * frames_per_second
        )
    )

    immediate = max(
        4,
        int(
            round(
                1.4
                * frames_per_second
            )
        ),
    )

    earlier = max(
        8,
        int(
            round(
                3.0
                * frames_per_second
            )
        ),
    )

    immediate_start = max(
        0,
        frame - immediate,
    )

    earlier_start = max(
        0,
        immediate_start
        - earlier,
    )

    immediate_energy = (
        mean_window(
            analysis[
                "rms"
            ],
            immediate_start,
            frame,
        )
    )

    earlier_energy = (
        mean_window(
            analysis[
                "rms"
            ],
            earlier_start,
            immediate_start,
        )
    )

    immediate_activity = (
        mean_window(
            analysis[
                "onset_density"
            ],
            immediate_start,
            frame,
        )
    )

    earlier_activity = (
        mean_window(
            analysis[
                "onset_density"
            ],
            earlier_start,
            immediate_start,
        )
    )

    score = (
        2.0
        * max(
            0.0,
            earlier_energy
            - immediate_energy,
        )
        + 1.5
        * max(
            0.0,
            earlier_activity
            - immediate_activity,
        )
    )

    if immediate_energy < 0.22:
        score += 0.14

    if immediate_activity < 0.22:
        score += 0.10

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


def calculate_quiet_after_hit(
    analysis,
    event_time,
    sr,
):
    frames_per_second = (
        sr / HOP_LENGTH
    )

    frame = int(
        round(
            event_time
            * frames_per_second
        )
    )

    before_window = max(
        5,
        int(
            round(
                1.4
                * frames_per_second
            )
        ),
    )

    after_margin = max(
        2,
        int(
            round(
                0.16
                * frames_per_second
            )
        ),
    )

    after_window = max(
        5,
        int(
            round(
                1.35
                * frames_per_second
            )
        ),
    )

    before_start = max(
        0,
        frame
        - before_window,
    )

    after_start = min(
        len(
            analysis[
                "rms"
            ]
        ),
        frame
        + after_margin,
    )

    after_end = min(
        len(
            analysis[
                "rms"
            ]
        ),
        after_start
        + after_window,
    )

    before_energy = (
        mean_window(
            analysis[
                "rms"
            ],
            before_start,
            frame,
        )
    )

    after_energy = (
        mean_window(
            analysis[
                "rms"
            ],
            after_start,
            after_end,
        )
    )

    before_activity = (
        mean_window(
            analysis[
                "onset_density"
            ],
            before_start,
            frame,
        )
    )

    after_activity = (
        mean_window(
            analysis[
                "onset_density"
            ],
            after_start,
            after_end,
        )
    )

    score = (
        2.2
        * max(
            0.0,
            before_energy
            - after_energy,
        )
        + 1.7
        * max(
            0.0,
            before_activity
            - after_activity,
        )
    )

    if after_energy < 0.22:
        score += 0.12

    if after_activity < 0.22:
        score += 0.12

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


def calculate_burst_payoff(
    analysis,
    event_time,
    sr,
):
    frames_per_second = (
        sr / HOP_LENGTH
    )

    frame = int(
        round(
            event_time
            * frames_per_second
        )
    )

    pre_window = max(
        8,
        int(
            round(
                2.5
                * frames_per_second
            )
        ),
    )

    post_margin = max(
        2,
        int(
            round(
                0.15
                * frames_per_second
            )
        ),
    )

    post_window = max(
        5,
        int(
            round(
                1.25
                * frames_per_second
            )
        ),
    )

    pre_start = max(
        0,
        frame
        - pre_window,
    )

    post_start = min(
        len(
            analysis[
                "onset_density"
            ]
        ),
        frame
        + post_margin,
    )

    post_end = min(
        len(
            analysis[
                "onset_density"
            ]
        ),
        post_start
        + post_window,
    )

    pre_activity = (
        mean_window(
            analysis[
                "onset_density"
            ],
            pre_start,
            frame,
        )
    )

    post_activity = (
        mean_window(
            analysis[
                "onset_density"
            ],
            post_start,
            post_end,
        )
    )

    drop = max(
        0.0,
        pre_activity
        - post_activity,
    )

    score = (
        0.45
        * pre_activity
        + 0.55
        * np.clip(
            drop * 2.2,
            0.0,
            1.0,
        )
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


def calculate_directional_gap_scores(
    candidates,
):
    if not candidates:
        return (
            [],
            [],
        )

    times = np.asarray(
        [
            event["Time"]
            for event
            in candidates
        ],
        dtype=float,
    )

    substantial = np.asarray(
        [
            (
                event[
                    "Hit strength"
                ]
                >= 0.62
                and event[
                    "Local prominence"
                ]
                >= 0.78
            )
            or event[
                "Pause before"
            ]
            >= 0.45
            or event[
                "Build before"
            ]
            >= 0.55
            for event
            in candidates
        ],
        dtype=bool,
    )

    before_scores = []
    after_scores = []

    for (
        index,
        event_time,
    ) in enumerate(
        times
    ):
        earlier = times[
            (
                times
                < event_time
                - 0.12
            )
            & substantial
        ]

        later = times[
            (
                times
                > event_time
                + 0.12
            )
            & substantial
        ]

        if len(earlier) == 0:
            before_gap = 4.0
        else:
            before_gap = (
                event_time
                - float(
                    np.max(
                        earlier
                    )
                )
            )

        if len(later) == 0:
            after_gap = 4.0
        else:
            after_gap = (
                float(
                    np.min(
                        later
                    )
                )
                - event_time
            )

        before_score = float(
            np.clip(
                (
                    before_gap
                    - 0.35
                )
                / 3.0,
                0.0,
                1.0,
            )
        )

        after_score = float(
            np.clip(
                (
                    after_gap
                    - 0.35
                )
                / 3.0,
                0.0,
                1.0,
            )
        )

        before_scores.append(
            before_score
        )

        after_scores.append(
            after_score
        )

    return (
        before_scores,
        after_scores,
    )


def augment_editorial_candidates_with_opening_events(
    editorial_candidates,
    events,
    beat_times,
    opening_seconds=3.0,
):
    candidates = [
        dict(event)
        for event
        in editorial_candidates
    ]

    if not events:
        return sorted(
            candidates,
            key=lambda item: item[
                "Time"
            ],
        )

    raw_prominence = (
        calculate_local_prominence(
            events,
            window_seconds=0.65,
        )
    )

    existing_times = [
        event["Time"]
        for event
        in candidates
    ]

    for (
        event,
        prominence,
    ) in zip(
        events,
        raw_prominence,
    ):
        if (
            event["Time"]
            > opening_seconds
        ):
            break

        if event["Time"] < 0.10:
            continue

        if (
            event[
                "Hit strength"
            ]
            < 0.38
            and event["Onset"]
            < 0.42
        ):
            continue

        if existing_times:
            nearest = min(
                abs(
                    event["Time"]
                    - value
                )
                for value
                in existing_times
            )

            if nearest <= 0.08:
                continue

        context_score = max(
            event[
                "Pause before"
            ],
            event[
                "Build before"
            ],
        )

        distinctive_score = max(
            event["Bass"],
            event[
                "High / tonal"
            ],
        )

        editorial_score = float(
            np.clip(
                0.42
                * event[
                    "Hit strength"
                ]
                + 0.18
                * event[
                    "Beat alignment"
                ]
                + 0.18
                * prominence
                + 0.14
                * context_score
                + 0.08
                * distinctive_score,
                0.0,
                1.0,
            )
        )

        opening_candidate = dict(
            event
        )

        opening_candidate[
            "Local prominence"
        ] = float(
            prominence
        )

        opening_candidate[
            "Editorial score"
        ] = editorial_score

        opening_candidate[
            "Candidate reason"
        ] = (
            "Opening rhythm candidate"
        )

        candidates.append(
            opening_candidate
        )

        existing_times.append(
            event["Time"]
        )

    return sorted(
        candidates,
        key=lambda item: item[
            "Time"
        ],
    )


def calculate_local_landmark_fields(
    candidates,
    window_seconds=2.50,
):
    if not candidates:
        return []

    ordered = sorted(
        [
            dict(event)
            for event
            in candidates
        ],
        key=lambda item: item[
            "Time"
        ],
    )

    times = np.asarray(
        [
            event["Time"]
            for event
            in ordered
        ],
        dtype=float,
    )

    def ratio_and_rank(
        key,
    ):
        values = np.asarray(
            [
                event[key]
                for event
                in ordered
            ],
            dtype=float,
        )

        ratios = []
        ranks = []

        for (
            index,
            event_time,
        ) in enumerate(
            times
        ):
            mask = (
                np.abs(
                    times
                    - event_time
                )
                <= window_seconds
            )

            local = values[
                mask
            ]

            if len(local) == 0:
                ratios.append(
                    0.0
                )
                ranks.append(
                    0.0
                )
                continue

            local_max = float(
                np.max(
                    local
                )
            )

            if local_max <= 1e-9:
                ratio = 0.0
            else:
                ratio = float(
                    values[index]
                    / local_max
                )

            rank = float(
                np.mean(
                    local
                    <= values[index]
                    + 1e-9
                )
            )

            ratios.append(
                float(
                    np.clip(
                        ratio,
                        0.0,
                        1.0,
                    )
                )
            )

            ranks.append(
                float(
                    np.clip(
                        rank,
                        0.0,
                        1.0,
                    )
                )
            )

        return (
            ratios,
            ranks,
        )

    (
        structural_ratio,
        structural_rank,
    ) = ratio_and_rank(
        "Structural novelty"
    )

    (
        tonal_ratio,
        tonal_rank,
    ) = ratio_and_rank(
        "Tonal change"
    )

    (
        hit_ratio,
        hit_rank,
    ) = ratio_and_rank(
        "Hit strength"
    )

    for (
        index,
        event,
    ) in enumerate(
        ordered
    ):
        structural_landmark = float(
            np.clip(
                0.40
                * structural_ratio[
                    index
                ]
                + 0.25
                * structural_rank[
                    index
                ]
                + 0.20
                * tonal_ratio[
                    index
                ]
                + 0.15
                * tonal_rank[
                    index
                ],
                0.0,
                1.0,
            )
        )

        wide_hit_landmark = float(
            np.clip(
                0.55
                * hit_ratio[
                    index
                ]
                + 0.45
                * hit_rank[
                    index
                ],
                0.0,
                1.0,
            )
        )

        event[
            "Structural local ratio"
        ] = structural_ratio[
            index
        ]

        event[
            "Structural local rank"
        ] = structural_rank[
            index
        ]

        event[
            "Tonal local ratio"
        ] = tonal_ratio[
            index
        ]

        event[
            "Tonal local rank"
        ] = tonal_rank[
            index
        ]

        event[
            "Wide hit ratio"
        ] = hit_ratio[
            index
        ]

        event[
            "Wide hit rank"
        ] = hit_rank[
            index
        ]

        event[
            "Structural landmark"
        ] = structural_landmark

        event[
            "Wide hit landmark"
        ] = wide_hit_landmark

    return ordered


def calculate_anchor_v3_candidates(
    editorial_candidates,
    analysis,
    sr,
):
    if not editorial_candidates:
        return []

    candidates = sorted(
        [
            dict(event)
            for event
            in editorial_candidates
        ],
        key=lambda item: item[
            "Time"
        ],
    )

    beat_times = analysis[
        "beat_times"
    ]

    beat_period = (
        get_median_beat_period(
            beat_times
        )
    )

    raw_structural = []

    for event in candidates:
        raw_structural.append(
            calculate_structural_raw_features(
                event["Time"],
                analysis,
                sr,
                beat_period,
            )
        )

    tonal_relative = (
        robust_relative_normalize(
            [
                item[
                    "Tonal change raw"
                ]
                for item
                in raw_structural
            ]
        )
    )

    timbre_relative = (
        robust_relative_normalize(
            [
                item[
                    "Timbre change raw"
                ]
                for item
                in raw_structural
            ]
        )
    )

    rhythm_relative = (
        robust_relative_normalize(
            [
                item[
                    "Rhythm change raw"
                ]
                for item
                in raw_structural
            ]
        )
    )

    dynamics_relative = (
        robust_relative_normalize(
            [
                item[
                    "Dynamics change raw"
                ]
                for item
                in raw_structural
            ]
        )
    )

    (
        before_gap_scores,
        after_gap_scores,
    ) = (
        calculate_directional_gap_scores(
            candidates
        )
    )

    opening_suppression_end = min(
        2.40,
        max(
            1.45,
            3.25
            * beat_period,
        ),
    )

    first_pass = []

    for (
        index,
        event,
    ) in enumerate(
        candidates
    ):
        tonal_change = float(
            tonal_relative[
                index
            ]
        )

        timbre_change = float(
            timbre_relative[
                index
            ]
        )

        rhythm_change = float(
            rhythm_relative[
                index
            ]
        )

        dynamics_change = float(
            dynamics_relative[
                index
            ]
        )

        structural_novelty = float(
            np.clip(
                0.38
                * tonal_change
                + 0.28
                * timbre_change
                + 0.20
                * rhythm_change
                + 0.14
                * dynamics_change,
                0.0,
                1.0,
            )
        )

        if (
            event["Time"]
            < opening_suppression_end
        ):
            structural_scale = float(
                np.clip(
                    event["Time"]
                    / max(
                        opening_suppression_end,
                        0.001,
                    ),
                    0.0,
                    1.0,
                )
            )

            structural_novelty *= (
                structural_scale
            )

            tonal_change *= (
                structural_scale
            )

            timbre_change *= (
                structural_scale
            )

            rhythm_change *= (
                structural_scale
            )

            dynamics_change *= (
                structural_scale
            )

        long_quiet_before = (
            calculate_long_quiet_before(
                analysis,
                event["Time"],
                sr,
            )
        )

        quiet_after = (
            calculate_quiet_after_hit(
                analysis,
                event["Time"],
                sr,
            )
        )

        burst_payoff = (
            calculate_burst_payoff(
                analysis,
                event["Time"],
                sr,
            )
        )

        enriched = dict(
            event
        )

        enriched.update(
            {
                "Structural novelty":
                    structural_novelty,
                "Tonal change":
                    tonal_change,
                "Timbre change":
                    timbre_change,
                "Rhythm change":
                    rhythm_change,
                "Dynamics change":
                    dynamics_change,
                "Long quiet before":
                    long_quiet_before,
                "Quiet after":
                    quiet_after,
                "Burst payoff":
                    burst_payoff,
                "Before gap":
                    before_gap_scores[
                        index
                    ],
                "After gap":
                    after_gap_scores[
                        index
                    ],
            }
        )

        first_pass.append(
            enriched
        )

    first_pass = (
        calculate_local_landmark_fields(
            first_pass,
            window_seconds=2.50,
        )
    )

    output = []

    for event in first_pass:
        structural_novelty = (
            event[
                "Structural novelty"
            ]
        )

        structural_landmark = (
            event[
                "Structural landmark"
            ]
        )

        wide_hit_landmark = (
            event[
                "Wide hit landmark"
            ]
        )

        before_gap = (
            event[
                "Before gap"
            ]
        )

        after_gap = (
            event[
                "After gap"
            ]
        )

        context_before = max(
            event[
                "Pause before"
            ],
            event[
                "Long quiet before"
            ],
        )

        distinctive = max(
            event["Bass"],
            event[
                "High / tonal"
            ],
        )

        musical_quality = float(
            np.clip(
                0.28
                * event[
                    "Hit strength"
                ]
                + 0.20
                * event[
                    "Local prominence"
                ]
                + 0.16
                * event[
                    "Beat alignment"
                ]
                + 0.12
                * distinctive
                + 0.12
                * event[
                    "Editorial score"
                ]
                + 0.12
                * wide_hit_landmark,
                0.0,
                1.0,
            )
        )

        change_path = float(
            np.clip(
                0.32
                * structural_novelty
                + 0.26
                * structural_landmark
                + 0.14
                * event[
                    "Hit strength"
                ]
                + 0.08
                * event[
                    "Local prominence"
                ]
                + 0.06
                * event[
                    "Beat alignment"
                ]
                + 0.08
                * max(
                    before_gap,
                    after_gap,
                )
                + 0.06
                * wide_hit_landmark,
                0.0,
                1.0,
            )
        )

        release_path = float(
            np.clip(
                0.32
                * context_before
                + 0.18
                * event[
                    "Hit strength"
                ]
                + 0.12
                * event[
                    "Local prominence"
                ]
                + 0.08
                * event[
                    "Beat alignment"
                ]
                + 0.08
                * before_gap
                + 0.08
                * structural_novelty
                + 0.08
                * structural_landmark
                + 0.06
                * wide_hit_landmark,
                0.0,
                1.0,
            )
        )

        build_path = float(
            np.clip(
                0.34
                * event[
                    "Build before"
                ]
                + 0.18
                * event[
                    "Hit strength"
                ]
                + 0.10
                * event[
                    "Local prominence"
                ]
                + 0.06
                * event[
                    "Beat alignment"
                ]
                + 0.08
                * structural_novelty
                + 0.08
                * structural_landmark
                + 0.08
                * before_gap
                + 0.08
                * wide_hit_landmark,
                0.0,
                1.0,
            )
        )

        ending_path = float(
            np.clip(
                0.30
                * event[
                    "Quiet after"
                ]
                + 0.22
                * event[
                    "Burst payoff"
                ]
                + 0.16
                * event[
                    "Hit strength"
                ]
                + 0.08
                * event[
                    "Local prominence"
                ]
                + 0.05
                * event[
                    "Beat alignment"
                ]
                + 0.07
                * after_gap
                + 0.07
                * structural_landmark
                + 0.05
                * wide_hit_landmark,
                0.0,
                1.0,
            )
        )

        rhythmic_support = float(
            np.clip(
                0.30
                * event[
                    "Beat alignment"
                ]
                + 0.22
                * event[
                    "Hit strength"
                ]
                + 0.12
                * event[
                    "Local prominence"
                ]
                + 0.18
                * wide_hit_landmark
                + 0.18
                * structural_landmark,
                0.0,
                1.0,
            )
        )

        pathways = {
            "Structural change":
                change_path,
            "Release after quiet":
                release_path,
            "Build payoff":
                build_path,
            "Phrase/burst ending":
                ending_path,
        }

        ordered_paths = sorted(
            pathways.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        (
            top_path_name,
            top_path_score,
        ) = ordered_paths[0]

        evidence_votes = 0

        if (
            structural_landmark
            >= 0.67
            and structural_novelty
            >= 0.35
        ):
            evidence_votes += 1

        if (
            event[
                "Tonal local rank"
            ]
            >= 0.78
            and event[
                "Tonal change"
            ]
            >= 0.45
        ):
            evidence_votes += 1

        if context_before >= 0.48:
            evidence_votes += 1

        if (
            event[
                "Build before"
            ]
            >= 0.52
        ):
            evidence_votes += 1

        if (
            event[
                "Quiet after"
            ]
            >= 0.50
            or event[
                "Burst payoff"
            ]
            >= 0.60
        ):
            evidence_votes += 1

        if (
            max(
                before_gap,
                after_gap,
            )
            >= 0.60
        ):
            evidence_votes += 1

        if (
            event[
                "Beat alignment"
            ]
            >= 0.72
            and event[
                "Hit strength"
            ]
            >= 0.68
            and wide_hit_landmark
            >= 0.68
        ):
            evidence_votes += 1

        anchor_score = (
            top_path_score
        )

        if evidence_votes >= 2:
            anchor_score += (
                0.025
                * min(
                    3,
                    evidence_votes
                    - 1,
                )
            )

        if (
            rhythmic_support
            >= 0.75
            and top_path_score
            >= 0.58
        ):
            anchor_score += 0.02

        if (
            musical_quality
            >= 0.84
            and top_path_score
            >= 0.60
        ):
            anchor_score += 0.02

        anchor_score = float(
            np.clip(
                anchor_score,
                0.0,
                1.0,
            )
        )

        reasons = [
            top_path_name
        ]

        if (
            structural_landmark
            >= 0.70
        ):
            reasons.append(
                "Local structural landmark"
            )

        if (
            event[
                "Tonal local rank"
            ]
            >= 0.82
        ):
            reasons.append(
                "Local tonal landmark"
            )

        if (
            structural_novelty
            >= 0.65
        ):
            reasons.append(
                "Strong musical change"
            )

        if context_before >= 0.55:
            reasons.append(
                "Quiet before"
            )

        if (
            event[
                "Build before"
            ]
            >= 0.55
        ):
            reasons.append(
                "After build"
            )

        if (
            event[
                "Quiet after"
            ]
            >= 0.58
        ):
            reasons.append(
                "Quiet after"
            )

        if (
            event[
                "Burst payoff"
            ]
            >= 0.62
        ):
            reasons.append(
                "End of dense passage"
            )

        if (
            rhythmic_support
            >= 0.78
        ):
            reasons.append(
                "Strong rhythmic support"
            )

        if anchor_score >= 0.80:
            anchor_tier = "CORE"
        elif anchor_score >= 0.68:
            anchor_tier = "LIKELY"
        elif anchor_score >= 0.58:
            anchor_tier = "POSSIBLE"
        else:
            anchor_tier = "SUPPORT"

        enriched = dict(
            event
        )

        enriched.update(
            {
                "Anchor score":
                    anchor_score,
                "Anchor tier":
                    anchor_tier,
                "Anchor pathway":
                    top_path_name,
                "Anchor reason":
                    " + ".join(
                        reasons
                    ),
                "Evidence votes":
                    evidence_votes,
                "Musical quality":
                    musical_quality,
                "Change path":
                    change_path,
                "Release path":
                    release_path,
                "Build path":
                    build_path,
                "Ending path":
                    ending_path,
                "Rhythmic support":
                    rhythmic_support,
                "Opening grid support":
                    0.0,
                "Opening score":
                    0.0,
            }
        )

        output.append(
            enriched
        )

    return output


def calculate_opening_grid_support(
    event_time,
    anchor_candidates,
    beat_period,
    lookahead_beats=4,
):
    if (
        not anchor_candidates
        or beat_period <= 0
    ):
        return 0.0

    tolerance = max(
        0.08,
        min(
            0.18,
            0.30
            * beat_period,
        ),
    )

    support_values = []

    for step in range(
        1,
        lookahead_beats + 1,
    ):
        target = (
            event_time
            + step
            * beat_period
        )

        nearby = [
            event
            for event
            in anchor_candidates
            if (
                abs(
                    event["Time"]
                    - target
                )
                <= tolerance
            )
        ]

        if not nearby:
            support_values.append(
                0.0
            )
            continue

        strongest = max(
            nearby,
            key=lambda item: item[
                "Hit strength"
            ],
        )

        support_values.append(
            float(
                np.clip(
                    0.65
                    * strongest[
                        "Hit strength"
                    ]
                    + 0.35
                    * strongest[
                        "Beat alignment"
                    ],
                    0.0,
                    1.0,
                )
            )
        )

    if not support_values:
        return 0.0

    return float(
        np.mean(
            support_values
        )
    )


def choose_opening_anchor_v3(
    preview,
    anchor_candidates,
    beat_period,
):
    if not anchor_candidates:
        return preview

    opening_end = min(
        3.20,
        max(
            1.90,
            6.0
            * beat_period,
        ),
    )

    opening_candidates = [
        event
        for event
        in anchor_candidates
        if (
            0.10
            <= event["Time"]
            <= opening_end
        )
    ]

    if not opening_candidates:
        return preview

    scored = []

    for event in opening_candidates:
        grid_support = (
            calculate_opening_grid_support(
                event["Time"],
                anchor_candidates,
                beat_period,
                lookahead_beats=4,
            )
        )

        opening_score = float(
            np.clip(
                0.34
                * event[
                    "Beat alignment"
                ]
                + 0.24
                * event[
                    "Hit strength"
                ]
                + 0.18
                * grid_support
                + 0.14
                * event[
                    "Local prominence"
                ]
                + 0.10
                * event[
                    "Onset"
                ],
                0.0,
                1.0,
            )
        )

        candidate = dict(
            event
        )

        candidate[
            "Opening grid support"
        ] = grid_support

        candidate[
            "Opening score"
        ] = opening_score

        scored.append(
            candidate
        )

    best_score = max(
        event[
            "Opening score"
        ]
        for event
        in scored
    )

    competitive = [
        event
        for event
        in scored
        if (
            event[
                "Opening score"
            ]
            >= best_score
            - 0.055
            and event[
                "Beat alignment"
            ]
            >= 0.60
            and event[
                "Opening grid support"
            ]
            >= 0.30
        )
    ]

    if competitive:
        chosen = min(
            competitive,
            key=lambda item: item[
                "Time"
            ],
        )
    else:
        chosen = max(
            scored,
            key=lambda item: item[
                "Opening score"
            ],
        )

    if (
        chosen[
            "Opening score"
        ]
        < 0.52
    ):
        return preview

    result = [
        dict(event)
        for event
        in preview
        if (
            event["Time"]
            > opening_end
        )
    ]

    chosen[
        "Anchor pathway"
    ] = "Opening rhythm"

    chosen[
        "Anchor reason"
    ] = (
        "Opening rhythm + "
        "Repeating grid support"
    )

    chosen[
        "Anchor score"
    ] = max(
        chosen[
            "Anchor score"
        ],
        chosen[
            "Opening score"
        ],
    )

    if (
        chosen[
            "Anchor score"
        ]
        >= 0.80
    ):
        chosen[
            "Anchor tier"
        ] = "CORE"

    elif (
        chosen[
            "Anchor score"
        ]
        >= 0.68
    ):
        chosen[
            "Anchor tier"
        ] = "LIKELY"

    else:
        chosen[
            "Anchor tier"
        ] = "POSSIBLE"

    result.append(
        chosen
    )

    return sorted(
        result,
        key=lambda item: item[
            "Time"
        ],
    )


def collapse_anchor_alternatives_v3(
    anchor_candidates,
    beat_period,
):
    if not anchor_candidates:
        return []

    group_seconds = min(
        0.82,
        max(
            0.46,
            1.35
            * beat_period,
        ),
    )

    ordered = sorted(
        anchor_candidates,
        key=lambda item: item[
            "Time"
        ],
    )

    groups = []

    current_group = [
        ordered[0]
    ]

    group_start = (
        ordered[0]["Time"]
    )

    for event in ordered[1:]:
        if (
            event["Time"]
            - group_start
            <= group_seconds
        ):
            current_group.append(
                event
            )
        else:
            groups.append(
                current_group
            )

            current_group = [
                event
            ]

            group_start = (
                event["Time"]
            )

    groups.append(
        current_group
    )

    selected = []

    for group in groups:
        scored = []

        for event in group:
            selection_score = float(
                np.clip(
                    event[
                        "Anchor score"
                    ]
                    + 0.06
                    * event[
                        "Structural landmark"
                    ]
                    + 0.04
                    * event[
                        "Wide hit landmark"
                    ],
                    0.0,
                    1.15,
                )
            )

            scored.append(
                (
                    event,
                    selection_score,
                )
            )

        (
            best_event,
            best_score,
        ) = max(
            scored,
            key=lambda item: item[1],
        )

        comparable = [
            event
            for (
                event,
                score,
            ) in scored
            if (
                score
                >= best_score
                - 0.025
                and event[
                    "Anchor pathway"
                ]
                == best_event[
                    "Anchor pathway"
                ]
                and abs(
                    event[
                        "Structural landmark"
                    ]
                    - best_event[
                        "Structural landmark"
                    ]
                )
                <= 0.08
            )
        ]

        if comparable:
            chosen = min(
                comparable,
                key=lambda item: item[
                    "Time"
                ],
            )
        else:
            chosen = best_event

        selected.append(
            dict(
                chosen
            )
        )

    return selected


def rescue_long_anchor_gaps_v3(
    preview,
    anchor_candidates,
    analysis,
):
    if not anchor_candidates:
        return preview

    beat_period = (
        get_median_beat_period(
            analysis[
                "beat_times"
            ]
        )
    )

    duration = (
        float(
            analysis[
                "times"
            ][-1]
        )
        if len(
            analysis[
                "times"
            ]
        )
        else 0.0
    )

    minimum_gap = max(
        5.0,
        10.0
        * beat_period,
    )

    selected = sorted(
        [
            dict(event)
            for event
            in preview
        ],
        key=lambda item: item[
            "Time"
        ],
    )

    boundaries = [
        0.0
    ]

    boundaries.extend(
        event["Time"]
        for event
        in selected
    )

    boundaries.append(
        duration
    )

    additions = []

    for (
        start,
        end,
    ) in zip(
        boundaries[:-1],
        boundaries[1:],
    ):
        if (
            end - start
            < minimum_gap
        ):
            continue

        pool = [
            event
            for event
            in anchor_candidates
            if (
                event["Time"]
                > start
                + 0.80
                and event["Time"]
                < end
                - 0.80
                and event[
                    "Hit strength"
                ]
                >= 0.66
                and event[
                    "Local prominence"
                ]
                >= 0.76
                and event[
                    "Beat alignment"
                ]
                >= 0.55
                and event[
                    "Wide hit landmark"
                ]
                >= 0.60
            )
        ]

        if not pool:
            continue

        rescored = []

        for event in pool:
            if not (
                event[
                    "Structural landmark"
                ]
                >= 0.38
                or event[
                    "Wide hit rank"
                ]
                >= 0.72
            ):
                continue

            rescue_score = float(
                np.clip(
                    0.28
                    * event[
                        "Rhythmic support"
                    ]
                    + 0.24
                    * event[
                        "Structural landmark"
                    ]
                    + 0.18
                    * event[
                        "Wide hit landmark"
                    ]
                    + 0.15
                    * event[
                        "Beat alignment"
                    ]
                    + 0.15
                    * max(
                        event[
                            "Before gap"
                        ],
                        event[
                            "After gap"
                        ],
                    ),
                    0.0,
                    1.0,
                )
            )

            if (
                rescue_score
                >= 0.62
            ):
                rescored.append(
                    (
                        event,
                        rescue_score,
                    )
                )

        if not rescored:
            continue

        (
            chosen,
            rescue_score,
        ) = max(
            rescored,
            key=lambda item: item[1],
        )

        rescued = dict(
            chosen
        )

        rescued[
            "Anchor pathway"
        ] = (
            "Long-gap rhythmic landmark"
        )

        rescued[
            "Anchor reason"
        ] = (
            "Long-gap rhythmic landmark "
            "+ Strong local beat"
        )

        rescued[
            "Anchor score"
        ] = max(
            rescued[
                "Anchor score"
            ],
            rescue_score,
        )

        if (
            rescued[
                "Anchor score"
            ]
            >= 0.80
        ):
            rescued[
                "Anchor tier"
            ] = "CORE"

        elif (
            rescued[
                "Anchor score"
            ]
            >= 0.68
        ):
            rescued[
                "Anchor tier"
            ] = "LIKELY"

        else:
            rescued[
                "Anchor tier"
            ] = "POSSIBLE"

        additions.append(
            rescued
        )

    combined = (
        selected
        + additions
    )

    return sorted(
        combined,
        key=lambda item: item[
            "Time"
        ],
    )


def build_anchor_v3_preview(
    anchor_candidates,
    analysis,
):
    if not anchor_candidates:
        return []

    beat_period = (
        get_median_beat_period(
            analysis[
                "beat_times"
            ]
        )
    )

    preview_pool = []

    for event in anchor_candidates:
        keep = False

        context_before = max(
            event[
                "Pause before"
            ],
            event[
                "Long quiet before"
            ],
        )

        if (
            event[
                "Anchor score"
            ]
            >= 0.74
            and event[
                "Evidence votes"
            ]
            >= 2
        ):
            keep = True

        if (
            event[
                "Change path"
            ]
            >= 0.62
            and event[
                "Structural landmark"
            ]
            >= 0.66
            and event[
                "Structural local rank"
            ]
            >= 0.68
            and event[
                "Hit strength"
            ]
            >= 0.42
        ):
            keep = True

        if (
            event[
                "Release path"
            ]
            >= 0.64
            and context_before
            >= 0.52
            and event[
                "Hit strength"
            ]
            >= 0.42
        ):
            keep = True

        if (
            event[
                "Build path"
            ]
            >= 0.64
            and event[
                "Build before"
            ]
            >= 0.50
        ):
            keep = True

        if (
            event[
                "Ending path"
            ]
            >= 0.64
            and (
                event[
                    "Quiet after"
                ]
                >= 0.50
                or event[
                    "Burst payoff"
                ]
                >= 0.60
            )
        ):
            keep = True

        if (
            event[
                "Rhythmic support"
            ]
            >= 0.76
            and event[
                "Beat alignment"
            ]
            >= 0.72
            and event[
                "Wide hit landmark"
            ]
            >= 0.70
            and event[
                "Structural landmark"
            ]
            >= 0.64
            and (
                event[
                    "Structural local rank"
                ]
                >= 0.72
                or event[
                    "Tonal local rank"
                ]
                >= 0.80
            )
        ):
            keep = True

        if keep:
            preview_pool.append(
                event
            )

    preview = (
        collapse_anchor_alternatives_v3(
            preview_pool,
            beat_period,
        )
    )

    preview = (
        rescue_long_anchor_gaps_v3(
            preview,
            anchor_candidates,
            analysis,
        )
    )

    preview = (
        collapse_anchor_alternatives_v3(
            preview,
            beat_period,
        )
    )

    preview = (
        choose_opening_anchor_v3(
            preview,
            anchor_candidates,
            beat_period,
        )
    )

    return preview


def anchor_v3_dataframe(
    anchor_candidates,
    song_name,
):
    rows = []

    for event in (
        anchor_candidates
    ):
        rows.append(
            {
                "Song": song_name,
                "Time": format_time(
                    event["Time"]
                ),
                "Seconds": round(
                    event["Time"],
                    3,
                ),
                "Anchor tier":
                    event[
                        "Anchor tier"
                    ],
                "Anchor score":
                    round(
                        event[
                            "Anchor score"
                        ],
                        3,
                    ),
                "Anchor pathway":
                    event[
                        "Anchor pathway"
                    ],
                "Anchor reason":
                    event[
                        "Anchor reason"
                    ],
                "Evidence votes":
                    event[
                        "Evidence votes"
                    ],
                "Structural novelty":
                    round(
                        event[
                            "Structural novelty"
                        ],
                        3,
                    ),
                "Structural landmark":
                    round(
                        event[
                            "Structural landmark"
                        ],
                        3,
                    ),
                "Structural local rank":
                    round(
                        event[
                            "Structural local rank"
                        ],
                        3,
                    ),
                "Tonal change":
                    round(
                        event[
                            "Tonal change"
                        ],
                        3,
                    ),
                "Tonal local rank":
                    round(
                        event[
                            "Tonal local rank"
                        ],
                        3,
                    ),
                "Timbre change":
                    round(
                        event[
                            "Timbre change"
                        ],
                        3,
                    ),
                "Rhythm change":
                    round(
                        event[
                            "Rhythm change"
                        ],
                        3,
                    ),
                "Dynamics change":
                    round(
                        event[
                            "Dynamics change"
                        ],
                        3,
                    ),
                "Quiet before":
                    round(
                        event[
                            "Long quiet before"
                        ],
                        3,
                    ),
                "Quiet after":
                    round(
                        event[
                            "Quiet after"
                        ],
                        3,
                    ),
                "Burst payoff":
                    round(
                        event[
                            "Burst payoff"
                        ],
                        3,
                    ),
                "Before gap":
                    round(
                        event[
                            "Before gap"
                        ],
                        3,
                    ),
                "After gap":
                    round(
                        event[
                            "After gap"
                        ],
                        3,
                    ),
                "Wide hit landmark":
                    round(
                        event[
                            "Wide hit landmark"
                        ],
                        3,
                    ),
                "Wide hit rank":
                    round(
                        event[
                            "Wide hit rank"
                        ],
                        3,
                    ),
                "Rhythmic support":
                    round(
                        event[
                            "Rhythmic support"
                        ],
                        3,
                    ),
                "Opening grid support":
                    round(
                        event.get(
                            "Opening grid support",
                            0.0,
                        ),
                        3,
                    ),
                "Opening score":
                    round(
                        event.get(
                            "Opening score",
                            0.0,
                        ),
                        3,
                    ),
                "Hit":
                    round(
                        event[
                            "Hit strength"
                        ],
                        3,
                    ),
                "Prominence":
                    round(
                        event[
                            "Local prominence"
                        ],
                        3,
                    ),
                "Beat":
                    round(
                        event[
                            "Beat alignment"
                        ],
                        3,
                    ),
                "Pause":
                    round(
                        event[
                            "Pause before"
                        ],
                        3,
                    ),
                "Build":
                    round(
                        event[
                            "Build before"
                        ],
                        3,
                    ),
                "Bass":
                    round(
                        event[
                            "Bass"
                        ],
                        3,
                    ),
                "High":
                    round(
                        event[
                            "High / tonal"
                        ],
                        3,
                    ),
                "Event":
                    event[
                        "Event"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


def render_checkerboard_reference(
    audio_path,
    cut_times,
    total_duration,
    fps_value,
    fps_ffmpeg,
    work_dir,
    output_name="anchor_v3.mp4",
):
    total_video_frames = (
        math.ceil(
            total_duration
            * fps_value
        )
    )

    scene_change_frames = []

    for cut_time in cut_times:
        frame_number = (
            beat_time_to_frame(
                float(
                    cut_time
                ),
                fps_value,
            )
        )

        if (
            0
            < frame_number
            < total_video_frames
        ):
            scene_change_frames.append(
                frame_number
            )

    scene_change_frames = (
        sorted(
            set(
                scene_change_frames
            )
        )
    )

    raw_video_path = (
        work_dir
        / "anchor_v3.rgb"
    )

    change_index = 0
    pattern_index = -1

    with open(
        raw_video_path,
        "wb",
    ) as raw_video:
        for frame_number in range(
            total_video_frames
        ):
            while (
                change_index
                < len(
                    scene_change_frames
                )
                and frame_number
                >= scene_change_frames[
                    change_index
                ]
            ):
                pattern_index += 1
                change_index += 1

            if pattern_index < 0:
                frame_data = (
                    BLACK_FRAME
                )
            elif (
                pattern_index
                % 2
                == 0
            ):
                frame_data = (
                    PATTERN_A_FRAME
                )
            else:
                frame_data = (
                    PATTERN_B_FRAME
                )

            raw_video.write(
                frame_data
            )

    output_path = (
        work_dir
        / output_name
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
            (
                f"{INTERNAL_WIDTH}"
                f"x{INTERNAL_HEIGHT}"
            ),
            "-r",
            fps_ffmpeg,
            "-i",
            str(
                raw_video_path
            ),
            "-i",
            str(
                audio_path
            ),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            (
                f"scale="
                f"{VIDEO_WIDTH}:"
                f"{VIDEO_HEIGHT}:"
                f"flags=neighbor,"
                f"format=yuv420p"
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
            str(
                output_path
            ),
        ]
    )

    return (
        output_path.read_bytes(),
        scene_change_frames,
    )


mode = st.radio(
    "Mode",
    [
        "Beat",
        "Smart Edit",
        "Song Analyzer",
    ],
    horizontal=True,
)

uploaded_file = (
    st.file_uploader(
        "Upload audio",
        type=[
            "mp3",
            "wav",
            "m4a",
        ],
    )
)

st.caption(
    "MP3, WAV or M4A · Maximum length: 6 minutes"
)

if uploaded_file is not None:
    st.audio(
        uploaded_file
    )


if mode == "Song Analyzer":
    st.write(
        "Song Analyzer finds beats, hits, pauses, "
        "build-ups and distinctive musical accents."
    )

    if uploaded_file is not None:
        if st.button(
            "🔍 Analyze Song",
            type="primary",
        ):
            status = None

            try:
                status = st.status(
                    "Analyzing song...",
                    expanded=True,
                )

                with tempfile.TemporaryDirectory() as work_dir:
                    work_dir = Path(
                        work_dir
                    )

                    status.write(
                        "1/4 · Preparing audio"
                    )

                    audio_path = (
                        save_uploaded_audio(
                            uploaded_file,
                            work_dir,
                        )
                    )

                    duration = (
                        get_audio_duration(
                            str(
                                audio_path
                            )
                        )
                    )

                    if (
                        duration
                        > MAX_AUDIO_DURATION
                    ):
                        status.update(
                            label="Audio is too long",
                            state="error",
                        )

                        st.error(
                            "Please upload a track "
                            "of 6 minutes or less."
                        )

                        st.stop()

                    analysis_path = (
                        create_analysis_wav(
                            audio_path,
                            work_dir,
                        )
                    )

                    status.write(
                        "2/4 · Reading waveform"
                    )

                    y, sr = (
                        librosa.load(
                            str(
                                analysis_path
                            ),
                            sr=None,
                            mono=True,
                        )
                    )

                    status.write(
                        "3/4 · Detecting musical structure"
                    )

                    status.write(
                        "4/4 · Finding editing opportunities"
                    )

                    analysis_output = (
                        y,
                        sr,
                    )

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

                st.error(
                    "Song analysis failed."
                )

                st.code(
                    str(
                        error
                    )
                )


elif mode == "Smart Edit":
    st.write(
        "Anchor v3 is still a P1-first calibration test. "
        "A strong beat is no longer enough by itself. "
        "P1 now needs structural/contextual importance, "
        "local landmark status, or a long-gap rhythmic rescue. "
        "Opening pickup correction is also stronger."
    )

    st.subheader(
        "Anchor v3 settings"
    )

    st.caption(
        "Output: 16:9 · 1920×1080 · P1 anchor preview"
    )

    smart_fps_choice = st.radio(
        "Frame rate",
        options=[
            "25 fps",
            "23.976 fps",
        ],
        horizontal=True,
        key="anchor_v3_fps",
    )

    SMART_FPS_VALUE = (
        FPS_OPTIONS[
            smart_fps_choice
        ][
            "value"
        ]
    )

    SMART_FPS_FFMPEG = (
        FPS_OPTIONS[
            smart_fps_choice
        ][
            "ffmpeg"
        ]
    )

    if uploaded_file is not None:
        smart_signature = (
            uploaded_file.name,
            len(
                uploaded_file.getvalue()
            ),
            smart_fps_choice,
            "anchor-v3",
        )

        if (
            st.session_state.get(
                "anchor_v3_signature"
            )
            != smart_signature
        ):
            st.session_state.pop(
                "anchor_v3_result",
                None,
            )

        if st.button(
            "🎯 Analyze P1 anchors v3",
            type="primary",
        ):
            status = None

            try:
                status = st.status(
                    "Finding Anchor v3 P1 moments...",
                    expanded=True,
                )

                with tempfile.TemporaryDirectory() as work_dir:
                    work_dir = Path(
                        work_dir
                    )

                    status.write(
                        "1/8 · Preparing audio"
                    )

                    audio_path = (
                        save_uploaded_audio(
                            uploaded_file,
                            work_dir,
                        )
                    )

                    total_duration = (
                        get_audio_duration(
                            str(
                                audio_path
                            )
                        )
                    )

                    if (
                        total_duration
                        > MAX_AUDIO_DURATION
                    ):
                        status.update(
                            label="Audio is too long",
                            state="error",
                        )

                        st.error(
                            "Please upload a track "
                            "of 6 minutes or less."
                        )

                        st.stop()

                    analysis_path = (
                        create_analysis_wav(
                            audio_path,
                            work_dir,
                        )
                    )

                    status.write(
                        "2/8 · Reading waveform"
                    )

                    y, sr = (
                        librosa.load(
                            str(
                                analysis_path
                            ),
                            sr=None,
                            mono=True,
                        )
                    )

                    status.write(
                        "3/8 · Detecting beats, hits, harmony and timbre"
                    )

                    analysis = (
                        build_song_analysis(
                            y,
                            sr,
                        )
                    )

                    status.write(
                        "4/8 · Building sensitive editorial candidate map"
                    )

                    builds = (
                        detect_build_regions(
                            analysis,
                            sr,
                        )
                    )

                    (
                        events,
                        raw_event_count,
                        merged_micro_hits,
                        suppressed_pause_hits,
                    ) = detect_song_events(
                        analysis,
                        sr,
                        builds,
                    )

                    editorial_candidates = (
                        build_editorial_candidates(
                            events
                        )
                    )

                    editorial_candidates = (
                        augment_editorial_candidates_with_opening_events(
                            editorial_candidates,
                            events,
                            analysis[
                                "beat_times"
                            ],
                            opening_seconds=3.0,
                        )
                    )

                    status.write(
                        "5/8 · Measuring local musical landmarks"
                    )

                    anchor_candidates = (
                        calculate_anchor_v3_candidates(
                            editorial_candidates,
                            analysis,
                            sr,
                        )
                    )

                    status.write(
                        "6/8 · Selecting P1 anchors without rhythmic flooding"
                    )

                    anchor_preview = (
                        build_anchor_v3_preview(
                            anchor_candidates,
                            analysis,
                        )
                    )

                    song_name = (
                        sanitize_song_name(
                            uploaded_file.name
                        )
                    )

                    full_anchor_dataframe = (
                        anchor_v3_dataframe(
                            anchor_candidates,
                            song_name,
                        )
                    )

                    preview_dataframe = (
                        anchor_v3_dataframe(
                            anchor_preview,
                            song_name,
                        )
                    )

                    status.write(
                        "7/8 · Building frame-accurate checkerboard preview"
                    )

                    preview_times = [
                        event["Time"]
                        for event
                        in anchor_preview
                    ]

                    (
                        video_bytes,
                        scene_frames,
                    ) = render_checkerboard_reference(
                        audio_path=audio_path,
                        cut_times=preview_times,
                        total_duration=total_duration,
                        fps_value=SMART_FPS_VALUE,
                        fps_ffmpeg=SMART_FPS_FFMPEG,
                        work_dir=work_dir,
                        output_name="anchor_v3.mp4",
                    )

                    status.write(
                        "8/8 · Finalizing Anchor v3 report"
                    )

                    beat_period = (
                        get_median_beat_period(
                            analysis[
                                "beat_times"
                            ]
                        )
                    )

                    tempo = (
                        60.0
                        / beat_period
                        if beat_period > 0
                        else 0.0
                    )

                    event_rate = (
                        len(events)
                        / total_duration
                        if total_duration > 0
                        else 0.0
                    )

                    video_filename = (
                        f"{song_name}_"
                        f"DetectTheBeat_"
                        f"AnchorV3.mp4"
                    )

                    anchor_csv_filename = (
                        f"{song_name}_"
                        f"DetectTheBeat_"
                        f"AnchorV3_Analysis.csv"
                    )

                    preview_csv_filename = (
                        f"{song_name}_"
                        f"DetectTheBeat_"
                        f"AnchorV3_Preview.csv"
                    )

                st.session_state[
                    "anchor_v3_signature"
                ] = smart_signature

                st.session_state[
                    "anchor_v3_result"
                ] = {
                    "video_bytes":
                        video_bytes,
                    "video_filename":
                        video_filename,
                    "anchor_dataframe":
                        full_anchor_dataframe,
                    "preview_dataframe":
                        preview_dataframe,
                    "anchor_csv_filename":
                        anchor_csv_filename,
                    "preview_csv_filename":
                        preview_csv_filename,
                    "preview_count":
                        len(
                            scene_frames
                        ),
                    "candidate_count":
                        len(
                            editorial_candidates
                        ),
                    "raw_event_count":
                        len(
                            events
                        ),
                    "tempo":
                        tempo,
                    "event_rate":
                        event_rate,
                    "merged_micro_hits":
                        merged_micro_hits,
                    "suppressed_pause_hits":
                        suppressed_pause_hits,
                }

                status.update(
                    label="Anchor v3 analysis ready",
                    state="complete",
                    expanded=False,
                )

            except Exception as error:
                if status is not None:
                    try:
                        status.update(
                            label="Anchor v3 analysis failed",
                            state="error",
                        )
                    except Exception:
                        pass

                st.error(
                    "Anchor v3 analysis failed."
                )

                st.code(
                    str(
                        error
                    )
                )

        smart_result = (
            st.session_state.get(
                "anchor_v3_result"
            )
        )

        if (
            smart_result
            is not None
            and st.session_state.get(
                "anchor_v3_signature"
            )
            == smart_signature
        ):
            st.success(
                f"Anchor v3 preview contains "
                f"{smart_result['preview_count']} "
                f"scene changes."
            )

            (
                metric1,
                metric2,
                metric3,
                metric4,
            ) = st.columns(4)

            with metric1:
                st.metric(
                    "P1 preview",
                    smart_result[
                        "preview_count"
                    ],
                )

            with metric2:
                st.metric(
                    "Candidates",
                    smart_result[
                        "candidate_count"
                    ],
                )

            with metric3:
                st.metric(
                    "Tempo",
                    (
                        f"{smart_result['tempo']:.1f} BPM"
                    ),
                )

            with metric4:
                st.metric(
                    "Event rate",
                    (
                        f"{smart_result['event_rate']:.1f}/s"
                    ),
                )

            st.caption(
                "Still P1 calibration only — not final Balanced. "
                "V3 is testing whether local structural importance "
                "can keep the good V2 recall while stopping ordinary "
                "strong beats from flooding the P1 layer."
            )

            st.video(
                smart_result[
                    "video_bytes"
                ]
            )

            st.download_button(
                "📥 Download Anchor v3 preview video",
                data=smart_result[
                    "video_bytes"
                ],
                file_name=smart_result[
                    "video_filename"
                ],
                mime="video/mp4",
                on_click="ignore",
            )

            st.subheader(
                "Anchor v3 P1 preview"
            )

            st.dataframe(
                smart_result[
                    "preview_dataframe"
                ],
                use_container_width=True,
                hide_index=True,
                height=520,
            )

            preview_csv_bytes = (
                smart_result[
                    "preview_dataframe"
                ]
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            )

            st.download_button(
                "Download Anchor v3 preview CSV",
                data=preview_csv_bytes,
                file_name=smart_result[
                    "preview_csv_filename"
                ],
                mime="text/csv",
                on_click="ignore",
            )

            with st.expander(
                "Full Anchor v3 analysis"
            ):
                st.dataframe(
                    smart_result[
                        "anchor_dataframe"
                    ],
                    use_container_width=True,
                    hide_index=True,
                    height=520,
                )

                full_csv_bytes = (
                    smart_result[
                        "anchor_dataframe"
                    ]
                    .to_csv(
                        index=False
                    )
                    .encode(
                        "utf-8"
                    )
                )

                st.download_button(
                    "Download full Anchor v3 Analysis CSV",
                    data=full_csv_bytes,
                    file_name=smart_result[
                        "anchor_csv_filename"
                    ],
                    mime="text/csv",
                    on_click="ignore",
                )


elif mode == "Beat":
    st.write(
        "Beat mode creates a checkerboard "
        "reference video for Scene Edit Detection."
    )

    st.subheader(
        "Video settings"
    )

    st.caption(
        "Output: 16:9 · 1920×1080"
    )

    fps_choice = st.radio(
        "Frame rate",
        options=[
            "25 fps",
            "23.976 fps",
        ],
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

    FPS_VALUE = (
        FPS_OPTIONS[
            fps_choice
        ][
            "value"
        ]
    )

    FPS_FFMPEG = (
        FPS_OPTIONS[
            fps_choice
        ][
            "ffmpeg"
        ]
    )

    BEAT_INTERVAL = (
        BEAT_INTERVALS[
            beat_choice
        ]
    )

    if uploaded_file is not None:
        download_filename = (
            make_output_filename(
                uploaded_file.name,
                beat_choice,
            )
        )

        if st.button(
            "🚀 Generate Video",
            type="primary",
        ):
            status = None

            try:
                status = st.status(
                    "Preparing audio...",
                    expanded=True,
                )

                with tempfile.TemporaryDirectory() as work_dir:
                    work_dir = Path(
                        work_dir
                    )

                    status.write(
                        "1/5 · Checking audio"
                    )

                    audio_path = (
                        save_uploaded_audio(
                            uploaded_file,
                            work_dir,
                        )
                    )

                    total_duration = (
                        get_audio_duration(
                            str(
                                audio_path
                            )
                        )
                    )

                    if (
                        total_duration
                        > MAX_AUDIO_DURATION
                    ):
                        status.update(
                            label="Audio is too long",
                            state="error",
                        )

                        st.error(
                            "Please upload a track "
                            "of 6 minutes or less."
                        )

                        st.stop()

                    minutes = int(
                        total_duration
                        // 60
                    )

                    seconds = int(
                        total_duration
                        % 60
                    )

                    status.write(
                        f"Audio length: "
                        f"{minutes}:"
                        f"{seconds:02d}"
                    )

                    status.write(
                        "2/5 · Preparing audio for beat detection"
                    )

                    analysis_path = (
                        create_analysis_wav(
                            audio_path,
                            work_dir,
                        )
                    )

                    status.write(
                        "3/5 · Detecting rhythm and strong musical accents"
                    )

                    y, sr = (
                        librosa.load(
                            str(
                                analysis_path
                            ),
                            sr=None,
                            mono=True,
                        )
                    )

                    onset_envelope = (
                        librosa.onset.onset_strength(
                            y=y,
                            sr=sr,
                            hop_length=HOP_LENGTH,
                        )
                    )

                    (
                        _,
                        beat_frames,
                    ) = (
                        librosa.beat.beat_track(
                            onset_envelope=onset_envelope,
                            sr=sr,
                            hop_length=HOP_LENGTH,
                        )
                    )

                    beat_times = (
                        librosa.frames_to_time(
                            beat_frames,
                            sr=sr,
                            hop_length=HOP_LENGTH,
                        )
                    )

                    if len(
                        beat_times
                    ) == 0:
                        raise RuntimeError(
                            "No reliable beats were "
                            "detected in this track."
                        )

                    accent_curve = (
                        build_accent_curve(
                            y=y,
                            sr=sr,
                            onset_envelope=onset_envelope,
                        )
                    )

                    (
                        selected_beats,
                        phrase_states,
                    ) = (
                        select_phrase_locked_beats(
                            beat_times=beat_times,
                            accent_curve=accent_curve,
                            sr=sr,
                            interval=BEAT_INTERVAL,
                        )
                    )

                    status.write(
                        f"Detected "
                        f"{len(beat_times)} "
                        f"base beats"
                    )

                    if BEAT_INTERVAL == 1:
                        status.write(
                            "Using every detected beat"
                        )

                    else:
                        status.write(
                            f"Selected "
                            f"{len(selected_beats)} "
                            f"phrase-locked edit points"
                        )

                        if len(
                            phrase_states
                        ) > 0:
                            initial_offset = (
                                phrase_states[
                                    0
                                ][
                                    "offset"
                                ]
                            )

                            offset_ms = int(
                                round(
                                    initial_offset
                                    * 1000
                                )
                            )

                            status.write(
                                f"Opening rhythm alignment: "
                                f"{offset_ms:+d} ms"
                            )

                            number_of_changes = 0

                            for state_index in range(
                                1,
                                len(
                                    phrase_states
                                ),
                            ):
                                current_state = (
                                    phrase_states[
                                        state_index
                                    ]
                                )

                                previous_state = (
                                    phrase_states[
                                        state_index - 1
                                    ]
                                )

                                if (
                                    current_state[
                                        "phase"
                                    ]
                                    != previous_state[
                                        "phase"
                                    ]
                                    or abs(
                                        current_state[
                                            "offset"
                                        ]
                                        - previous_state[
                                            "offset"
                                        ]
                                    )
                                    > 0.01
                                ):
                                    number_of_changes += 1

                            status.write(
                                f"Rhythm alignment changes: "
                                f"{number_of_changes}"
                            )

                    del y
                    del onset_envelope
                    del accent_curve

                    total_video_frames = (
                        math.ceil(
                            total_duration
                            * FPS_VALUE
                        )
                    )

                    scene_change_frames = []

                    for beat in selected_beats:
                        frame_number = (
                            beat_time_to_frame(
                                float(
                                    beat
                                ),
                                FPS_VALUE,
                            )
                        )

                        if (
                            0
                            < frame_number
                            < total_video_frames
                        ):
                            scene_change_frames.append(
                                frame_number
                            )

                    scene_change_frames = (
                        sorted(
                            set(
                                scene_change_frames
                            )
                        )
                    )

                    status.write(
                        f"Creating "
                        f"{len(scene_change_frames)} "
                        f"scene changes"
                    )

                    status.write(
                        "4/5 · Building frame-accurate video"
                    )

                    raw_video_path = (
                        work_dir
                        / "reference.rgb"
                    )

                    change_index = 0
                    pattern_index = -1

                    with open(
                        raw_video_path,
                        "wb",
                    ) as raw_video:
                        for frame_number in range(
                            total_video_frames
                        ):
                            while (
                                change_index
                                < len(
                                    scene_change_frames
                                )
                                and frame_number
                                >= scene_change_frames[
                                    change_index
                                ]
                            ):
                                pattern_index += 1
                                change_index += 1

                            if pattern_index < 0:
                                frame_data = BLACK_FRAME
                            elif (
                                pattern_index
                                % 2
                                == 0
                            ):
                                frame_data = PATTERN_A_FRAME
                            else:
                                frame_data = PATTERN_B_FRAME

                            raw_video.write(
                                frame_data
                            )

                    status.write(
                        "5/5 · Rendering video"
                    )

                    output_path = (
                        work_dir
                        / "detectthebeat_video.mp4"
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
                            (
                                f"{INTERNAL_WIDTH}"
                                f"x{INTERNAL_HEIGHT}"
                            ),
                            "-r",
                            FPS_FFMPEG,
                            "-i",
                            str(
                                raw_video_path
                            ),
                            "-i",
                            str(
                                audio_path
                            ),
                            "-map",
                            "0:v:0",
                            "-map",
                            "1:a:0",
                            "-vf",
                            (
                                f"scale="
                                f"{VIDEO_WIDTH}:"
                                f"{VIDEO_HEIGHT}:"
                                f"flags=neighbor,"
                                f"format=yuv420p"
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
                            str(
                                output_path
                            ),
                        ]
                    )

                    video_bytes = (
                        output_path
                        .read_bytes()
                    )

                status.update(
                    label="Video ready!",
                    state="complete",
                    expanded=False,
                )

                st.success(
                    f"Created "
                    f"{len(scene_change_frames)} "
                    f"scene changes."
                )

                st.video(
                    video_bytes
                )

                st.download_button(
                    label="📥 Download Video",
                    data=video_bytes,
                    file_name=download_filename,
                    mime="video/mp4",
                    on_click="ignore",
                )

                st.info(
                    "Import the MP4 into your editing "
                    "software and use Scene Edit Detection "
                    "to create cuts at the checkerboard "
                    "changes. Use your original audio file "
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

                st.error(
                    "Video generation failed."
                )

                st.code(
                    str(
                        error
                    )
                )
