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
SWITCH_MARGIN = 0.10

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
            path=audio_path
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

    audio_path.write_bytes(
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

    return (
        song_name
        or "Song"
    )


def make_output_filename(
    original_name,
    beat_choice,
):
    labels = {
        "Every beat": "EveryBeat",
        "Every 2 beats": "Every2Beats",
        "Every 4 beats": "Every4Beats",
    }

    return (
        f"{sanitize_song_name(original_name)}_"
        f"DetectTheBeat_"
        f"{labels[beat_choice]}.mp4"
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
        False,
    )
)

PATTERN_B_FRAME = (
    create_checkerboard_frame(
        INTERNAL_WIDTH,
        INTERNAL_HEIGHT,
        True,
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

    return np.clip(
        (
            values - low
        )
        / (
            high - low
        ),
        0.0,
        1.0,
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

    if (
        high
        <= low + 1e-9
    ):
        return np.zeros_like(
            values
        )

    return np.clip(
        (
            values - low
        )
        / (
            high - low
        ),
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

    if (
        len(values)
        == target_length
    ):
        return values

    if (
        len(values)
        > target_length
    ):
        return values[
            :target_length
        ]

    if len(values) == 0:
        return np.zeros(
            target_length
        )

    return np.concatenate(
        [
            values,
            np.full(
                target_length
                - len(values),
                values[-1],
            ),
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

    return np.convolve(
        values,
        np.ones(
            window
        )
        / window,
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

    smoothed = (
        moving_average(
            values,
            max(
                2,
                half // 4,
            ),
        )
    )

    recent = (
        moving_average(
            smoothed,
            half,
        )
    )

    previous = np.roll(
        recent,
        half,
    )

    trend = np.maximum(
        recent - previous,
        0,
    )

    trend[
        :half
    ] = 0

    return normalize_feature(
        trend
    )


def build_accent_curve(
    y,
    sr,
    onset_envelope,
):
    onset_curve = (
        normalize_feature(
            onset_envelope
        )
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

    mel_freqs = (
        librosa.mel_frequencies(
            n_mels=32,
            fmin=30,
            fmax=1000,
        )
    )

    bass_mask = (
        mel_freqs
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

    bass_curve = (
        normalize_feature(
            bass_curve
        )
    )

    rms_curve = (
        normalize_feature(
            librosa.feature.rms(
                y=y,
                frame_length=1024,
                hop_length=HOP_LENGTH,
            )[0]
        )
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

    return normalize_feature(
        0.55 * onset_curve
        + 0.30 * bass_curve
        + 0.15 * rms_curve
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
        HOP_LENGTH
        / sr
    )

    maximum_shift = (
        beat_period
        * 0.55
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
    return [
        beat_index
        for beat_index
        in range(
            block_start,
            block_end,
        )
        if (
            beat_index
            - block_start
        )
        % interval
        == phase
    ]


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
    strengths = []

    for beat_index in (
        get_phrase_selected_indices(
            block_start,
            block_end,
            phase,
            interval,
        )
    ):
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

        strengths.append(
            sample_curve_at_time(
                accent_curve,
                shifted_time,
                sr,
                radius=1,
            )
        )

    if not strengths:
        return -999.0

    strengths = np.asarray(
        strengths
    )

    score = (
        0.50
        * float(
            np.median(
                strengths
            )
        )
        + 0.25
        * float(
            np.mean(
                strengths
            )
        )
        + 0.25
        * float(
            np.percentile(
                strengths,
                25,
            )
        )
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
                    beat_times,
                    accent_curve,
                    sr,
                    block_start,
                    block_end,
                    interval,
                    phase,
                    float(
                        offset
                    ),
                    ignore_first_beats,
                )
            )

            if score > best_score:
                best_phase = phase
                best_offset = float(
                    offset
                )
                best_score = score

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
            ) = (
                find_best_phrase_state(
                    beat_times,
                    accent_curve,
                    sr,
                    block_start,
                    block_end,
                    interval,
                    offset_candidates,
                    ignore_first_beats,
                )
            )

        else:
            (
                best_phase,
                best_offset,
                best_score,
            ) = (
                find_best_phrase_state(
                    beat_times,
                    accent_curve,
                    sr,
                    block_start,
                    block_end,
                    interval,
                    offset_candidates,
                    0,
                )
            )

            previous_score = (
                score_phrase_state(
                    beat_times,
                    accent_curve,
                    sr,
                    block_start,
                    block_end,
                    interval,
                    previous_phase,
                    previous_offset,
                    0,
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

            phase_change_penalty = (
                0.04
                if (
                    best_phase
                    != previous_phase
                )
                else 0.0
            )

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
                chosen_phase = (
                    best_phase
                )

                chosen_offset = (
                    best_offset
                )

                chosen_score = (
                    best_score
                )

            else:
                chosen_phase = (
                    previous_phase
                )

                chosen_offset = (
                    previous_offset
                )

                chosen_score = (
                    previous_score
                )

        for beat_index in (
            get_phrase_selected_indices(
                block_start,
                block_end,
                chosen_phase,
                interval,
            )
        ):
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

            selected_records.append(
                {
                    "time":
                        shifted_time,
                    "strength":
                        sample_curve_at_time(
                            accent_curve,
                            shifted_time,
                            sr,
                            radius=1,
                        ),
                    "beat_index":
                        beat_index,
                    "block":
                        block_number,
                }
            )

        phrase_states.append(
            {
                "block":
                    block_number,
                "start_beat":
                    block_start,
                "end_beat":
                    block_end,
                "phase":
                    chosen_phase,
                "offset":
                    chosen_offset,
                "score":
                    chosen_score,
            }
        )

        previous_phase = (
            chosen_phase
        )

        previous_offset = (
            chosen_offset
        )

        block_number += 1

    selected_records = sorted(
        selected_records,
        key=lambda item:
            item["time"],
    )

    if not selected_records:
        return (
            [],
            phrase_states,
        )

    minimum_gap = (
        beat_period
        * interval
        * 0.55
    )

    cleaned = []

    for record in (
        selected_records
    ):
        if not cleaned:
            cleaned.append(
                record
            )
            continue

        gap = (
            record["time"]
            - cleaned[-1][
                "time"
            ]
        )

        if gap < minimum_gap:
            if (
                record[
                    "strength"
                ]
                > cleaned[-1][
                    "strength"
                ]
            ):
                cleaned[-1] = (
                    record
                )

        else:
            cleaned.append(
                record
            )

    return (
        [
            item["time"]
            for item
            in cleaned
        ],
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
            frequencies
            >= 30
        )
        & (
            frequencies
            <= 220
        )
    )

    high_mask = (
        (
            frequencies
            >= 600
        )
        & (
            frequencies
            <= 8000
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

    return np.clip(
        (
            matrix - center
        )
        / scale,
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

    onset = (
        normalize_feature(
            onset_envelope
        )
    )

    target_length = len(
        onset
    )

    rms_raw = (
        match_feature_length(
            librosa.feature.rms(
                y=y,
                frame_length=2048,
                hop_length=HOP_LENGTH,
            )[0],
            target_length,
        )
    )

    rms = (
        normalize_feature(
            rms_raw
        )
    )

    brightness_raw = (
        match_feature_length(
            librosa.feature.spectral_centroid(
                y=y,
                sr=sr,
                n_fft=2048,
                hop_length=HOP_LENGTH,
            )[0],
            target_length,
        )
    )

    brightness = (
        normalize_feature(
            brightness_raw
        )
    )

    (
        bass_hit,
        high_hit,
    ) = (
        calculate_band_activity(
            y,
            sr,
        )
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

    if (
        chroma.shape[1]
        > target_length
    ):
        chroma = (
            chroma[
                :,
                :target_length
            ]
        )

    elif (
        chroma.shape[1]
        < target_length
    ):
        chroma = np.pad(
            chroma,
            (
                (
                    0,
                    0,
                ),
                (
                    0,
                    target_length
                    - chroma.shape[1],
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

    if (
        mfcc.shape[1]
        > target_length
    ):
        mfcc = (
            mfcc[
                :,
                :target_length
            ]
        )

    elif (
        mfcc.shape[1]
        < target_length
    ):
        mfcc = np.pad(
            mfcc,
            (
                (
                    0,
                    0,
                ),
                (
                    0,
                    target_length
                    - mfcc.shape[1],
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
        normalize_feature(
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
    )

    build_score = (
        normalize_feature(
            0.40
            * energy_rise
            + 0.22
            * brightness_rise
            + 0.25
            * density_rise
            + 0.13
            * percussive_activity
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
        "times":
            times,
        "onset":
            onset,
        "onset_raw":
            onset_envelope,
        "rms":
            rms,
        "rms_raw":
            rms_raw,
        "brightness":
            brightness,
        "brightness_raw":
            brightness_raw,
        "bass_hit":
            bass_hit,
        "high_hit":
            high_hit,
        "onset_density":
            onset_density,
        "build_score":
            build_score,
        "beat_times":
            beat_times,
        "beat_pulse":
            beat_pulse,
        "chroma":
            chroma,
        "mfcc":
            mfcc,
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
        beat_period
        * 0.45,
    )

    return float(
        np.clip(
            1.0
            - distance
            / tolerance,
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
        sr
        / HOP_LENGTH
    )

    short_window = int(
        0.7
        * fps
    )

    context_window = int(
        2.5
        * fps
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

    if len(
        pre_values
    ):
        pre_energy = float(
            np.mean(
                pre_values
            )
        )

    else:
        pre_energy = 0.0

    context_values = rms[
        context_start:
        max(
            context_start + 1,
            pause_start,
        )
    ]

    if len(
        context_values
    ) == 0:
        return 0.0

    context_energy = float(
        np.mean(
            context_values
        )
    )

    score = (
        (
            context_energy
            - pre_energy
        )
        * 3.0
    )

    if pre_energy < 0.20:
        score += 0.20

    return float(
        np.clip(
            score,
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

            best_score = max(
                best_score,
                build["Score"]
                * max(
                    0.0,
                    proximity,
                ),
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
    distinctive = max(
        event["Bass"],
        event["High / tonal"],
        event["Onset"],
    )

    context = max(
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
        * context
        + 0.15
        * distinctive
    )

    if (
        event[
            "High / tonal"
        ]
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
        event[
            "Pause before"
        ]
        >= 0.45
        and event[
            "Onset"
        ]
        >= 0.35
    ):
        labels.append(
            "Post-pause hit"
        )

    if (
        event[
            "Build before"
        ]
        >= 0.55
        and event[
            "Onset"
        ]
        >= 0.35
    ):
        labels.append(
            "Post-build hit"
        )

    if (
        event["Onset"]
        >= 0.78
    ):
        labels.append(
            "Very strong hit"
        )

    elif (
        event["Onset"]
        >= 0.48
    ):
        labels.append(
            "Strong hit"
        )

    else:
        labels.append(
            "Accent"
        )

    if (
        event[
            "High / tonal"
        ]
        >= 0.55
        and event[
            "High / tonal"
        ]
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
        > event[
            "High / tonal"
        ]
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
    event[
        "Opportunity"
    ] = (
        calculate_event_opportunity(
            event
        )
    )

    event[
        "Tier"
    ] = (
        get_event_tier(
            event[
                "Opportunity"
            ]
        )
    )

    event[
        "Event"
    ] = (
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
        key=lambda item:
            item["Time"],
    )

    clusters = []
    current = [
        events[0]
    ]

    cluster_start = (
        events[0][
            "Time"
        ]
    )

    for event in events[1:]:
        if (
            event["Time"]
            - cluster_start
            <= cluster_seconds
        ):
            current.append(
                event
            )

        else:
            clusters.append(
                current
            )

            current = [
                event
            ]

            cluster_start = (
                event["Time"]
            )

    clusters.append(
        current
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

    return (
        merged_events,
        len(events)
        - len(
            merged_events
        ),
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
        key=lambda item:
            item["Time"],
    )

    last_kept = None
    suppressed = 0

    for event in events:
        is_pause = (
            event[
                "Pause before"
            ]
            >= 0.45
            and event[
                "Onset"
            ]
            >= 0.35
        )

        if not is_pause:
            continue

        if last_kept is None:
            last_kept = (
                event["Time"]
            )
            continue

        if (
            event["Time"]
            - last_kept
            <= suppression_seconds
        ):
            event[
                "Pause before"
            ] = 0.0

            refresh_event_ranking(
                event
            )

            suppressed += 1

        else:
            last_kept = (
                event["Time"]
            )

    return (
        events,
        suppressed,
    )


def calculate_local_prominence(
    events,
    window_seconds=0.65,
):
    if not events:
        return []

    times = np.asarray(
        [
            event[
                "Time"
            ]
            for event
            in events
        ],
        dtype=float,
    )

    strengths = np.asarray(
        [
            event[
                "Hit strength"
            ]
            for event
            in events
        ],
        dtype=float,
    )

    result = []

    for (
        index,
        event_time,
    ) in enumerate(
        times
    ):
        local = strengths[
            np.abs(
                times
                - event_time
            )
            <= window_seconds
        ]

        if len(local):
            local_max = float(
                np.max(
                    local
                )
            )

        else:
            local_max = 0.0

        if local_max > 0:
            value = (
                strengths[
                    index
                ]
                / local_max
            )

        else:
            value = 0.0

        result.append(
            float(
                np.clip(
                    value,
                    0.0,
                    1.0,
                )
            )
        )

    return result


def build_editorial_candidates(
    events,
):
    if not events:
        return []

    events = sorted(
        events,
        key=lambda item:
            item["Time"],
    )

    prominence_values = (
        calculate_local_prominence(
            events,
            0.65,
        )
    )

    candidates = []

    for (
        event,
        prominence,
    ) in zip(
        events,
        prominence_values,
    ):
        context = max(
            event[
                "Pause before"
            ],
            event[
                "Build before"
            ],
        )

        distinctive = max(
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
                * context
                + 0.08
                * distinctive,
                0.0,
                1.0,
            )
        )

        reasons = []

        if (
            event[
                "Pause before"
            ]
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
            event[
                "Build before"
            ]
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

        if (
            editorial_score
            >= 0.64
        ):
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
    onset = (
        analysis[
            "onset"
        ]
    )

    bass_hit = (
        analysis[
            "bass_hit"
        ]
    )

    high_hit = (
        analysis[
            "high_hit"
        ]
    )

    rms = (
        analysis[
            "rms"
        ]
    )

    beat_times = (
        analysis[
            "beat_times"
        ]
    )

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

    for frame in (
        peak_frames
    ):
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

        distinctive = max(
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
            onset_strength
            < 0.20
            and distinctive
            < 0.25
        ):
            continue

        event = {
            "Time":
                event_time,
            "Hit strength":
                hit_strength,
            "Beat alignment":
                beat_alignment,
            "Pause before":
                pause_score,
            "Build before":
                build_context,
            "Bass":
                bass_strength,
            "High / tonal":
                high_strength,
            "Onset":
                onset_strength,
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
    ) = (
        cluster_micro_hits(
            raw_events,
            0.12,
        )
    )

    (
        events,
        suppressed_pause_count,
    ) = (
        keep_first_post_pause_hit(
            events,
            1.75,
        )
    )

    return (
        sorted(
            events,
            key=lambda item:
                item["Time"],
        ),
        len(
            raw_events
        ),
        merged_count,
        suppressed_pause_count,
    )


def detect_build_regions(
    analysis,
    sr,
):
    build = (
        analysis[
            "build_score"
        ]
    )

    times = (
        analysis[
            "times"
        ]
    )

    rms = (
        analysis[
            "rms"
        ]
    )

    onset_density = (
        analysis[
            "onset_density"
        ]
    )

    brightness = (
        analysis[
            "brightness"
        ]
    )

    fps = (
        sr
        / HOP_LENGTH
    )

    mask = (
        build
        >= 0.68
    )

    max_gap_frames = max(
        1,
        int(
            round(
                0.30
                * fps
            )
        ),
    )

    active = np.flatnonzero(
        mask
    )

    candidate_regions = []

    if len(active):
        start = int(
            active[0]
        )

        previous = (
            start
        )

        for index in (
            active[1:]
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
                previous = (
                    index
                )

            else:
                candidate_regions.append(
                    (
                        start,
                        previous + 1,
                    )
                )

                start = (
                    index
                )

                previous = (
                    index
                )

        candidate_regions.append(
            (
                start,
                previous + 1,
            )
        )

    minimum_frames = max(
        2,
        int(
            round(
                1.50
                * fps
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

        early = slice(
            start,
            min(
                start + quarter,
                end,
            ),
        )

        late = slice(
            max(
                start,
                end - quarter,
            ),
            end,
        )

        energy_rise = float(
            np.mean(
                rms[
                    late
                ]
            )
            - np.mean(
                rms[
                    early
                ]
            )
        )

        density_rise = float(
            np.mean(
                onset_density[
                    late
                ]
            )
            - np.mean(
                onset_density[
                    early
                ]
            )
        )

        brightness_rise = float(
            np.mean(
                brightness[
                    late
                ]
            )
            - np.mean(
                brightness[
                    early
                ]
            )
        )

        rising_signals = sum(
            [
                energy_rise
                >= 0.08,
                density_rise
                >= 0.08,
                brightness_rise
                >= 0.06,
            ]
        )

        if rising_signals < 2:
            continue

        segment = (
            build[
                start:end
            ]
        )

        peak_index = (
            start
            + int(
                np.argmax(
                    segment
                )
            )
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
                "Start":
                    float(
                        times[
                            start
                        ]
                    ),
                "End":
                    float(
                        times[
                            min(
                                end - 1,
                                len(times) - 1,
                            )
                        ]
                    ),
                "Peak":
                    float(
                        times[
                            peak_index
                        ]
                    ),
                "Score":
                    region_score,
                "Energy rise":
                    energy_rise,
                "Density rise":
                    density_rise,
                "Brightness rise":
                    brightness_rise,
            }
        )

    return regions


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
    ) = (
        detect_song_events(
            analysis,
            sr,
            builds,
        )
    )

    editorial_candidates = (
        build_editorial_candidates(
            events
        )
    )

    beat_times = (
        analysis[
            "beat_times"
        ]
    )

    if len(
        beat_times
    ) > 1:
        tempo = (
            60.0
            / get_median_beat_period(
                beat_times
            )
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
        c1,
        c2,
        c3,
        c4,
    ) = st.columns(4)

    c1.metric(
        "Estimated tempo",
        (
            f"{tempo:.1f} BPM"
            if tempo > 0
            else "—"
        ),
    )

    c2.metric(
        "Detected beats",
        len(
            beat_times
        ),
    )

    c3.metric(
        "Raw events",
        len(
            events
        ),
    )

    c4.metric(
        "Editorial candidates",
        len(
            editorial_candidates
        ),
    )

    if (
        merged_micro_hits
        > 0
    ):
        st.caption(
            f"Cleaned "
            f"{merged_micro_hits} "
            f"duplicate micro-hits "
            f"from "
            f"{raw_event_count} "
            f"raw onset candidates."
        )

    if (
        suppressed_pause_hits
        > 0
    ):
        st.caption(
            f"Kept post-pause context on the first meaningful hit "
            f"and removed it from {suppressed_pause_hits} following hits."
        )

    st.subheader(
        "Energy"
    )

    st.line_chart(
        pd.DataFrame(
            {
                "Time":
                    analysis[
                        "times"
                    ],
                "Energy":
                    analysis[
                        "rms"
                    ],
            }
        ).set_index(
            "Time"
        ),
        height=220,
    )

    st.subheader(
        "Hits and musical accents"
    )

    st.line_chart(
        pd.DataFrame(
            {
                "Time":
                    analysis[
                        "times"
                    ],
                "Overall onset":
                    analysis[
                        "onset"
                    ],
                "Bass hit":
                    analysis[
                        "bass_hit"
                    ],
                "High / tonal hit":
                    analysis[
                        "high_hit"
                    ],
                "Beat":
                    analysis[
                        "beat_pulse"
                    ],
            }
        ).set_index(
            "Time"
        ),
        height=300,
    )

    st.subheader(
        "Build-up analysis"
    )

    st.line_chart(
        pd.DataFrame(
            {
                "Time":
                    analysis[
                        "times"
                    ],
                "Build score":
                    analysis[
                        "build_score"
                    ],
                "Onset density":
                    analysis[
                        "onset_density"
                    ],
                "Brightness":
                    analysis[
                        "brightness"
                    ],
            }
        ).set_index(
            "Time"
        ),
        height=260,
    )

    st.subheader(
        "Possible builds"
    )

    if not builds:
        st.write(
            "No clear sustained build regions detected."
        )

    else:
        build_rows = []

        for build in builds:
            build_rows.append(
                {
                    "Start":
                        format_time(
                            build[
                                "Start"
                            ]
                        ),
                    "End":
                        format_time(
                            build[
                                "End"
                            ]
                        ),
                    "Peak":
                        format_time(
                            build[
                                "Peak"
                            ]
                        ),
                    "Score":
                        round(
                            build[
                                "Score"
                            ],
                            2,
                        ),
                    "Energy rise":
                        round(
                            build[
                                "Energy rise"
                            ],
                            2,
                        ),
                    "Activity rise":
                        round(
                            build[
                                "Density rise"
                            ],
                            2,
                        ),
                    "Brightness rise":
                        round(
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
                "Song":
                    song_name,
                "Time":
                    format_time(
                        event[
                            "Time"
                        ]
                    ),
                "Seconds":
                    round(
                        event[
                            "Time"
                        ],
                        3,
                    ),
                "Tier":
                    event[
                        "Tier"
                    ],
                "Event":
                    event[
                        "Event"
                    ],
                "Candidate reason":
                    event[
                        "Candidate reason"
                    ],
                "Editorial score":
                    round(
                        event[
                            "Editorial score"
                        ],
                        2,
                    ),
                "Opportunity":
                    round(
                        event[
                            "Opportunity"
                        ],
                        2,
                    ),
                "Hit":
                    round(
                        event[
                            "Hit strength"
                        ],
                        2,
                    ),
                "Prominence":
                    round(
                        event[
                            "Local prominence"
                        ],
                        2,
                    ),
                "Beat":
                    round(
                        event[
                            "Beat alignment"
                        ],
                        2,
                    ),
                "Pause":
                    round(
                        event[
                            "Pause before"
                        ],
                        2,
                    ),
                "Build":
                    round(
                        event[
                            "Build before"
                        ],
                        2,
                    ),
                "Bass":
                    round(
                        event[
                            "Bass"
                        ],
                        2,
                    ),
                "High":
                    round(
                        event[
                            "High / tonal"
                        ],
                        2,
                    ),
            }
        )

    candidate_df = pd.DataFrame(
        candidate_rows
    )

    st.dataframe(
        candidate_df,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    if not candidate_df.empty:
        st.download_button(
            "Download analysis CSV",
            data=(
                candidate_df
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            ),
            file_name=(
                f"{song_name}_"
                f"DetectTheBeat_"
                f"SongAnalysis.csv"
            ),
            mime="text/csv",
            on_click="ignore",
        )


# ============================================================
# ANCHOR V4
# ============================================================


def safe_cosine_distance(
    vector_a,
    vector_b,
):
    a = np.asarray(
        vector_a,
        dtype=float,
    )

    b = np.asarray(
        vector_b,
        dtype=float,
    )

    if (
        a.size == 0
        or b.size == 0
    ):
        return 0.0

    denominator = (
        np.linalg.norm(
            a
        )
        * np.linalg.norm(
            b
        )
    )

    if denominator <= 1e-9:
        return 0.0

    similarity = float(
        np.dot(
            a,
            b,
        )
        / denominator
    )

    return float(
        np.clip(
            1.0
            - similarity,
            0.0,
            1.0,
        )
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

    left = (
        values[
            :-lag
        ]
        - np.mean(
            values[
                :-lag
            ]
        )
    )

    right = (
        values[
            lag:
        ]
        - np.mean(
            values[
                lag:
            ]
        )
    )

    denominator = (
        np.linalg.norm(
            left
        )
        * np.linalg.norm(
            right
        )
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


def calculate_structural_raw_features_scale(
    event_time,
    analysis,
    sr,
    beat_period,
    window_seconds,
    margin_seconds,
):
    fps = (
        sr
        / HOP_LENGTH
    )

    frame = int(
        round(
            event_time
            * fps
        )
    )

    compare_window = max(
        6,
        int(
            round(
                window_seconds
                * fps
            )
        ),
    )

    event_margin = max(
        2,
        int(
            round(
                margin_seconds
                * fps
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
                min(
                    0.60,
                    window_seconds
                    * 0.35,
                )
                * fps
            )
        ),
    )

    if (
        pre_end
        - pre_start
        < minimum_frames
        or post_end
        - post_start
        < minimum_frames
    ):
        return {
            "tonal":
                0.0,
            "timbre":
                0.0,
            "rhythm":
                0.0,
            "dynamics":
                0.0,
        }

    tonal = (
        safe_cosine_distance(
            feature_mean_window(
                analysis[
                    "chroma"
                ],
                pre_start,
                pre_end,
            ),
            feature_mean_window(
                analysis[
                    "chroma"
                ],
                post_start,
                post_end,
            ),
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

    timbre = float(
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
        * fps,
    )

    pre_onset = (
        analysis[
            "onset"
        ][
            pre_start:
            pre_end
        ]
    )

    post_onset = (
        analysis[
            "onset"
        ][
            post_start:
            post_end
        ]
    )

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

    rhythm = float(
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

    dynamics = float(
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
        "tonal":
            tonal,
        "timbre":
            timbre,
        "rhythm":
            rhythm,
        "dynamics":
            dynamics,
    }


def calculate_long_quiet_before(
    analysis,
    event_time,
    sr,
):
    fps = (
        sr
        / HOP_LENGTH
    )

    frame = int(
        round(
            event_time
            * fps
        )
    )

    immediate = max(
        4,
        int(
            round(
                1.4
                * fps
            )
        ),
    )

    earlier = max(
        8,
        int(
            round(
                3.0
                * fps
            )
        ),
    )

    immediate_start = max(
        0,
        frame
        - immediate,
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
    fps = (
        sr
        / HOP_LENGTH
    )

    frame = int(
        round(
            event_time
            * fps
        )
    )

    before_window = max(
        5,
        int(
            round(
                1.4
                * fps
            )
        ),
    )

    after_margin = max(
        2,
        int(
            round(
                0.16
                * fps
            )
        ),
    )

    after_window = max(
        5,
        int(
            round(
                1.35
                * fps
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
    fps = (
        sr
        / HOP_LENGTH
    )

    frame = int(
        round(
            event_time
            * fps
        )
    )

    pre_window = max(
        8,
        int(
            round(
                2.5
                * fps
            )
        ),
    )

    post_margin = max(
        2,
        int(
            round(
                0.15
                * fps
            )
        ),
    )

    post_window = max(
        5,
        int(
            round(
                1.25
                * fps
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

    return float(
        np.clip(
            0.45
            * pre_activity
            + 0.55
            * np.clip(
                drop
                * 2.2,
                0.0,
                1.0,
            ),
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
            event[
                "Time"
            ]
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

    for event_time in times:
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

        if len(
            earlier
        ) == 0:
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

        if len(
            later
        ) == 0:
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

        before_scores.append(
            float(
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
        )

        after_scores.append(
            float(
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
        )

    return (
        before_scores,
        after_scores,
    )


def augment_editorial_candidates_with_opening_events(
    editorial_candidates,
    events,
    opening_seconds=3.5,
):
    candidates = [
        dict(
            event
        )
        for event
        in editorial_candidates
    ]

    if not events:
        return sorted(
            candidates,
            key=lambda item:
                item["Time"],
        )

    raw_prominence = (
        calculate_local_prominence(
            events,
            0.65,
        )
    )

    existing_times = [
        event[
            "Time"
        ]
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

        if event[
            "Time"
        ] < 0.10:
            continue

        if (
            event[
                "Hit strength"
            ]
            < 0.34
            and event[
                "Onset"
            ]
            < 0.38
        ):
            continue

        if existing_times:
            nearest = min(
                abs(
                    event[
                        "Time"
                    ]
                    - t
                )
                for t
                in existing_times
            )

            if nearest <= 0.08:
                continue

        context = max(
            event[
                "Pause before"
            ],
            event[
                "Build before"
            ],
        )

        distinctive = max(
            event[
                "Bass"
            ],
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
                * context
                + 0.08
                * distinctive,
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
        ] = (
            editorial_score
        )

        opening_candidate[
            "Candidate reason"
        ] = (
            "Opening rhythm candidate"
        )

        candidates.append(
            opening_candidate
        )

        existing_times.append(
            event[
                "Time"
            ]
        )

    return sorted(
        candidates,
        key=lambda item:
            item["Time"],
    )


def local_ratio_rank(
    candidates,
    key,
    window_seconds,
):
    times = np.asarray(
        [
            event[
                "Time"
            ]
            for event
            in candidates
        ],
        dtype=float,
    )

    values = np.asarray(
        [
            event[
                key
            ]
            for event
            in candidates
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
        local = values[
            np.abs(
                times
                - event_time
            )
            <= window_seconds
        ]

        if len(
            local
        ) == 0:
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

        if local_max > 1e-9:
            ratio = (
                values[
                    index
                ]
                / local_max
            )

        else:
            ratio = 0.0

        rank = float(
            np.mean(
                local
                <= values[
                    index
                ]
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


def calculate_landmark_fields_v4(
    candidates,
):
    if not candidates:
        return []

    ordered = sorted(
        [
            dict(
                event
            )
            for event
            in candidates
        ],
        key=lambda item:
            item["Time"],
    )

    (
        local_ratio,
        local_rank,
    ) = (
        local_ratio_rank(
            ordered,
            "Local structural novelty",
            2.5,
        )
    )

    (
        macro_ratio,
        macro_rank,
    ) = (
        local_ratio_rank(
            ordered,
            "Macro structural novelty",
            6.5,
        )
    )

    (
        tonal_ratio,
        tonal_rank,
    ) = (
        local_ratio_rank(
            ordered,
            "Local tonal change",
            3.0,
        )
    )

    (
        hit_ratio,
        hit_rank,
    ) = (
        local_ratio_rank(
            ordered,
            "Hit strength",
            2.8,
        )
    )

    for (
        i,
        event,
    ) in enumerate(
        ordered
    ):
        event[
            "Local structural ratio"
        ] = local_ratio[
            i
        ]

        event[
            "Local structural rank"
        ] = local_rank[
            i
        ]

        event[
            "Macro structural ratio"
        ] = macro_ratio[
            i
        ]

        event[
            "Macro structural rank"
        ] = macro_rank[
            i
        ]

        event[
            "Tonal local ratio"
        ] = tonal_ratio[
            i
        ]

        event[
            "Tonal local rank"
        ] = tonal_rank[
            i
        ]

        event[
            "Wide hit ratio"
        ] = hit_ratio[
            i
        ]

        event[
            "Wide hit rank"
        ] = hit_rank[
            i
        ]

        event[
            "Local structural landmark"
        ] = float(
            np.clip(
                0.55
                * local_ratio[
                    i
                ]
                + 0.45
                * local_rank[
                    i
                ],
                0.0,
                1.0,
            )
        )

        event[
            "Macro structural landmark"
        ] = float(
            np.clip(
                0.58
                * macro_ratio[
                    i
                ]
                + 0.42
                * macro_rank[
                    i
                ],
                0.0,
                1.0,
            )
        )

        event[
            "Tonal landmark"
        ] = float(
            np.clip(
                0.55
                * tonal_ratio[
                    i
                ]
                + 0.45
                * tonal_rank[
                    i
                ],
                0.0,
                1.0,
            )
        )

        event[
            "Wide hit landmark"
        ] = float(
            np.clip(
                0.55
                * hit_ratio[
                    i
                ]
                + 0.45
                * hit_rank[
                    i
                ],
                0.0,
                1.0,
            )
        )

    return ordered


def calculate_anchor_v4_candidates(
    editorial_candidates,
    analysis,
    sr,
):
    if not editorial_candidates:
        return []

    candidates = sorted(
        [
            dict(
                event
            )
            for event
            in editorial_candidates
        ],
        key=lambda item:
            item["Time"],
    )

    beat_times = (
        analysis[
            "beat_times"
        ]
    )

    beat_period = (
        get_median_beat_period(
            beat_times
        )
    )

    local_raw = []
    macro_raw = []

    for event in candidates:
        local_raw.append(
            calculate_structural_raw_features_scale(
                event[
                    "Time"
                ],
                analysis,
                sr,
                beat_period,
                window_seconds=1.25,
                margin_seconds=0.12,
            )
        )

        macro_raw.append(
            calculate_structural_raw_features_scale(
                event[
                    "Time"
                ],
                analysis,
                sr,
                beat_period,
                window_seconds=3.6,
                margin_seconds=0.28,
            )
        )

    local_tonal = (
        robust_relative_normalize(
            [
                x[
                    "tonal"
                ]
                for x
                in local_raw
            ]
        )
    )

    local_timbre = (
        robust_relative_normalize(
            [
                x[
                    "timbre"
                ]
                for x
                in local_raw
            ]
        )
    )

    local_rhythm = (
        robust_relative_normalize(
            [
                x[
                    "rhythm"
                ]
                for x
                in local_raw
            ]
        )
    )

    local_dynamics = (
        robust_relative_normalize(
            [
                x[
                    "dynamics"
                ]
                for x
                in local_raw
            ]
        )
    )

    macro_tonal = (
        robust_relative_normalize(
            [
                x[
                    "tonal"
                ]
                for x
                in macro_raw
            ]
        )
    )

    macro_timbre = (
        robust_relative_normalize(
            [
                x[
                    "timbre"
                ]
                for x
                in macro_raw
            ]
        )
    )

    macro_rhythm = (
        robust_relative_normalize(
            [
                x[
                    "rhythm"
                ]
                for x
                in macro_raw
            ]
        )
    )

    macro_dynamics = (
        robust_relative_normalize(
            [
                x[
                    "dynamics"
                ]
                for x
                in macro_raw
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
        2.8,
        max(
            1.6,
            4.0
            * beat_period,
        ),
    )

    first_pass = []

    for (
        i,
        event,
    ) in enumerate(
        candidates
    ):
        local_novelty = float(
            np.clip(
                0.42
                * local_tonal[
                    i
                ]
                + 0.24
                * local_timbre[
                    i
                ]
                + 0.18
                * local_rhythm[
                    i
                ]
                + 0.16
                * local_dynamics[
                    i
                ],
                0.0,
                1.0,
            )
        )

        macro_novelty = float(
            np.clip(
                0.16
                * macro_tonal[
                    i
                ]
                + 0.32
                * macro_timbre[
                    i
                ]
                + 0.30
                * macro_rhythm[
                    i
                ]
                + 0.22
                * macro_dynamics[
                    i
                ],
                0.0,
                1.0,
            )
        )

        if (
            event["Time"]
            < opening_suppression_end
        ):
            scale = float(
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

            local_novelty *= (
                scale
            )

            macro_novelty *= (
                scale
                * 0.65
            )

            lt = (
                float(
                    local_tonal[
                        i
                    ]
                )
                * scale
            )

            lti = (
                float(
                    local_timbre[
                        i
                    ]
                )
                * scale
            )

            lr = (
                float(
                    local_rhythm[
                        i
                    ]
                )
                * scale
            )

            ld = (
                float(
                    local_dynamics[
                        i
                    ]
                )
                * scale
            )

            mt = (
                float(
                    macro_tonal[
                        i
                    ]
                )
                * scale
                * 0.65
            )

            mti = (
                float(
                    macro_timbre[
                        i
                    ]
                )
                * scale
                * 0.65
            )

            mr = (
                float(
                    macro_rhythm[
                        i
                    ]
                )
                * scale
                * 0.65
            )

            md = (
                float(
                    macro_dynamics[
                        i
                    ]
                )
                * scale
                * 0.65
            )

        else:
            (
                lt,
                lti,
                lr,
                ld,
            ) = map(
                float,
                [
                    local_tonal[
                        i
                    ],
                    local_timbre[
                        i
                    ],
                    local_rhythm[
                        i
                    ],
                    local_dynamics[
                        i
                    ],
                ],
            )

            (
                mt,
                mti,
                mr,
                md,
            ) = map(
                float,
                [
                    macro_tonal[
                        i
                    ],
                    macro_timbre[
                        i
                    ],
                    macro_rhythm[
                        i
                    ],
                    macro_dynamics[
                        i
                    ],
                ],
            )

        enriched = dict(
            event
        )

        enriched.update(
            {
                "Local structural novelty":
                    local_novelty,
                "Macro structural novelty":
                    macro_novelty,
                "Local tonal change":
                    lt,
                "Local timbre change":
                    lti,
                "Local rhythm change":
                    lr,
                "Local dynamics change":
                    ld,
                "Macro tonal change":
                    mt,
                "Macro timbre change":
                    mti,
                "Macro rhythm change":
                    mr,
                "Macro dynamics change":
                    md,
                "Long quiet before":
                    calculate_long_quiet_before(
                        analysis,
                        event[
                            "Time"
                        ],
                        sr,
                    ),
                "Quiet after":
                    calculate_quiet_after_hit(
                        analysis,
                        event[
                            "Time"
                        ],
                        sr,
                    ),
                "Burst payoff":
                    calculate_burst_payoff(
                        analysis,
                        event[
                            "Time"
                        ],
                        sr,
                    ),
                "Before gap":
                    before_gap_scores[
                        i
                    ],
                "After gap":
                    after_gap_scores[
                        i
                    ],
            }
        )

        first_pass.append(
            enriched
        )

    first_pass = (
        calculate_landmark_fields_v4(
            first_pass
        )
    )

    output = []

    for event in (
        first_pass
    ):
        context_before = max(
            event[
                "Pause before"
            ],
            event[
                "Long quiet before"
            ],
        )

        distinctive = max(
            event[
                "Bass"
            ],
            event[
                "High / tonal"
            ],
        )

        wide_gap = max(
            event[
                "Before gap"
            ],
            event[
                "After gap"
            ],
        )

        musical_quality = float(
            np.clip(
                0.26
                * event[
                    "Hit strength"
                ]
                + 0.18
                * event[
                    "Local prominence"
                ]
                + 0.14
                * event[
                    "Beat alignment"
                ]
                + 0.10
                * distinctive
                + 0.12
                * event[
                    "Editorial score"
                ]
                + 0.10
                * event[
                    "Wide hit landmark"
                ]
                + 0.10
                * event[
                    "Local structural landmark"
                ],
                0.0,
                1.0,
            )
        )

        macro_change_path = float(
            np.clip(
                0.38
                * event[
                    "Macro structural novelty"
                ]
                + 0.26
                * event[
                    "Macro structural landmark"
                ]
                + 0.10
                * event[
                    "Local structural novelty"
                ]
                + 0.08
                * event[
                    "Hit strength"
                ]
                + 0.06
                * event[
                    "Beat alignment"
                ]
                + 0.06
                * wide_gap
                + 0.06
                * event[
                    "Wide hit landmark"
                ],
                0.0,
                1.0,
            )
        )

        tonal_phrase_path = float(
            np.clip(
                0.26
                * event[
                    "Tonal landmark"
                ]
                + 0.18
                * event[
                    "Local tonal change"
                ]
                + 0.16
                * event[
                    "Beat alignment"
                ]
                + 0.14
                * event[
                    "Hit strength"
                ]
                + 0.10
                * event[
                    "Wide hit landmark"
                ]
                + 0.08
                * event[
                    "Macro structural landmark"
                ]
                + 0.08
                * wide_gap,
                0.0,
                1.0,
            )
        )

        release_path = float(
            np.clip(
                0.34
                * context_before
                + 0.17
                * event[
                    "Hit strength"
                ]
                + 0.10
                * event[
                    "Local prominence"
                ]
                + 0.07
                * event[
                    "Beat alignment"
                ]
                + 0.08
                * event[
                    "Before gap"
                ]
                + 0.08
                * event[
                    "Macro structural landmark"
                ]
                + 0.08
                * event[
                    "Local structural landmark"
                ]
                + 0.08
                * event[
                    "Wide hit landmark"
                ],
                0.0,
                1.0,
            )
        )

        build_path = float(
            np.clip(
                0.36
                * event[
                    "Build before"
                ]
                + 0.17
                * event[
                    "Hit strength"
                ]
                + 0.09
                * event[
                    "Local prominence"
                ]
                + 0.06
                * event[
                    "Beat alignment"
                ]
                + 0.08
                * event[
                    "Macro structural landmark"
                ]
                + 0.08
                * event[
                    "Local structural landmark"
                ]
                + 0.08
                * event[
                    "Before gap"
                ]
                + 0.08
                * event[
                    "Wide hit landmark"
                ],
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
                + 0.24
                * event[
                    "Burst payoff"
                ]
                + 0.15
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
                + 0.06
                * event[
                    "After gap"
                ]
                + 0.06
                * event[
                    "Macro structural landmark"
                ]
                + 0.06
                * event[
                    "Wide hit landmark"
                ],
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
                + 0.14
                * event[
                    "Local prominence"
                ]
                + 0.18
                * event[
                    "Wide hit landmark"
                ]
                + 0.10
                * event[
                    "Tonal landmark"
                ]
                + 0.06
                * event[
                    "Macro structural landmark"
                ],
                0.0,
                1.0,
            )
        )

        pathways = {
            "Macro structural change":
                macro_change_path,
            "Tonal phrase landmark":
                tonal_phrase_path,
            "Release after quiet":
                release_path,
            "Build payoff":
                build_path,
            "Phrase/burst ending":
                ending_path,
        }

        ordered_paths = sorted(
            pathways.items(),
            key=lambda item:
                item[1],
            reverse=True,
        )

        (
            top_path_name,
            top_path_score,
        ) = (
            ordered_paths[0]
        )

        evidence_votes = 0

        if (
            event[
                "Macro structural landmark"
            ]
            >= 0.68
            and event[
                "Macro structural novelty"
            ]
            >= 0.34
        ):
            evidence_votes += 1

        if (
            event[
                "Tonal landmark"
            ]
            >= 0.78
            and event[
                "Local tonal change"
            ]
            >= 0.42
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

        if wide_gap >= 0.60:
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
            and event[
                "Wide hit landmark"
            ]
            >= 0.70
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
            event[
                "Macro structural landmark"
            ]
            >= 0.72
        ):
            reasons.append(
                "Macro section landmark"
            )

        if (
            event[
                "Local structural landmark"
            ]
            >= 0.78
        ):
            reasons.append(
                "Strong local change"
            )

        if (
            event[
                "Tonal landmark"
            ]
            >= 0.82
        ):
            reasons.append(
                "Tonal phrase landmark"
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
            >= 0.80
        ):
            reasons.append(
                "Strong rhythmic support"
            )

        if anchor_score >= 0.80:
            tier = (
                "CORE"
            )

        elif anchor_score >= 0.68:
            tier = (
                "LIKELY"
            )

        elif anchor_score >= 0.58:
            tier = (
                "POSSIBLE"
            )

        else:
            tier = (
                "SUPPORT"
            )

        enriched = dict(
            event
        )

        enriched.update(
            {
                "Anchor score":
                    anchor_score,
                "Anchor tier":
                    tier,
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
                "Macro change path":
                    macro_change_path,
                "Tonal phrase path":
                    tonal_phrase_path,
                "Release path":
                    release_path,
                "Build path":
                    build_path,
                "Ending path":
                    ending_path,
                "Rhythmic support":
                    rhythmic_support,
                "Opening grid alignment":
                    0.0,
                "Opening accent":
                    0.0,
                "Opening score":
                    0.0,
                "Opening grid offset":
                    0.0,
            }
        )

        output.append(
            enriched
        )

    return output


def calculate_opening_grid_v4(
    analysis,
    accent_curve,
    sr,
):
    beat_times = (
        analysis[
            "beat_times"
        ]
    )

    beat_period = (
        get_median_beat_period(
            beat_times
        )
    )

    if (
        len(beat_times)
        < 4
        or beat_period
        <= 0
    ):
        return {
            "offset":
                0.0,
            "aligned_beats":
                [
                    float(x)
                    for x
                    in beat_times
                ],
            "score":
                0.0,
        }

    block_end = min(
        len(
            beat_times
        ),
        12,
    )

    offset_candidates = (
        create_offset_candidates(
            beat_period,
            sr,
        )
    )

    (
        _,
        best_offset,
        best_score,
    ) = (
        find_best_phrase_state(
            beat_times=beat_times,
            accent_curve=accent_curve,
            sr=sr,
            block_start=0,
            block_end=block_end,
            interval=1,
            offset_candidates=offset_candidates,
            ignore_first_beats=min(
                2,
                max(
                    0,
                    block_end - 2,
                ),
            ),
        )
    )

    aligned = [
        float(t)
        + best_offset
        for t
        in beat_times[
            :block_end
        ]
        if (
            float(t)
            + best_offset
            > 0
        )
    ]

    return {
        "offset":
            best_offset,
        "aligned_beats":
            aligned,
        "score":
            best_score,
    }


def choose_opening_anchor_v4(
    preview,
    anchor_candidates,
    analysis,
    accent_curve,
    sr,
):
    if not anchor_candidates:
        return preview

    grid = (
        calculate_opening_grid_v4(
            analysis,
            accent_curve,
            sr,
        )
    )

    aligned_beats = (
        grid[
            "aligned_beats"
        ]
    )

    beat_period = (
        get_median_beat_period(
            analysis[
                "beat_times"
            ]
        )
    )

    opening_end = min(
        4.2,
        max(
            2.4,
            7.0
            * beat_period,
        ),
    )

    tolerance = max(
        0.09,
        min(
            0.20,
            0.34
            * beat_period,
        ),
    )

    opening_candidates = [
        dict(
            event
        )
        for event
        in anchor_candidates
        if (
            0.10
            <= event[
                "Time"
            ]
            <= opening_end
        )
    ]

    if (
        not opening_candidates
        or not aligned_beats
    ):
        return preview

    scored = []

    for event in (
        opening_candidates
    ):
        distances = np.abs(
            np.asarray(
                aligned_beats
            )
            - event[
                "Time"
            ]
        )

        nearest_index = int(
            np.argmin(
                distances
            )
        )

        nearest_distance = float(
            distances[
                nearest_index
            ]
        )

        grid_alignment = float(
            np.clip(
                1.0
                - nearest_distance
                / tolerance,
                0.0,
                1.0,
            )
        )

        accent = (
            sample_curve_at_time(
                accent_curve,
                aligned_beats[
                    nearest_index
                ],
                sr,
                radius=1,
            )
        )

        opening_score = float(
            np.clip(
                0.38
                * grid_alignment
                + 0.22
                * accent
                + 0.18
                * event[
                    "Hit strength"
                ]
                + 0.12
                * event[
                    "Local prominence"
                ]
                + 0.10
                * event[
                    "Beat alignment"
                ],
                0.0,
                1.0,
            )
        )

        event[
            "Opening grid alignment"
        ] = (
            grid_alignment
        )

        event[
            "Opening accent"
        ] = (
            accent
        )

        event[
            "Opening score"
        ] = (
            opening_score
        )

        event[
            "Opening grid offset"
        ] = (
            grid[
                "offset"
            ]
        )

        scored.append(
            event
        )

    valid = [
        event
        for event
        in scored
        if (
            event[
                "Opening grid alignment"
            ]
            >= 0.62
            and event[
                "Opening accent"
            ]
            >= 0.38
            and event[
                "Hit strength"
            ]
            >= 0.36
        )
    ]

    if not valid:
        valid = sorted(
            scored,
            key=lambda item:
                item[
                    "Opening score"
                ],
            reverse=True,
        )[
            :4
        ]

    best_score = max(
        event[
            "Opening score"
        ]
        for event
        in valid
    )

    competitive = [
        event
        for event
        in valid
        if (
            event[
                "Opening score"
            ]
            >= best_score
            - 0.08
            and event[
                "Opening grid alignment"
            ]
            >= 0.58
        )
    ]

    chosen = min(
        competitive
        or valid,
        key=lambda item:
            item["Time"],
    )

    if (
        chosen[
            "Opening score"
        ]
        < 0.50
    ):
        return preview

    result = [
        dict(
            event
        )
        for event
        in preview
        if (
            event["Time"]
            > opening_end
        )
    ]

    chosen[
        "Anchor pathway"
    ] = (
        "Opening rhythm grid"
    )

    chosen[
        "Anchor reason"
    ] = (
        "Opening rhythm grid + "
        "Phrase-aligned pulse"
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
        key=lambda item:
            item["Time"],
    )


def collapse_anchor_alternatives_v4(
    anchor_candidates,
    beat_period,
):
    if not anchor_candidates:
        return []

    group_seconds = min(
        0.78,
        max(
            0.44,
            1.25
            * beat_period,
        ),
    )

    ordered = sorted(
        anchor_candidates,
        key=lambda item:
            item["Time"],
    )

    groups = []
    current = [
        ordered[0]
    ]

    group_start = (
        ordered[0][
            "Time"
        ]
    )

    for event in (
        ordered[1:]
    ):
        if (
            event["Time"]
            - group_start
            <= group_seconds
        ):
            current.append(
                event
            )

        else:
            groups.append(
                current
            )

            current = [
                event
            ]

            group_start = (
                event["Time"]
            )

    groups.append(
        current
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
                    + 0.07
                    * event[
                        "Macro structural landmark"
                    ]
                    + 0.04
                    * event[
                        "Wide hit landmark"
                    ]
                    + 0.03
                    * event[
                        "Tonal landmark"
                    ],
                    0.0,
                    1.2,
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
            key=lambda item:
                item[1],
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
                - 0.02
                and event[
                    "Anchor pathway"
                ]
                == best_event[
                    "Anchor pathway"
                ]
                and abs(
                    event[
                        "Macro structural landmark"
                    ]
                    - best_event[
                        "Macro structural landmark"
                    ]
                )
                <= 0.07
            )
        ]

        if comparable:
            chosen = min(
                comparable,
                key=lambda item:
                    item["Time"],
            )

        else:
            chosen = (
                best_event
            )

        selected.append(
            dict(
                chosen
            )
        )

    return selected


def rescue_long_anchor_gaps_v4(
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

    if len(
        analysis[
            "times"
        ]
    ):
        duration = float(
            analysis[
                "times"
            ][-1]
        )

    else:
        duration = 0.0

    minimum_gap = max(
        5.0,
        10.0
        * beat_period,
    )

    selected = sorted(
        [
            dict(
                event
            )
            for event
            in preview
        ],
        key=lambda item:
            item["Time"],
    )

    boundaries = (
        [
            0.0
        ]
        + [
            event[
                "Time"
            ]
            for event
            in selected
        ]
        + [
            duration
        ]
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
                start
                + 0.80
                < event[
                    "Time"
                ]
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
                and (
                    event[
                        "Macro structural landmark"
                    ]
                    >= 0.42
                    or event[
                        "Tonal landmark"
                    ]
                    >= 0.70
                    or event[
                        "Wide hit rank"
                    ]
                    >= 0.78
                )
            )
        ]

        rescored = []

        for event in pool:
            rescue_score = float(
                np.clip(
                    0.26
                    * event[
                        "Rhythmic support"
                    ]
                    + 0.22
                    * event[
                        "Macro structural landmark"
                    ]
                    + 0.18
                    * event[
                        "Tonal landmark"
                    ]
                    + 0.16
                    * event[
                        "Wide hit landmark"
                    ]
                    + 0.10
                    * event[
                        "Beat alignment"
                    ]
                    + 0.08
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
                >= 0.63
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
            key=lambda item:
                item[1],
        )

        rescued = dict(
            chosen
        )

        rescued[
            "Anchor pathway"
        ] = (
            "Long-gap phrase landmark"
        )

        rescued[
            "Anchor reason"
        ] = (
            "Long-gap phrase landmark + "
            "Strong musical pulse"
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

    return sorted(
        selected
        + additions,
        key=lambda item:
            item["Time"],
    )


def build_anchor_v4_preview(
    anchor_candidates,
    analysis,
    accent_curve,
    sr,
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

    for event in (
        anchor_candidates
    ):
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
                "Macro change path"
            ]
            >= 0.64
            and event[
                "Macro structural landmark"
            ]
            >= 0.70
            and event[
                "Macro structural rank"
            ]
            >= 0.72
            and event[
                "Hit strength"
            ]
            >= 0.40
        ):
            keep = True

        if (
            event[
                "Tonal phrase path"
            ]
            >= 0.64
            and event[
                "Tonal landmark"
            ]
            >= 0.80
            and event[
                "Beat alignment"
            ]
            >= 0.66
            and event[
                "Hit strength"
            ]
            >= 0.58
            and (
                event[
                    "Macro structural landmark"
                ]
                >= 0.46
                or max(
                    event[
                        "Before gap"
                    ],
                    event[
                        "After gap"
                    ],
                )
                >= 0.52
                or event[
                    "Wide hit rank"
                ]
                >= 0.82
            )
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
                "Anchor score"
            ]
            >= 0.76
            and event[
                "Evidence votes"
            ]
            >= 3
        ):
            keep = True

        if keep:
            preview_pool.append(
                event
            )

    preview = (
        collapse_anchor_alternatives_v4(
            preview_pool,
            beat_period,
        )
    )

    preview = (
        rescue_long_anchor_gaps_v4(
            preview,
            anchor_candidates,
            analysis,
        )
    )

    preview = (
        collapse_anchor_alternatives_v4(
            preview,
            beat_period,
        )
    )

    preview = (
        choose_opening_anchor_v4(
            preview,
            anchor_candidates,
            analysis,
            accent_curve,
            sr,
        )
    )

    return preview


def anchor_v4_dataframe(
    anchor_candidates,
    song_name,
):
    rows = []

    for event in (
        anchor_candidates
    ):
        rows.append(
            {
                "Song":
                    song_name,
                "Time":
                    format_time(
                        event[
                            "Time"
                        ]
                    ),
                "Seconds":
                    round(
                        event[
                            "Time"
                        ],
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
                "Macro structural novelty":
                    round(
                        event[
                            "Macro structural novelty"
                        ],
                        3,
                    ),
                "Macro structural landmark":
                    round(
                        event[
                            "Macro structural landmark"
                        ],
                        3,
                    ),
                "Macro structural rank":
                    round(
                        event[
                            "Macro structural rank"
                        ],
                        3,
                    ),
                "Local structural novelty":
                    round(
                        event[
                            "Local structural novelty"
                        ],
                        3,
                    ),
                "Local structural landmark":
                    round(
                        event[
                            "Local structural landmark"
                        ],
                        3,
                    ),
                "Local structural rank":
                    round(
                        event[
                            "Local structural rank"
                        ],
                        3,
                    ),
                "Local tonal change":
                    round(
                        event[
                            "Local tonal change"
                        ],
                        3,
                    ),
                "Tonal landmark":
                    round(
                        event[
                            "Tonal landmark"
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
                "Macro timbre change":
                    round(
                        event[
                            "Macro timbre change"
                        ],
                        3,
                    ),
                "Macro rhythm change":
                    round(
                        event[
                            "Macro rhythm change"
                        ],
                        3,
                    ),
                "Macro dynamics change":
                    round(
                        event[
                            "Macro dynamics change"
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
                "Opening grid alignment":
                    round(
                        event.get(
                            "Opening grid alignment",
                            0.0,
                        ),
                        3,
                    ),
                "Opening accent":
                    round(
                        event.get(
                            "Opening accent",
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
                "Opening grid offset":
                    round(
                        event.get(
                            "Opening grid offset",
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
    output_name="anchor_v4.mp4",
):
    total_video_frames = (
        math.ceil(
            total_duration
            * fps_value
        )
    )

    scene_change_frames = sorted(
        set(
            beat_time_to_frame(
                float(
                    cut_time
                ),
                fps_value,
            )
            for cut_time
            in cut_times
            if (
                0
                < beat_time_to_frame(
                    float(
                        cut_time
                    ),
                    fps_value,
                )
                < total_video_frames
            )
        )
    )

    raw_video_path = (
        work_dir
        / "anchor_v4.rgb"
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


# ============================================================
# UI
# ============================================================


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

    if (
        uploaded_file
        is not None
        and st.button(
            "🔍 Analyze Song",
            type="primary",
        )
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
        "Anchor v4 is still a P1-first calibration test. "
        "V4 separates local musical changes from macro section changes, "
        "and the first anchor now uses the same phrase/grid alignment idea as Basic Beat."
    )

    st.subheader(
        "Anchor v4 settings"
    )

    st.caption(
        "Output: 16:9 · 1920×1080 · P1 anchor preview"
    )

    smart_fps_choice = st.radio(
        "Frame rate",
        [
            "25 fps",
            "23.976 fps",
        ],
        horizontal=True,
        key="anchor_v4_fps",
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
            "anchor-v4",
        )

        if (
            st.session_state.get(
                "anchor_v4_signature"
            )
            != smart_signature
        ):
            st.session_state.pop(
                "anchor_v4_result",
                None,
            )

        if st.button(
            "🎯 Analyze P1 anchors v4",
            type="primary",
        ):
            status = None

            try:
                status = st.status(
                    "Finding Anchor v4 P1 moments...",
                    expanded=True,
                )

                with tempfile.TemporaryDirectory() as work_dir:
                    work_dir = Path(
                        work_dir
                    )

                    status.write(
                        "1/9 · Preparing audio"
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
                        "2/9 · Reading waveform"
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
                        "3/9 · Detecting beats, hits, harmony and timbre"
                    )

                    analysis = (
                        build_song_analysis(
                            y,
                            sr,
                        )
                    )

                    accent_curve = (
                        build_accent_curve(
                            y,
                            sr,
                            analysis[
                                "onset_raw"
                            ],
                        )
                    )

                    status.write(
                        "4/9 · Building sensitive editorial candidate map"
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
                    ) = (
                        detect_song_events(
                            analysis,
                            sr,
                            builds,
                        )
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
                            opening_seconds=3.5,
                        )
                    )

                    status.write(
                        "5/9 · Measuring local vs macro musical change"
                    )

                    anchor_candidates = (
                        calculate_anchor_v4_candidates(
                            editorial_candidates,
                            analysis,
                            sr,
                        )
                    )

                    status.write(
                        "6/9 · Selecting P1 anchors without structural flooding"
                    )

                    anchor_preview = (
                        build_anchor_v4_preview(
                            anchor_candidates,
                            analysis,
                            accent_curve,
                            sr,
                        )
                    )

                    song_name = (
                        sanitize_song_name(
                            uploaded_file.name
                        )
                    )

                    full_df = (
                        anchor_v4_dataframe(
                            anchor_candidates,
                            song_name,
                        )
                    )

                    preview_df = (
                        anchor_v4_dataframe(
                            anchor_preview,
                            song_name,
                        )
                    )

                    status.write(
                        "7/9 · Building frame-accurate checkerboard preview"
                    )

                    preview_times = [
                        event[
                            "Time"
                        ]
                        for event
                        in anchor_preview
                    ]

                    (
                        video_bytes,
                        scene_frames,
                    ) = (
                        render_checkerboard_reference(
                            audio_path=audio_path,
                            cut_times=preview_times,
                            total_duration=total_duration,
                            fps_value=SMART_FPS_VALUE,
                            fps_ffmpeg=SMART_FPS_FFMPEG,
                            work_dir=work_dir,
                            output_name="anchor_v4.mp4",
                        )
                    )

                    status.write(
                        "8/9 · Calculating summary"
                    )

                    beat_period = (
                        get_median_beat_period(
                            analysis[
                                "beat_times"
                            ]
                        )
                    )

                    if beat_period > 0:
                        tempo = (
                            60.0
                            / beat_period
                        )

                    else:
                        tempo = 0.0

                    if total_duration > 0:
                        event_rate = (
                            len(
                                events
                            )
                            / total_duration
                        )

                    else:
                        event_rate = 0.0

                    opening_grid = (
                        calculate_opening_grid_v4(
                            analysis,
                            accent_curve,
                            sr,
                        )
                    )

                    status.write(
                        "9/9 · Finalizing Anchor v4 report"
                    )

                    video_filename = (
                        f"{song_name}_"
                        f"DetectTheBeat_"
                        f"AnchorV4.mp4"
                    )

                    analysis_filename = (
                        f"{song_name}_"
                        f"DetectTheBeat_"
                        f"AnchorV4_Analysis.csv"
                    )

                    preview_filename = (
                        f"{song_name}_"
                        f"DetectTheBeat_"
                        f"AnchorV4_Preview.csv"
                    )

                st.session_state[
                    "anchor_v4_signature"
                ] = (
                    smart_signature
                )

                st.session_state[
                    "anchor_v4_result"
                ] = {
                    "video_bytes":
                        video_bytes,
                    "video_filename":
                        video_filename,
                    "anchor_dataframe":
                        full_df,
                    "preview_dataframe":
                        preview_df,
                    "anchor_csv_filename":
                        analysis_filename,
                    "preview_csv_filename":
                        preview_filename,
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
                    "opening_offset_ms":
                        int(
                            round(
                                opening_grid[
                                    "offset"
                                ]
                                * 1000
                            )
                        ),
                    "merged_micro_hits":
                        merged_micro_hits,
                    "suppressed_pause_hits":
                        suppressed_pause_hits,
                }

                status.update(
                    label="Anchor v4 analysis ready",
                    state="complete",
                    expanded=False,
                )

            except Exception as error:
                if status is not None:
                    try:
                        status.update(
                            label="Anchor v4 analysis failed",
                            state="error",
                        )

                    except Exception:
                        pass

                st.error(
                    "Anchor v4 analysis failed."
                )

                st.code(
                    str(
                        error
                    )
                )

        smart_result = (
            st.session_state.get(
                "anchor_v4_result"
            )
        )

        if (
            smart_result
            is not None
            and st.session_state.get(
                "anchor_v4_signature"
            )
            == smart_signature
        ):
            st.success(
                f"Anchor v4 preview contains "
                f"{smart_result['preview_count']} "
                f"scene changes."
            )

            (
                c1,
                c2,
                c3,
                c4,
            ) = st.columns(4)

            c1.metric(
                "P1 preview",
                smart_result[
                    "preview_count"
                ],
            )

            c2.metric(
                "Candidates",
                smart_result[
                    "candidate_count"
                ],
            )

            c3.metric(
                "Tempo",
                (
                    f"{smart_result['tempo']:.1f} BPM"
                ),
            )

            c4.metric(
                "Opening grid",
                (
                    f"{smart_result['opening_offset_ms']:+d} ms"
                ),
            )

            st.caption(
                "V4 tests two things specifically: whether macro structure "
                "reduces V3's false P1 changes, and whether phrase-aligned "
                "opening detection fixes pickup-tone starts."
            )

            st.video(
                smart_result[
                    "video_bytes"
                ]
            )

            st.download_button(
                "📥 Download Anchor v4 preview video",
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
                "Anchor v4 P1 preview"
            )

            st.dataframe(
                smart_result[
                    "preview_dataframe"
                ],
                use_container_width=True,
                hide_index=True,
                height=520,
            )

            st.download_button(
                "Download Anchor v4 preview CSV",
                data=(
                    smart_result[
                        "preview_dataframe"
                    ]
                    .to_csv(
                        index=False
                    )
                    .encode(
                        "utf-8"
                    )
                ),
                file_name=smart_result[
                    "preview_csv_filename"
                ],
                mime="text/csv",
                on_click="ignore",
            )

            with st.expander(
                "Full Anchor v4 analysis"
            ):
                st.dataframe(
                    smart_result[
                        "anchor_dataframe"
                    ],
                    use_container_width=True,
                    hide_index=True,
                    height=520,
                )

                st.download_button(
                    "Download full Anchor v4 Analysis CSV",
                    data=(
                        smart_result[
                            "anchor_dataframe"
                        ]
                        .to_csv(
                            index=False
                        )
                        .encode(
                            "utf-8"
                        )
                    ),
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
        [
            "25 fps",
            "23.976 fps",
        ],
        horizontal=True,
    )

    beat_choice = st.radio(
        "Scene change",
        [
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
                            y,
                            sr,
                            onset_envelope,
                        )
                    )

                    (
                        selected_beats,
                        phrase_states,
                    ) = (
                        select_phrase_locked_beats(
                            beat_times,
                            accent_curve,
                            sr,
                            BEAT_INTERVAL,
                        )
                    )

                    status.write(
                        f"Detected "
                        f"{len(beat_times)} "
                        f"base beats"
                    )

                    if (
                        BEAT_INTERVAL
                        == 1
                    ):
                        status.write(
                            "Using every detected beat"
                        )

                    else:
                        status.write(
                            f"Selected "
                            f"{len(selected_beats)} "
                            f"phrase-locked edit points"
                        )

                        if phrase_states:
                            offset_ms = int(
                                round(
                                    phrase_states[
                                        0
                                    ][
                                        "offset"
                                    ]
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
                                current = (
                                    phrase_states[
                                        state_index
                                    ]
                                )

                                previous = (
                                    phrase_states[
                                        state_index - 1
                                    ]
                                )

                                if (
                                    current[
                                        "phase"
                                    ]
                                    != previous[
                                        "phase"
                                    ]
                                    or abs(
                                        current[
                                            "offset"
                                        ]
                                        - previous[
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

                    scene_change_frames = sorted(
                        set(
                            beat_time_to_frame(
                                float(
                                    beat
                                ),
                                FPS_VALUE,
                            )
                            for beat
                            in selected_beats
                            if (
                                0
                                < beat_time_to_frame(
                                    float(
                                        beat
                                    ),
                                    FPS_VALUE,
                                )
                                < total_video_frames
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
                    "📥 Download Video",
                    data=video_bytes,
                    file_name=download_filename,
                    mime="video/mp4",
                    on_click="ignore",
                )

                st.info(
                    "Import the MP4 into your editing software and use "
                    "Scene Edit Detection to create cuts at the checkerboard "
                    "changes. Use your original audio file for the final edit."
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
