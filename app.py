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
        raise RuntimeError(
            f"FFmpeg error:\n{result.stderr[-5000:]}"
        )

    return result


def get_audio_duration(audio_path):
    return float(
        librosa.get_duration(
            path=audio_path
        )
    )


def beat_time_to_frame(time_seconds, fps):
    return round(time_seconds * fps)


def format_time(seconds):
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60

    return f"{minutes}:{remaining:05.2f}"


def save_uploaded_audio(uploaded_file, work_dir):
    extension = Path(
        uploaded_file.name
    ).suffix.lower()

    if extension not in [
        ".mp3",
        ".wav",
        ".m4a"
    ]:
        extension = ".mp3"

    audio_path = (
        work_dir
        /
        f"input{extension}"
    )

    with open(
        audio_path,
        "wb"
    ) as file:
        file.write(
            uploaded_file.getvalue()
        )

    return audio_path


def create_analysis_wav(
    audio_path,
    work_dir
):
    analysis_path = (
        work_dir
        /
        "analysis.wav"
    )

    run_command([
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
    ])

    return analysis_path


# =========================================================
# FILENAME
# =========================================================

def make_output_filename(
    original_name,
    beat_choice
):
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


# =========================================================
# CHECKERBOARD VIDEO
# =========================================================

def create_solid_frame(
    width,
    height,
    color
):
    return bytes(color) * (
        width * height
    )


def create_checkerboard_frame(
    width,
    height,
    inverted=False
):
    block_size = 4
    pixels = bytearray()

    for y in range(height):
        for x in range(width):

            checker = (
                (
                    x // block_size
                )
                +
                (
                    y // block_size
                )
            ) % 2

            if inverted:
                checker = 1 - checker

            if checker == 0:
                pixels.extend(
                    (0, 0, 0)
                )

            else:
                pixels.extend(
                    (255, 255, 255)
                )

    return bytes(pixels)


BLACK_FRAME = create_solid_frame(
    INTERNAL_WIDTH,
    INTERNAL_HEIGHT,
    (0, 0, 0),
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


# =========================================================
# FEATURE HELPERS
# =========================================================

def normalize_feature(values):
    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) == 0:
        return values

    low = np.percentile(
        values,
        10
    )

    high = np.percentile(
        values,
        90
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
        1.0
    )


def match_feature_length(
    values,
    target_length
):
    values = np.asarray(
        values,
        dtype=float
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
        target_length - len(values),
        values[-1]
    )

    return np.concatenate([
        values,
        padding
    ])


def sample_curve_at_time(
    curve,
    time_seconds,
    sr,
    radius=1
):
    frame = int(
        round(
            time_seconds
            *
            sr
            /
            HOP_LENGTH
        )
    )

    start = max(
        0,
        frame - radius
    )

    end = min(
        len(curve),
        frame + radius + 1
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
    window
):
    values = np.asarray(
        values,
        dtype=float
    )

    if window <= 1:
        return values.copy()

    kernel = (
        np.ones(window)
        /
        window
    )

    return np.convolve(
        values,
        kernel,
        mode="same"
    )


def rising_trend(
    values,
    window_frames
):
    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) == 0:
        return values

    half = max(
        2,
        window_frames // 2
    )

    smoothed = moving_average(
        values,
        max(
            2,
            half // 4
        )
    )

    recent = moving_average(
        smoothed,
        half
    )

    previous = np.roll(
        recent,
        half
    )

    trend = (
        recent
        -
        previous
    )

    trend[
        :half
    ] = 0

    trend = np.maximum(
        trend,
        0
    )

    return normalize_feature(
        trend
    )


# =========================================================
# BASIC BEAT V1
# =========================================================

def build_accent_curve(
    y,
    sr,
    onset_envelope
):
    onset_curve = normalize_feature(
        onset_envelope
    )

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

    mel_frequencies = (
        librosa.mel_frequencies(
            n_mels=32,
            fmin=30,
            fmax=1000,
        )
    )

    bass_mask = (
        mel_frequencies
        <=
        180
    )

    if np.any(
        bass_mask
    ):
        bass_curve = np.mean(
            mel[
                bass_mask,
                :
            ],
            axis=0
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
            target_length
        )
    )

    rms_curve = (
        match_feature_length(
            rms_curve,
            target_length
        )
    )

    accent_curve = (
        0.55
        *
        onset_curve

        +

        0.30
        *
        bass_curve

        +

        0.15
        *
        rms_curve
    )

    return normalize_feature(
        accent_curve
    )


def get_median_beat_period(
    beat_times
):
    if len(
        beat_times
    ) < 2:
        return 0.5

    differences = np.diff(
        beat_times
    )

    differences = (
        differences[
            differences > 0
        ]
    )

    if len(
        differences
    ) == 0:
        return 0.5

    return float(
        np.median(
            differences
        )
    )


def create_offset_candidates(
    beat_period,
    sr
):
    analysis_step = (
        HOP_LENGTH
        /
        sr
    )

    maximum_shift = (
        beat_period
        *
        0.55
    )

    offsets = np.arange(
        -maximum_shift,
        maximum_shift
        +
        analysis_step / 2,
        analysis_step,
    )

    offsets = np.append(
        offsets,
        0.0
    )

    return np.unique(
        np.round(
            offsets,
            6
        )
    )


def get_phrase_selected_indices(
    block_start,
    block_end,
    phase,
    interval
):
    selected = []

    for beat_index in range(
        block_start,
        block_end
    ):

        local_index = (
            beat_index
            -
            block_start
        )

        if (
            local_index
            %
            interval
            ==
            phase
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

    for beat_index in (
        selected_indices
    ):

        if (
            beat_index
            <
            block_start
            +
            ignore_first_beats
        ):
            continue

        shifted_time = (
            float(
                beat_times[
                    beat_index
                ]
            )
            +
            offset
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

    if len(
        strengths
    ) == 0:
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
            25
        )
    )

    score = (
        0.50
        *
        median_strength

        +

        0.25
        *
        mean_strength

        +

        0.25
        *
        lower_strength
    )

    beat_period = (
        get_median_beat_period(
            beat_times
        )
    )

    if beat_period > 0:
        score -= (
            0.025
            *
            abs(
                offset
            )
            /
            beat_period
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
                    ignore_first_beats=(
                        ignore_first_beats
                    ),
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
        best_score
    )


def select_phrase_locked_beats(
    beat_times,
    accent_curve,
    sr,
    interval
):
    beat_count = len(
        beat_times
    )

    if beat_count == 0:
        return [], []

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
            sr
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
        PHRASE_BEATS
    ):

        block_end = min(
            beat_count,
            block_start
            +
            PHRASE_BEATS
        )

        if block_number == 0:

            ignore_first_beats = min(
                2,
                max(
                    0,
                    block_end
                    -
                    block_start
                    -
                    1
                ),
            )

            (
                chosen_phase,
                chosen_offset,
                chosen_score,
            ) = (
                find_best_phrase_state(
                    beat_times=beat_times,
                    accent_curve=accent_curve,
                    sr=sr,
                    block_start=block_start,
                    block_end=block_end,
                    interval=interval,
                    offset_candidates=(
                        offset_candidates
                    ),
                    ignore_first_beats=(
                        ignore_first_beats
                    ),
                )
            )

        else:

            (
                best_phase,
                best_offset,
                best_score,
            ) = (
                find_best_phrase_state(
                    beat_times=beat_times,
                    accent_curve=accent_curve,
                    sr=sr,
                    block_start=block_start,
                    block_end=block_end,
                    interval=interval,
                    offset_candidates=(
                        offset_candidates
                    ),
                    ignore_first_beats=0,
                )
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
                -
                previous_offset
            )

            offset_change_penalty = (
                0.05
                *
                offset_change
                /
                max(
                    beat_period,
                    0.001
                )
            )

            phase_change_penalty = 0.0

            if (
                best_phase
                !=
                previous_phase
            ):
                phase_change_penalty = (
                    0.04
                )

            required_improvement = (
                SWITCH_MARGIN
                +
                offset_change_penalty
                +
                phase_change_penalty
            )

            if (
                best_score
                >
                previous_score
                +
                required_improvement
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

        phrase_indices = (
            get_phrase_selected_indices(
                block_start,
                block_end,
                chosen_phase,
                interval,
            )
        )

        for beat_index in (
            phrase_indices
        ):

            shifted_time = (
                float(
                    beat_times[
                        beat_index
                    ]
                )
                +
                chosen_offset
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

            selected_records.append({
                "time":
                shifted_time,

                "strength":
                strength,

                "beat_index":
                beat_index,

                "block":
                block_number,
            })

        phrase_states.append({
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
        })

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
        item["time"]
    )

    if len(
        selected_records
    ) == 0:
        return (
            [],
            phrase_states
        )

    target_gap = (
        beat_period
        *
        interval
    )

    minimum_gap = (
        target_gap
        *
        0.55
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

        previous = (
            cleaned[-1]
        )

        gap = (
            record["time"]
            -
            previous["time"]
        )

        if gap < minimum_gap:

            if (
                record["strength"]
                >
                previous["strength"]
            ):
                cleaned[-1] = (
                    record
                )

        else:
            cleaned.append(
                record
            )

    selected_times = [
        item["time"]
        for item
        in cleaned
    ]

    return (
        selected_times,
        phrase_states
    )


# =========================================================
# SONG ANALYZER 0.1
# =========================================================

def calculate_band_activity(
    y,
    sr
):
    """
    Low-frequency change:
    kick / bass / low percussion.

    High-frequency change:
    piano / guitar / claps / cymbals / bright synths.
    """

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
            n_fft=n_fft
        )
    )

    low_mask = (
        (
            frequencies >= 30
        )
        &
        (
            frequencies <= 220
        )
    )

    high_mask = (
        (
            frequencies >= 600
        )
        &
        (
            frequencies <= 8000
        )
    )

    low_energy = np.mean(
        magnitude[
            low_mask,
            :
        ],
        axis=0
    )

    high_energy = np.mean(
        magnitude[
            high_mask,
            :
        ],
        axis=0
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
            prepend=low_log[0]
        ),
        0
    )

    high_change = np.maximum(
        np.diff(
            high_log,
            prepend=high_log[0]
        ),
        0
    )

    return (
        normalize_feature(
            low_change
        ),
        normalize_feature(
            high_change
        ),
    )


def build_song_analysis(
    y,
    sr
):
    """
    Build independent musical signals.

    This describes the song.
    It does NOT choose Smart Edit cuts.
    """

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

    rms = (
        librosa.feature.rms(
            y=y,
            frame_length=2048,
            hop_length=HOP_LENGTH,
        )[0]
    )

    rms = normalize_feature(
        match_feature_length(
            rms,
            target_length
        )
    )

    brightness = (
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=HOP_LENGTH,
        )[0]
    )

    brightness = normalize_feature(
        match_feature_length(
            brightness,
            target_length
        )
    )

    (
        bass_hit,
        high_hit
    ) = calculate_band_activity(
        y,
        sr
    )

    bass_hit = (
        match_feature_length(
            bass_hit,
            target_length
        )
    )

    high_hit = (
        match_feature_length(
            high_hit,
            target_length
        )
    )

    _, beat_frames = (
        librosa.beat.beat_track(
            onset_envelope=(
                onset_envelope
            ),
            sr=sr,
            hop_length=HOP_LENGTH,
        )
    )

    beat_frames = np.asarray(
        beat_frames,
        dtype=int
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
            <=
            frame
            <
            target_length
        ):
            beat_pulse[
                frame
            ] = 1.0

    frames_per_second = (
        sr
        /
        HOP_LENGTH
    )

    density_window = max(
        2,
        int(
            round(
                2.0
                *
                frames_per_second
            )
        ),
    )

    onset_density = (
        normalize_feature(
            moving_average(
                onset,
                density_window
            )
        )
    )

    build_window = max(
        4,
        int(
            round(
                4.0
                *
                frames_per_second
            )
        ),
    )

    energy_rise = (
        rising_trend(
            rms,
            build_window
        )
    )

    brightness_rise = (
        rising_trend(
            brightness,
            build_window
        )
    )

    density_rise = (
        rising_trend(
            onset_density,
            build_window
        )
    )

    percussive_activity = (
        moving_average(
            np.maximum(
                onset,
                bass_hit
            ),
            max(
                2,
                int(
                    frames_per_second
                )
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
        *
        energy_rise

        +

        0.22
        *
        brightness_rise

        +

        0.25
        *
        density_rise

        +

        0.13
        *
        percussive_activity
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
        "times":
        times,

        "onset":
        onset,

        "onset_raw":
        onset_envelope,

        "rms":
        rms,

        "brightness":
        brightness,

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
    }


def calculate_beat_alignment(
    event_time,
    beat_times,
    beat_period
):
    if len(
        beat_times
    ) == 0:
        return 0.0

    distance = float(
        np.min(
            np.abs(
                beat_times
                -
                event_time
            )
        )
    )

    tolerance = max(
        0.08,
        beat_period
        *
        0.45
    )

    alignment = (
        1.0
        -
        distance
        /
        tolerance
    )

    return float(
        np.clip(
            alignment,
            0.0,
            1.0
        )
    )


def calculate_pause_before_hit(
    rms,
    frame,
    sr
):
    """
    High value means there was a noticeable
    energy drop before the hit.
    """

    fps = (
        sr
        /
        HOP_LENGTH
    )

    short_window = int(
        0.70
        *
        fps
    )

    context_window = int(
        2.50
        *
        fps
    )

    pause_start = max(
        0,
        frame
        -
        short_window
    )

    context_start = max(
        0,
        pause_start
        -
        context_window
    )

    pre_values = rms[
        pause_start:
        max(
            pause_start + 1,
            frame
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
            pause_start
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

    energy_drop = (
        context_energy
        -
        pre_energy
    )

    pause_score = (
        energy_drop
        *
        3.0
    )

    if pre_energy < 0.20:
        pause_score += 0.20

    return float(
        np.clip(
            pause_score,
            0.0,
            1.0
        )
    )


def calculate_preceding_build(
    build_score,
    frame,
    sr
):
    fps = (
        sr
        /
        HOP_LENGTH
    )

    lookback = int(
        6.0
        *
        fps
    )

    exclude_near_hit = int(
        0.15
        *
        fps
    )

    start = max(
        0,
        frame
        -
        lookback
    )

    end = max(
        start + 1,
        frame
        -
        exclude_near_hit
    )

    values = build_score[
        start:end
    ]

    if len(
        values
    ) == 0:
        return 0.0

    return float(
        np.max(
            values
        )
    )


def detect_song_events(
    analysis,
    sr
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

    build_score = analysis[
        "build_score"
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

    events = []

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

        preceding_build = (
            calculate_preceding_build(
                build_score,
                frame,
                sr,
            )
        )

        distinctive_score = max(
            bass_strength,
            high_strength,
            onset_strength,
        )

        # Loud / strong hits deliberately get
        # major weight.
        hit_strength = (
            0.60
            *
            onset_strength

            +

            0.15
            *
            bass_strength

            +

            0.15
            *
            high_strength

            +

            0.10
            *
            energy_strength
        )

        context_score = max(
            pause_score,
            preceding_build,
        )

        # Debugging score only.
        # Smart Edit will later use a separate
        # cut-selection model.
        opportunity = (
            0.40
            *
            hit_strength

            +

            0.20
            *
            beat_alignment

            +

            0.25
            *
            context_score

            +

            0.15
            *
            distinctive_score
        )

        opportunity = float(
            np.clip(
                opportunity,
                0.0,
                1.0
            )
        )

        # Keep fairly weak candidates too.
        if (
            onset_strength
            <
            0.20

            and

            distinctive_score
            <
            0.25
        ):
            continue

        labels = []

        if (
            pause_score
            >=
            0.45

            and

            onset_strength
            >=
            0.35
        ):
            labels.append(
                "Post-pause hit"
            )

        if (
            preceding_build
            >=
            0.60

            and

            onset_strength
            >=
            0.35
        ):
            labels.append(
                "Post-build hit"
            )

        if (
            onset_strength
            >=
            0.78
        ):
            labels.append(
                "Very strong hit"
            )

        elif (
            onset_strength
            >=
            0.48
        ):
            labels.append(
                "Strong hit"
            )

        else:
            labels.append(
                "Accent"
            )

        if (
            high_strength
            >=
            0.55

            and

            high_strength
            >
            bass_strength
            +
            0.08
        ):
            labels.append(
                "High / tonal"
            )

        if (
            bass_strength
            >=
            0.55

            and

            bass_strength
            >
            high_strength
            +
            0.05
        ):
            labels.append(
                "Bass / kick"
            )

        events.append({
            "Time":
            event_time,

            "Event":
            " + ".join(
                labels
            ),

            "Opportunity":
            opportunity,

            "Hit strength":
            hit_strength,

            "Beat alignment":
            beat_alignment,

            "Pause before":
            pause_score,

            "Build before":
            preceding_build,

            "Bass":
            bass_strength,

            "High / tonal":
            high_strength,

            "Onset":
            onset_strength,
        })

    return sorted(
        events,
        key=lambda item:
        item["Time"]
    )


def detect_build_regions(
    analysis,
    sr
):
    build = analysis[
        "build_score"
    ]

    times = analysis[
        "times"
    ]

    threshold = 0.52

    mask = (
        build
        >=
        threshold
    )

    minimum_frames = max(
        2,
        int(
            round(
                0.90
                *
                sr
                /
                HOP_LENGTH
            )
        ),
    )

    regions = []
    start = None

    for index, active in enumerate(
        mask
    ):

        if (
            active
            and
            start is None
        ):
            start = index

        if (
            start is not None

            and

            (
                not active

                or

                index
                ==
                len(mask) - 1
            )
        ):

            if active:
                end = index + 1

            else:
                end = index

            if (
                end - start
                >=
                minimum_frames
            ):

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

                regions.append({
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
                                len(times) - 1
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
                    float(
                        build[
                            peak_index
                        ]
                    ),
                })

            start = None

    return regions


def show_song_analyzer(
    y,
    sr
):
    analysis = (
        build_song_analysis(
            y,
            sr
        )
    )

    events = (
        detect_song_events(
            analysis,
            sr
        )
    )

    builds = (
        detect_build_regions(
            analysis,
            sr
        )
    )

    beat_times = analysis[
        "beat_times"
    ]

    if len(
        beat_times
    ) > 1:

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

    st.success(
        "Song analysis complete."
    )

    metric1, metric2, metric3 = (
        st.columns(3)
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
            len(
                beat_times
            ),
        )

    with metric3:
        st.metric(
            "Musical events",
            len(
                events
            ),
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
        pd.DataFrame({
            "Time":
            analysis[
                "times"
            ],

            "Energy":
            analysis[
                "rms"
            ],
        })
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
        "Overall onset = general attacks. "
        "Bass = kick/low-frequency attacks. "
        "High/Tonal = piano, guitar, cymbal, synth "
        "and other brighter attacks. "
        "Beat spikes show the detected beat grid."
    )

    hit_dataframe = (
        pd.DataFrame({
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
        })
        .set_index(
            "Time"
        )
    )

    st.line_chart(
        hit_dataframe,
        height=300
    )

    # -----------------------------------------------------
    # BUILDS
    # -----------------------------------------------------

    st.subheader(
        "Build-up analysis"
    )

    st.caption(
        "The build score looks for increasing energy, "
        "brightness, musical activity and percussion."
    )

    build_dataframe = (
        pd.DataFrame({
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
        })
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

    if len(
        builds
    ) == 0:

        st.write(
            "No clear build regions detected."
        )

    else:

        build_rows = []

        for build in builds:

            build_rows.append({
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

                "Build score":
                round(
                    build[
                        "Score"
                    ],
                    2
                ),
            })

        st.dataframe(
            pd.DataFrame(
                build_rows
            ),
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # EVENT TABLE
    # -----------------------------------------------------

    st.subheader(
        "Detected musical events"
    )

    st.caption(
        "These are candidates, not automatic cuts. "
        "Several nearby hits are intentionally kept."
    )

    event_rows = []

    for event in events:

        event_rows.append({
            "Time":
            format_time(
                event[
                    "Time"
                ]
            ),

            "Event":
            event[
                "Event"
            ],

            "Opportunity":
            round(
                event[
                    "Opportunity"
                ],
                2
            ),

            "Hit":
            round(
                event[
                    "Hit strength"
                ],
                2
            ),

            "Beat":
            round(
                event[
                    "Beat alignment"
                ],
                2
            ),

            "Pause":
            round(
                event[
                    "Pause before"
                ],
                2
            ),

            "Build":
            round(
                event[
                    "Build before"
                ],
                2
            ),

            "Bass":
            round(
                event[
                    "Bass"
                ],
                2
            ),

            "High":
            round(
                event[
                    "High / tonal"
                ],
                2
            ),
        })

    event_dataframe = (
        pd.DataFrame(
            event_rows
        )
    )

    st.dataframe(
        event_dataframe,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    if not event_dataframe.empty:

        csv_bytes = (
            event_dataframe
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )

        st.download_button(
            "Download analysis CSV",
            data=csv_bytes,
            file_name=(
                "DetectTheBeat_SongAnalysis.csv"
            ),
            mime="text/csv",
        )


# =========================================================
# MODE
# =========================================================

mode = st.radio(
    "Mode",
    [
        "Beat",
        "Song Analyzer"
    ],
    horizontal=True,
)


# =========================================================
# UPLOAD
# =========================================================

uploaded_file = (
    st.file_uploader(
        "Upload audio",
        type=[
            "mp3",
            "wav",
            "m4a"
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


# =========================================================
# SONG ANALYZER MODE
# =========================================================

if (
    mode
    ==
    "Song Analyzer"
):

    st.write(
        "Song Analyzer finds beats, hits, pauses, "
        "build-ups and distinctive musical accents. "
        "It does not choose Smart Edit cuts yet."
    )

    if (
        uploaded_file
        is not None
    ):

        if st.button(
            "🔍 Analyze Song",
            type="primary"
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
                        >
                        MAX_AUDIO_DURATION
                    ):

                        status.update(
                            label=(
                                "Audio is too long"
                            ),
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

                    y, sr = librosa.load(
                        str(
                            analysis_path
                        ),
                        sr=None,
                        mono=True,
                    )

                    status.write(
                        "3/4 · Detecting musical structure"
                    )

                    status.write(
                        "4/4 · Finding editing opportunities"
                    )

                    analysis_output = (
                        y,
                        sr
                    )

                status.update(
                    label="Analysis ready",
                    state="complete",
                    expanded=False,
                )

                show_song_analyzer(
                    analysis_output[0],
                    analysis_output[1],
                )

            except Exception as error:

                if status is not None:

                    try:
                        status.update(
                            label=(
                                "Song analysis failed"
                            ),
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


# =========================================================
# BASIC BEAT MODE
# =========================================================

else:

    st.write(
        "Beat mode creates a checkerboard reference video "
        "for Scene Edit Detection."
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
            "23.976 fps"
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

    if (
        uploaded_file
        is not None
    ):

        download_filename = (
            make_output_filename(
                uploaded_file.name,
                beat_choice,
            )
        )

        if st.button(
            "🚀 Generate Video",
            type="primary"
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

                    # ==========================================
                    # STEP 1
                    # ==========================================

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
                        >
                        MAX_AUDIO_DURATION
                    ):

                        status.update(
                            label=(
                                "Audio is too long"
                            ),
                            state="error",
                        )

                        st.error(
                            "Please upload a track "
                            "of 6 minutes or less."
                        )

                        st.stop()

                    minutes = int(
                        total_duration
                        //
                        60
                    )

                    seconds = int(
                        total_duration
                        %
                        60
                    )

                    status.write(
                        f"Audio length: "
                        f"{minutes}:"
                        f"{seconds:02d}"
                    )

                    # ==========================================
                    # STEP 2
                    # ==========================================

                    status.write(
                        "2/5 · Preparing audio "
                        "for beat detection"
                    )

                    analysis_path = (
                        create_analysis_wav(
                            audio_path,
                            work_dir,
                        )
                    )

                    # ==========================================
                    # STEP 3
                    # ==========================================

                    status.write(
                        "3/5 · Detecting rhythm "
                        "and strong musical accents"
                    )

                    y, sr = librosa.load(
                        str(
                            analysis_path
                        ),
                        sr=None,
                        mono=True,
                    )

                    onset_envelope = (
                        librosa.onset.onset_strength(
                            y=y,
                            sr=sr,
                            hop_length=(
                                HOP_LENGTH
                            ),
                        )
                    )

                    _, beat_frames = (
                        librosa.beat.beat_track(
                            onset_envelope=(
                                onset_envelope
                            ),
                            sr=sr,
                            hop_length=(
                                HOP_LENGTH
                            ),
                        )
                    )

                    beat_times = (
                        librosa.frames_to_time(
                            beat_frames,
                            sr=sr,
                            hop_length=(
                                HOP_LENGTH
                            ),
                        )
                    )

                    if (
                        len(
                            beat_times
                        )
                        ==
                        0
                    ):

                        raise RuntimeError(
                            "No reliable beats were "
                            "detected in this track."
                        )

                    accent_curve = (
                        build_accent_curve(
                            y=y,
                            sr=sr,
                            onset_envelope=(
                                onset_envelope
                            ),
                        )
                    )

                    (
                        selected_beats,
                        phrase_states,
                    ) = (
                        select_phrase_locked_beats(
                            beat_times=(
                                beat_times
                            ),
                            accent_curve=(
                                accent_curve
                            ),
                            sr=sr,
                            interval=(
                                BEAT_INTERVAL
                            ),
                        )
                    )

                    status.write(
                        f"Detected "
                        f"{len(beat_times)} "
                        f"base beats"
                    )

                    if (
                        BEAT_INTERVAL
                        ==
                        1
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
                                    *
                                    1000
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
                                    !=
                                    previous_state[
                                        "phase"
                                    ]

                                    or

                                    abs(
                                        current_state[
                                            "offset"
                                        ]
                                        -
                                        previous_state[
                                            "offset"
                                        ]
                                    )
                                    >
                                    0.01
                                ):

                                    number_of_changes += (
                                        1
                                    )

                            status.write(
                                f"Rhythm alignment changes: "
                                f"{number_of_changes}"
                            )

                    del y
                    del onset_envelope
                    del accent_curve

                    # ==========================================
                    # FRAME CONVERSION
                    # ==========================================

                    total_video_frames = (
                        math.ceil(
                            total_duration
                            *
                            FPS_VALUE
                        )
                    )

                    scene_change_frames = []

                    for beat in (
                        selected_beats
                    ):

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
                            <
                            frame_number
                            <
                            total_video_frames
                        ):

                            scene_change_frames.append(
                                frame_number
                            )

                    scene_change_frames = sorted(
                        set(
                            scene_change_frames
                        )
                    )

                    status.write(
                        f"Creating "
                        f"{len(scene_change_frames)} "
                        f"scene changes"
                    )

                    # ==========================================
                    # STEP 4
                    # ==========================================

                    status.write(
                        "4/5 · Building "
                        "frame-accurate video"
                    )

                    raw_video_path = (
                        work_dir
                        /
                        "reference.rgb"
                    )

                    change_index = 0
                    pattern_index = -1

                    with open(
                        raw_video_path,
                        "wb"
                    ) as raw_video:

                        for frame_number in range(
                            total_video_frames
                        ):

                            while (
                                change_index
                                <
                                len(
                                    scene_change_frames
                                )

                                and

                                frame_number
                                >=
                                scene_change_frames[
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
                                %
                                2
                                ==
                                0
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

                    # ==========================================
                    # STEP 5
                    # ==========================================

                    status.write(
                        "5/5 · Rendering video"
                    )

                    output_path = (
                        work_dir
                        /
                        "detectthebeat_video.mp4"
                    )

                    run_command([
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
                            f"x"
                            f"{INTERNAL_HEIGHT}"
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
                    ])

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
                    file_name=(
                        download_filename
                    ),
                    mime="video/mp4",
                )

                st.info(
                    "Import the MP4 into your editing software "
                    "and use Scene Edit Detection to create cuts "
                    "at the checkerboard changes. Use your original "
                    "audio file for the final edit."
                )

            except Exception as error:

                if status is not None:

                    try:
                        status.update(
                            label=(
                                "Something went wrong"
                            ),
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
