import streamlit as st
import librosa
import tempfile
import subprocess
import math
import re
import numpy as np
import imageio_ffmpeg
from pathlib import Path


FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="DetectTheBeat",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 DetectTheBeat")

st.write(
    "Turn music into edit points. Upload a track, choose your settings, "
    "generate a beat-synced reference video, then use Scene Edit Detection "
    "in Premiere Pro or similar editing software."
)


# =========================================================
# SETTINGS
# =========================================================

MAX_AUDIO_DURATION = 6 * 60

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

INTERNAL_WIDTH = 64
INTERNAL_HEIGHT = 36

HOP_LENGTH = 512

# Keep an editing rhythm locked for approximately
# one musical phrase before reconsidering it.
PHRASE_BEATS = 16

# How much better a new alignment must be before
# DetectTheBeat is allowed to change its mind.
SWITCH_MARGIN = 0.10


FPS_OPTIONS = {
    "25 fps": {
        "value": 25.0,
        "ffmpeg": "25"
    },
    "23.976 fps": {
        "value": 24000 / 1001,
        "ffmpeg": "24000/1001"
    },
}


BEAT_INTERVALS = {
    "Every beat": 1,
    "Every 2 beats": 2,
    "Every 4 beats": 4,
}


# =========================================================
# GENERAL FUNCTIONS
# =========================================================

def run_command(command, cwd=None):

    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        error_text = result.stderr[-5000:]

        raise RuntimeError(
            f"FFmpeg error:\n{error_text}"
        )

    return result


def get_audio_duration(audio_path):

    return float(
        librosa.get_duration(
            path=audio_path
        )
    )


def beat_time_to_frame(
    time_seconds,
    fps
):

    return round(
        time_seconds * fps
    )


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
        '',
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
# CHECKERBOARD
# =========================================================

def create_solid_frame(
    width,
    height,
    color
):

    pixel = bytes(color)

    return pixel * (
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
    (0, 0, 0)
)


PATTERN_A_FRAME = (
    create_checkerboard_frame(
        INTERNAL_WIDTH,
        INTERNAL_HEIGHT,
        inverted=False
    )
)


PATTERN_B_FRAME = (
    create_checkerboard_frame(
        INTERNAL_WIDTH,
        INTERNAL_HEIGHT,
        inverted=True
    )
)


# =========================================================
# MUSIC ANALYSIS
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


    return np.concatenate(
        [
            values,
            padding
        ]
    )


def build_accent_curve(
    y,
    sr,
    onset_envelope
):

    """
    Build one continuous musical-strength curve.

    We combine:

    - transient/onset strength
    - low-frequency kick/bass energy
    - overall loudness

    This curve lets us test not only Librosa's exact
    beat positions, but also nearby positions.

    That is important for songs where an intro/pickup
    causes the original beat grid to start slightly early.
    """

    onset_curve = normalize_feature(
        onset_envelope
    )


    # -----------------------------------------------------
    # LOW-FREQUENCY ENERGY
    # -----------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=HOP_LENGTH,
        n_mels=32,
        fmin=30,
        fmax=1000,
        power=1.0
    )


    mel_frequencies = (
        librosa.mel_frequencies(
            n_mels=32,
            fmin=30,
            fmax=1000
        )
    )


    bass_mask = (
        mel_frequencies <= 180
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


    # -----------------------------------------------------
    # RMS / GENERAL ENERGY
    # -----------------------------------------------------

    rms_curve = (
        librosa.feature.rms(
            y=y,
            frame_length=1024,
            hop_length=HOP_LENGTH
        )[0]
    )


    rms_curve = normalize_feature(
        rms_curve
    )


    # -----------------------------------------------------
    # MATCH LENGTHS
    # -----------------------------------------------------

    target_length = len(
        onset_curve
    )


    bass_curve = match_feature_length(
        bass_curve,
        target_length
    )


    rms_curve = match_feature_length(
        rms_curve,
        target_length
    )


    # -----------------------------------------------------
    # COMBINED MUSIC-ACCENT CURVE
    # -----------------------------------------------------

    accent_curve = (

        0.55
        * onset_curve

        +

        0.30
        * bass_curve

        +

        0.15
        * rms_curve

    )


    return normalize_feature(
        accent_curve
    )


def sample_curve_at_time(
    curve,
    time_seconds,
    sr,
    radius=1
):

    """
    Return the strongest musical-accent value
    within a tiny window around a time position.
    """

    frame = int(
        round(
            time_seconds
            * sr
            / HOP_LENGTH
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


# =========================================================
# RHYTHM GRID
# =========================================================

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


    differences = differences[
        differences > 0
    ]


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

    """
    Search slightly more than half a beat in
    either direction.

    This lets us discover situations such as:

    pickup note ---> real rhythmic hit

    where the true editing pulse is between
    Librosa's original beat positions.
    """

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
        maximum_shift + analysis_step / 2,
        analysis_step
    )


    # Make absolutely sure 0 is tested too.
    offsets = np.append(
        offsets,
        0.0
    )


    offsets = np.unique(
        np.round(
            offsets,
            6
        )
    )


    return offsets


# =========================================================
# PHRASE SCORING
# =========================================================

def get_phrase_selected_indices(
    block_start,
    block_end,
    phase,
    interval
):

    """
    Return the beat indices that would become edit
    points for one particular phase.

    Example, Every 4:

    phase 0:
    x . . . x . . . x

    phase 1:
    . x . . . x . . .
    """

    selected = []


    for beat_index in range(
        block_start,
        block_end
    ):

        local_index = (
            beat_index
            - block_start
        )


        if (
            local_index
            % interval
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
    ignore_first_beats=0
):

    """
    Score one possible editing rhythm.

    A good rhythm should not merely contain one huge hit.

    It should contain beats that are repeatedly strong.

    Median and lower-percentile strength therefore matter
    more than a single maximum value.
    """

    selected_indices = (
        get_phrase_selected_indices(
            block_start,
            block_end,
            phase,
            interval
        )
    )


    strengths = []


    for beat_index in selected_indices:

        if (
            beat_index
            <
            block_start
            + ignore_first_beats
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
                radius=1
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


    # Consistency is more valuable than one huge transient.
    score = (

        0.50
        * median_strength

        +

        0.25
        * mean_strength

        +

        0.25
        * lower_strength

    )


    # Tiny preference for staying close to Librosa's
    # original grid if two alignments are nearly equal.
    beat_period = get_median_beat_period(
        beat_times
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
    ignore_first_beats=0
):

    best_phase = 0
    best_offset = 0.0
    best_score = -999.0


    for phase in range(
        interval
    ):


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
                ignore_first_beats=(
                    ignore_first_beats
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


# =========================================================
# PHRASE-LOCKED EDITING RHYTHM
# =========================================================

def select_phrase_locked_beats(
    beat_times,
    accent_curve,
    sr,
    interval
):

    """
    DetectTheBeat Basic Beat rhythm system.

    1. Analyze the first phrase before deciding where
       the song really starts.

    2. Ignore the first couple of beat detections while
       scoring the opening phrase so a pickup note cannot
       dominate.

    3. Once the best opening grid has been discovered,
       apply it back to the beginning.

    4. Keep one rhythm alignment locked for 16 beats.

    5. Only change phase/grid in a later phrase when the
       new alignment is clearly stronger.
    """

    beat_count = len(
        beat_times
    )


    if beat_count == 0:

        return [], []


    # Every beat remains exactly as Librosa detected it.
    if interval == 1:

        selected_times = [
            float(x)
            for x in beat_times
        ]

        return (
            selected_times,
            []
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
            + PHRASE_BEATS
        )


        # -------------------------------------------------
        # OPENING PHRASE
        # -------------------------------------------------

        if block_number == 0:

            # The first two detected beats do not influence
            # the opening alignment decision.
            #
            # Once the alignment is found, it is applied
            # back to the whole opening phrase.
            ignore_first_beats = min(
                2,
                max(
                    0,
                    block_end
                    - block_start
                    - 1
                )
            )


            (
                chosen_phase,
                chosen_offset,
                chosen_score
            ) = find_best_phrase_state(
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
                )
            )


        # -------------------------------------------------
        # LATER PHRASES
        # -------------------------------------------------

        else:

            (
                best_phase,
                best_offset,
                best_score
            ) = find_best_phrase_state(
                beat_times=beat_times,
                accent_curve=accent_curve,
                sr=sr,
                block_start=block_start,
                block_end=block_end,
                interval=interval,
                offset_candidates=(
                    offset_candidates
                ),
                ignore_first_beats=0
            )


            # Score the alignment we are already using.
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
                    ignore_first_beats=0
                )
            )


            # Changing alignment has a cost.
            # The larger the timing shift, the more evidence
            # we require before changing.
            offset_change = abs(
                best_offset
                - previous_offset
            )


            offset_change_penalty = (
                0.05
                * offset_change
                / max(
                    beat_period,
                    0.001
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


        # -------------------------------------------------
        # CREATE CUTS FOR THIS PHRASE
        # -------------------------------------------------

        phrase_indices = (
            get_phrase_selected_indices(
                block_start,
                block_end,
                chosen_phase,
                interval
            )
        )


        for beat_index in phrase_indices:


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
                    radius=1
                )
            )


            selected_records.append(
                {
                    "time": shifted_time,
                    "strength": strength,
                    "beat_index": beat_index,
                    "block": block_number
                }
            )


        phrase_states.append(
            {
                "block": block_number,
                "start_beat": block_start,
                "end_beat": block_end,
                "phase": chosen_phase,
                "offset": chosen_offset,
                "score": chosen_score
            }
        )


        previous_phase = (
            chosen_phase
        )


        previous_offset = (
            chosen_offset
        )


        block_number += 1


    # =====================================================
    # CLEAN UP PHRASE BOUNDARIES
    # =====================================================

    selected_records = sorted(
        selected_records,
        key=lambda item: item["time"]
    )


    if len(
        selected_records
    ) == 0:

        return [], phrase_states


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
            -
            previous["time"]
        )


        # If a phrase change accidentally creates two cuts
        # extremely close together, keep only the stronger one.
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
        for item in cleaned
    ]


    return (
        selected_times,
        phrase_states
    )


# =========================================================
# USER INTERFACE
# =========================================================

uploaded_file = st.file_uploader(
    "Upload audio",
    type=[
        "mp3",
        "wav",
        "m4a"
    ]
)


st.caption(
    "MP3, WAV or M4A · Maximum length: 6 minutes"
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
    horizontal=True
)


beat_choice = st.radio(
    "Scene change",
    options=[
        "Every beat",
        "Every 2 beats",
        "Every 4 beats"
    ],

    # Every 4 beats is default.
    index=2
)


FPS_VALUE = (
    FPS_OPTIONS[
        fps_choice
    ]["value"]
)


FPS_FFMPEG = (
    FPS_OPTIONS[
        fps_choice
    ]["ffmpeg"]
)


BEAT_INTERVAL = (
    BEAT_INTERVALS[
        beat_choice
    ]
)


# =========================================================
# GENERATION
# =========================================================

if uploaded_file is not None:


    st.audio(
        uploaded_file
    )


    download_filename = (
        make_output_filename(
            uploaded_file.name,
            beat_choice
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
                expanded=True
            )


            with tempfile.TemporaryDirectory() as work_dir:


                work_dir = Path(
                    work_dir
                )


                # =================================================
                # STEP 1
                # =================================================

                status.write(
                    "1/5 · Checking audio"
                )


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
                        label="Audio is too long",
                        state="error"
                    )


                    st.error(
                        "Please upload a track "
                        "of 6 minutes or less."
                    )


                    st.stop()


                minutes = int(
                    total_duration // 60
                )


                seconds = int(
                    total_duration % 60
                )


                status.write(
                    f"Audio length: "
                    f"{minutes}:"
                    f"{seconds:02d}"
                )


                # =================================================
                # STEP 2
                # =================================================

                status.write(
                    "2/5 · Preparing audio "
                    "for beat detection"
                )


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
                    str(
                        audio_path
                    ),

                    "-vn",

                    "-ac",
                    "1",

                    "-ar",
                    "22050",

                    str(
                        analysis_path
                    )
                ])


                # =================================================
                # STEP 3
                # =================================================

                status.write(
                    "3/5 · Detecting rhythm "
                    "and strong musical accents"
                )


                y, sr = librosa.load(
                    str(
                        analysis_path
                    ),
                    sr=None,
                    mono=True
                )


                onset_envelope = (
                    librosa.onset.onset_strength(
                        y=y,
                        sr=sr,
                        hop_length=(
                            HOP_LENGTH
                        )
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
                        )
                    )
                )


                beat_times = (
                    librosa.frames_to_time(
                        beat_frames,
                        sr=sr,
                        hop_length=(
                            HOP_LENGTH
                        )
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
                        )
                    )
                )


                (
                    selected_beats,
                    phrase_states
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
                        )
                    )
                )


                status.write(
                    f"Detected "
                    f"{len(beat_times)} "
                    f"base beats"
                )


                if (
                    BEAT_INTERVAL == 1
                ):

                    status.write(
                        "Using every detected beat"
                    )


                else:

                    status.write(
                        f"Selected "
                        f"{len(selected_beats)} "
                        f"phrase-locked "
                        f"edit points"
                    )


                    if len(
                        phrase_states
                    ) > 0:

                        initial_offset = (
                            phrase_states[
                                0
                            ]["offset"]
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
                            )
                        ):


                            current_state = (
                                phrase_states[
                                    state_index
                                ]
                            )


                            previous_state = (
                                phrase_states[
                                    state_index
                                    -
                                    1
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

                                number_of_changes += 1


                        status.write(
                            f"Rhythm alignment changes: "
                            f"{number_of_changes}"
                        )


                del y
                del onset_envelope
                del accent_curve


                # =================================================
                # FRAME CONVERSION
                # =================================================

                total_video_frames = (
                    math.ceil(
                        total_duration
                        *
                        FPS_VALUE
                    )
                )


                scene_change_frames = []


                for beat in selected_beats:


                    frame_number = (
                        beat_time_to_frame(
                            float(
                                beat
                            ),
                            FPS_VALUE
                        )
                    )


                    if (
                        frame_number > 0
                        and
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


                # =================================================
                # STEP 4
                # =================================================

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


                # =================================================
                # STEP 5
                # =================================================

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
                    )
                ])


                video_bytes = (
                    output_path.read_bytes()
                )


            # =================================================
            # FINISHED
            # =================================================

            status.update(
                label="Video ready!",
                state="complete",
                expanded=False
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
                mime="video/mp4"
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
                        label=(
                            "Something went wrong"
                        ),
                        state="error"
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