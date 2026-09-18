import streamlit as st
import librosa
import tempfile
import subprocess
import math
import re
import numpy as np
import imageio_ffmpeg

from pathlib import Path


# =========================================================
# FFMPEG
# =========================================================

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
# GENERAL
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
# CHECKERBOARD VIDEO
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


PATTERN_A_FRAME = create_checkerboard_frame(
    INTERNAL_WIDTH,
    INTERNAL_HEIGHT,
    inverted=False
)


PATTERN_B_FRAME = create_checkerboard_frame(
    INTERNAL_WIDTH,
    INTERNAL_HEIGHT,
    inverted=True
)


# =========================================================
# AUDIO FEATURE HELPERS
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


    values = (
        values - low
    ) / (
        high - low
    )


    return np.clip(
        values,
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


def sample_curve_near_time(
    curve,
    time_seconds,
    sr,
    radius_frames=1
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
        frame - radius_frames
    )


    end = min(
        len(curve),
        frame + radius_frames + 1
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
# EDITORIAL HIT STRENGTH
# =========================================================

def build_editorial_strength_curve(
    y,
    sr,
    onset_envelope
):

    """
    Build a continuous measure of how interesting each
    musical moment is likely to be as an editing hit.

    Onset/transient strength receives the most weight.

    That deliberately gives things such as piano hits,
    percussion, guitar attacks and cymbals a strong voice,
    rather than only looking for kick drums.
    """

    onset_curve = normalize_feature(
        onset_envelope
    )


    # -----------------------------------------------------
    # BASS / KICK ENERGY
    # -----------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=HOP_LENGTH,
        n_mels=40,
        fmin=30,
        fmax=8000,
        power=1.0
    )


    mel_frequencies = (
        librosa.mel_frequencies(
            n_mels=40,
            fmin=30,
            fmax=8000
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


    # -----------------------------------------------------
    # HIGH / MID TRANSIENT ENERGY
    #
    # Useful for piano notes, snaps, claps, guitar etc.
    # -----------------------------------------------------

    upper_mask = (
        mel_frequencies >= 700
    )


    if np.any(
        upper_mask
    ):

        upper_curve = np.mean(
            mel[
                upper_mask,
                :
            ],
            axis=0
        )

    else:

        upper_curve = np.zeros(
            mel.shape[1]
        )


    del mel


    bass_curve = normalize_feature(
        bass_curve
    )


    upper_curve = normalize_feature(
        upper_curve
    )


    # -----------------------------------------------------
    # GENERAL ENERGY
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


    upper_curve = match_feature_length(
        upper_curve,
        target_length
    )


    rms_curve = match_feature_length(
        rms_curve,
        target_length
    )


    # -----------------------------------------------------
    # COMBINED SCORE
    # -----------------------------------------------------

    strength_curve = (

        0.60
        * onset_curve

        +

        0.18
        * bass_curve

        +

        0.14
        * upper_curve

        +

        0.08
        * rms_curve

    )


    return normalize_feature(
        strength_curve
    )


def get_beat_strengths(
    beat_times,
    strength_curve,
    sr
):

    strengths = []


    for beat_time in beat_times:

        strength = sample_curve_near_time(
            strength_curve,
            float(beat_time),
            sr,
            radius_frames=1
        )


        strengths.append(
            strength
        )


    return np.asarray(
        strengths,
        dtype=float
    )


# =========================================================
# GLOBAL BEAT PATH OPTIMISATION
# =========================================================

def optimise_edit_path(
    beat_times,
    beat_strengths,
    interval
):

    """
    Choose the strongest useful sequence of editing beats
    across the WHOLE SONG.

    We keep approximately the requested pacing, but allow
    a cut to move one detected beat earlier or later.

    Every 2:
        normally 2 beats apart
        sometimes 1 or 3

    Every 4:
        normally 4 beats apart
        sometimes 3 or 5

    Crucially, the algorithm optimises the whole path rather
    than making each decision independently.

    This means changing phase once and staying there is cheap,
    while bouncing back and forth repeatedly is penalised.
    """

    beat_count = len(
        beat_times
    )


    if beat_count == 0:

        return []


    if interval == 1:

        return list(
            range(
                beat_count
            )
        )


    # Each intended cut may move one beat either side
    # of its regular target.
    deviations = [
        -1,
        0,
        1
    ]


    best_overall_score = -999999.0
    best_overall_path = []


    # -----------------------------------------------------
    # TRY EACH POSSIBLE STARTING PHASE
    # -----------------------------------------------------

    for base_phase in range(
        interval
    ):


        targets = list(
            range(
                base_phase,
                beat_count,
                interval
            )
        )


        if len(
            targets
        ) == 0:

            continue


        # DP stage:
        # candidate beat index ->
        # {
        #   score,
        #   path,
        #   deviation
        # }

        states = {}


        # -------------------------------------------------
        # FIRST EDIT POINT
        # -------------------------------------------------

        first_target = targets[0]


        for deviation in deviations:

            candidate = (
                first_target
                +
                deviation
            )


            if (
                candidate < 0
                or
                candidate >= beat_count
            ):

                continue


            # Avoid starting unnecessarily late,
            # but musical strength remains the priority.
            start_delay_penalty = (
                0.025
                * candidate
            )


            deviation_penalty = (
                0.07
                * abs(
                    deviation
                )
            )


            score = (

                float(
                    beat_strengths[
                        candidate
                    ]
                )

                -

                deviation_penalty

                -

                start_delay_penalty

            )


            states[
                candidate
            ] = {
                "score": score,
                "path": [
                    candidate
                ],
                "deviation": deviation
            }


        # -------------------------------------------------
        # REMAINING EDIT POINTS
        # -------------------------------------------------

        for target in targets[1:]:


            new_states = {}


            for deviation in deviations:


                candidate = (
                    target
                    +
                    deviation
                )


                if (
                    candidate < 0
                    or
                    candidate >= beat_count
                ):

                    continue


                best_candidate_state = None


                for (
                    previous_index,
                    previous_state
                ) in states.items():


                    if (
                        candidate
                        <=
                        previous_index
                    ):

                        continue


                    previous_deviation = (
                        previous_state[
                            "deviation"
                        ]
                    )


                    actual_gap = (
                        candidate
                        -
                        previous_index
                    )


                    # Normally:
                    #
                    # Every 2 -> 1,2,3 allowed
                    # Every 4 -> 3,4,5 allowed
                    #
                    # This check is mainly defensive.
                    minimum_gap = max(
                        1,
                        interval - 1
                    )


                    maximum_gap = (
                        interval + 1
                    )


                    if (
                        actual_gap
                        <
                        minimum_gap
                        or
                        actual_gap
                        >
                        maximum_gap
                    ):

                        continue


                    hit_score = float(
                        beat_strengths[
                            candidate
                        ]
                    )


                    # -------------------------------------
                    # DISTANCE FROM IDEAL SPACING
                    # -------------------------------------

                    spacing_penalty = (
                        0.11
                        * abs(
                            actual_gap
                            -
                            interval
                        )
                    )


                    # -------------------------------------
                    # PHASE-CHANGE PENALTY
                    #
                    # Moving from deviation 0 to +1 once
                    # is fine.
                    #
                    # Constantly doing:
                    # 0,+1,-1,+1...
                    # becomes expensive.
                    # -------------------------------------

                    phase_change_penalty = (
                        0.15
                        * abs(
                            deviation
                            -
                            previous_deviation
                        )
                    )


                    # Slight penalty just for being away
                    # from the ideal beat position.
                    deviation_penalty = (
                        0.035
                        * abs(
                            deviation
                        )
                    )


                    new_score = (

                        previous_state[
                            "score"
                        ]

                        +

                        hit_score

                        -

                        spacing_penalty

                        -

                        phase_change_penalty

                        -

                        deviation_penalty

                    )


                    if (
                        best_candidate_state
                        is None
                        or
                        new_score
                        >
                        best_candidate_state[
                            "score"
                        ]
                    ):


                        best_candidate_state = {
                            "score": new_score,
                            "path": (
                                previous_state[
                                    "path"
                                ]
                                +
                                [
                                    candidate
                                ]
                            ),
                            "deviation": deviation
                        }


                if (
                    best_candidate_state
                    is not None
                ):


                    existing = new_states.get(
                        candidate
                    )


                    if (
                        existing is None
                        or
                        best_candidate_state[
                            "score"
                        ]
                        >
                        existing[
                            "score"
                        ]
                    ):


                        new_states[
                            candidate
                        ] = (
                            best_candidate_state
                        )


            if len(
                new_states
            ) == 0:

                break


            states = new_states


        # -------------------------------------------------
        # BEST PATH FOR THIS START PHASE
        # -------------------------------------------------

        if len(
            states
        ) == 0:

            continue


        best_state = max(
            states.values(),
            key=lambda state: (
                state["score"]
                /
                max(
                    len(
                        state["path"]
                    ),
                    1
                )
            )
        )


        # Compare average musical quality rather than
        # blindly preferring the phase with one extra cut.
        average_score = (
            best_state[
                "score"
            ]
            /
            max(
                len(
                    best_state[
                        "path"
                    ]
                ),
                1
            )
        )


        if (
            average_score
            >
            best_overall_score
        ):


            best_overall_score = (
                average_score
            )


            best_overall_path = (
                best_state[
                    "path"
                ]
            )


    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    if len(
        best_overall_path
    ) == 0:


        best_overall_path = list(
            range(
                0,
                beat_count,
                interval
            )
        )


    return best_overall_path


# =========================================================
# SNAP SELECTED BEATS TO NEARBY MUSICAL HITS
# =========================================================

def snap_times_to_onsets(
    selected_times,
    onset_envelope,
    sr,
    beat_period
):

    """
    After choosing the general musical beat, look very
    close around it for the actual transient.

    Example:

        beat estimate = 31.35
        piano hit     = 31.46

    We may move the cut to 31.46.

    We keep this search small so we do not jump to a
    completely different beat.
    """

    if len(
        selected_times
    ) == 0:

        return [], 0


    normalized_onset = normalize_feature(
        onset_envelope
    )


    # Roughly ±120 ms.
    #
    # On very fast songs the search window scales down
    # so we don't accidentally jump into another beat.
    snap_window_seconds = min(
        0.120,
        beat_period * 0.30
    )


    snap_window_frames = max(
        1,
        int(
            round(
                snap_window_seconds
                * sr
                /
                HOP_LENGTH
            )
        )
    )


    snapped_times = []

    snap_count = 0


    for original_time in selected_times:


        center_frame = int(
            round(
                original_time
                * sr
                /
                HOP_LENGTH
            )
        )


        center_frame = int(
            np.clip(
                center_frame,
                0,
                len(
                    normalized_onset
                )
                - 1
            )
        )


        start = max(
            0,
            center_frame
            -
            snap_window_frames
        )


        end = min(
            len(
                normalized_onset
            ),
            center_frame
            +
            snap_window_frames
            +
            1
        )


        local_curve = (
            normalized_onset[
                start:end
            ]
        )


        if len(
            local_curve
        ) == 0:


            snapped_times.append(
                original_time
            )

            continue


        local_offset = int(
            np.argmax(
                local_curve
            )
        )


        strongest_frame = (
            start
            +
            local_offset
        )


        strongest_strength = float(
            normalized_onset[
                strongest_frame
            ]
        )


        original_strength = float(
            normalized_onset[
                center_frame
            ]
        )


        strongest_time = float(
            librosa.frames_to_time(
                strongest_frame,
                sr=sr,
                hop_length=HOP_LENGTH
            )
        )


        time_difference = abs(
            strongest_time
            -
            original_time
        )


        # Only move when the nearby transient is genuinely
        # more convincing than the current location.
        should_snap = (

            strongest_strength
            >
            original_strength
            +
            0.08

            and

            time_difference
            <=
            snap_window_seconds
            +
            0.001

        )


        if should_snap:


            snapped_times.append(
                strongest_time
            )


            snap_count += 1


        else:


            snapped_times.append(
                original_time
            )


    return (
        snapped_times,
        snap_count
    )


# =========================================================
# CLEAN RESULTING CUTS
# =========================================================

def clean_selected_times(
    selected_times,
    beat_period,
    interval
):

    """
    Snapping may occasionally cause two neighbouring
    points to become too close.

    Remove only obviously bad collisions.
    """

    if len(
        selected_times
    ) == 0:

        return []


    selected_times = sorted(
        selected_times
    )


    expected_gap = (
        beat_period
        *
        interval
    )


    # We already intentionally allow 3/5 or 1/3 beat
    # spacing. This threshold only catches genuine
    # accidental collisions.
    minimum_gap = (
        expected_gap
        *
        0.38
    )


    cleaned = [
        selected_times[0]
    ]


    for current_time in selected_times[1:]:


        if (
            current_time
            -
            cleaned[-1]
            >=
            minimum_gap
        ):


            cleaned.append(
                current_time
            )


    return cleaned


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

    # Every 4 beats remains the default.
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
# GENERATE
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
                # 1. CHECK AUDIO
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


                # =================================================
                # 2. PREPARE ANALYSIS AUDIO
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
                # 3. MUSIC ANALYSIS
                # =================================================

                status.write(
                    "3/5 · Finding strong "
                    "editorial beats"
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
                        hop_length=HOP_LENGTH
                    )
                )


                _, beat_frames = (
                    librosa.beat.beat_track(
                        onset_envelope=(
                            onset_envelope
                        ),
                        sr=sr,
                        hop_length=HOP_LENGTH
                    )
                )


                beat_times = (
                    librosa.frames_to_time(
                        beat_frames,
                        sr=sr,
                        hop_length=HOP_LENGTH
                    )
                )


                if len(
                    beat_times
                ) == 0:


                    raise RuntimeError(
                        "No reliable beats were "
                        "detected in this track."
                    )


                if len(
                    beat_times
                ) > 1:


                    beat_period = float(
                        np.median(
                            np.diff(
                                beat_times
                            )
                        )
                    )


                else:


                    beat_period = 0.5


                strength_curve = (
                    build_editorial_strength_curve(
                        y=y,
                        sr=sr,
                        onset_envelope=(
                            onset_envelope
                        )
                    )
                )


                beat_strengths = (
                    get_beat_strengths(
                        beat_times=(
                            beat_times
                        ),
                        strength_curve=(
                            strength_curve
                        ),
                        sr=sr
                    )
                )


                # -------------------------------------------------
                # EVERY BEAT
                # -------------------------------------------------

                if (
                    BEAT_INTERVAL
                    ==
                    1
                ):


                    selected_times = [
                        float(
                            beat
                        )
                        for beat
                        in beat_times
                    ]


                    snap_count = 0


                # -------------------------------------------------
                # EVERY 2 / EVERY 4
                # -------------------------------------------------

                else:


                    selected_indices = (
                        optimise_edit_path(
                            beat_times=(
                                beat_times
                            ),
                            beat_strengths=(
                                beat_strengths
                            ),
                            interval=(
                                BEAT_INTERVAL
                            )
                        )
                    )


                    selected_times = [
                        float(
                            beat_times[
                                index
                            ]
                        )
                        for index
                        in selected_indices
                    ]


                    (
                        selected_times,
                        snap_count
                    ) = (
                        snap_times_to_onsets(
                            selected_times=(
                                selected_times
                            ),
                            onset_envelope=(
                                onset_envelope
                            ),
                            sr=sr,
                            beat_period=(
                                beat_period
                            )
                        )
                    )


                    selected_times = (
                        clean_selected_times(
                            selected_times=(
                                selected_times
                            ),
                            beat_period=(
                                beat_period
                            ),
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
                        f"{len(selected_times)} "
                        f"optimized edit points"
                    )


                    status.write(
                        f"Snapped "
                        f"{snap_count} "
                        f"cuts to stronger nearby hits"
                    )


                del y
                del onset_envelope
                del strength_curve
                del beat_strengths


                # =================================================
                # CONVERT TO VIDEO FRAMES
                # =================================================

                total_video_frames = (
                    math.ceil(
                        total_duration
                        *
                        FPS_VALUE
                    )
                )


                scene_change_frames = []


                for beat_time in selected_times:


                    frame_number = (
                        beat_time_to_frame(
                            float(
                                beat_time
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
                # 4. BUILD RAW VIDEO
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
                # 5. RENDER MP4
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


            # =====================================================
            # FINISHED
            # =====================================================

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
