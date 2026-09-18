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


# ---------------------------------------------------------
# PAGE
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# GENERAL FUNCTIONS
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# FILENAME
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# CHECKERBOARD
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# MUSIC ANALYSIS
# ---------------------------------------------------------

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


def sample_feature_at_beats(
    feature,
    beat_frames,
    radius=1
):

    """
    Instead of reading exactly one analysis frame,
    take the strongest value very close to the beat.

    This helps when the detected beat lands a few
    milliseconds before or after the transient.
    """

    results = []

    feature_length = len(
        feature
    )

    for frame in beat_frames:

        frame = int(frame)

        start = max(
            0,
            frame - radius
        )

        end = min(
            feature_length,
            frame + radius + 1
        )

        if end <= start:

            results.append(0.0)

        else:

            results.append(
                float(
                    np.max(
                        feature[
                            start:end
                        ]
                    )
                )
            )

    return np.asarray(
        results
    )


def get_beat_accent_scores(
    y,
    sr,
    beat_frames,
    onset_envelope
):

    """
    Scores each detected beat according to how
    useful it is likely to be as an editing hit.

    We combine:

    55% transient/onset strength
    30% bass/kick energy
    15% overall energy

    We also give a small boost to beats that are
    noticeably stronger than their immediate neighbors.
    """

    if len(beat_frames) == 0:

        return np.array([])


    # -----------------------------------------------------
    # ONSET / TRANSIENT STRENGTH
    # -----------------------------------------------------

    onset_at_beats = (
        sample_feature_at_beats(
            onset_envelope,
            beat_frames,
            radius=1
        )
    )

    onset_scores = normalize_feature(
        onset_at_beats
    )


    # -----------------------------------------------------
    # LOW FREQUENCY / KICK / BASS ENERGY
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

    if np.any(bass_mask):

        bass_energy = np.mean(
            mel[bass_mask, :],
            axis=0
        )

        bass_at_beats = (
            sample_feature_at_beats(
                bass_energy,
                beat_frames,
                radius=1
            )
        )

        bass_scores = (
            normalize_feature(
                bass_at_beats
            )
        )

    else:

        bass_scores = (
            np.zeros_like(
                onset_scores
            )
        )


    # Free mel data as soon as possible
    del mel


    # -----------------------------------------------------
    # GENERAL ENERGY
    # -----------------------------------------------------

    rms = librosa.feature.rms(
        y=y,
        frame_length=1024,
        hop_length=HOP_LENGTH
    )[0]

    rms_at_beats = (
        sample_feature_at_beats(
            rms,
            beat_frames,
            radius=1
        )
    )

    rms_scores = normalize_feature(
        rms_at_beats
    )


    # -----------------------------------------------------
    # COMBINED STRENGTH
    # -----------------------------------------------------

    scores = (
        0.55 * onset_scores
        +
        0.30 * bass_scores
        +
        0.15 * rms_scores
    )


    # -----------------------------------------------------
    # LOCAL PROMINENCE
    #
    # A beat gets a small bonus if it is stronger
    # than the beats directly around it.
    # -----------------------------------------------------

    prominence = np.zeros_like(
        scores
    )

    for i in range(
        len(scores)
    ):

        neighbors = []

        if i > 0:

            neighbors.append(
                scores[i - 1]
            )

        if i < len(scores) - 1:

            neighbors.append(
                scores[i + 1]
            )

        if neighbors:

            neighbor_average = (
                float(
                    np.mean(
                        neighbors
                    )
                )
            )

            prominence[i] = max(
                0.0,
                scores[i]
                - neighbor_average
            )


    prominence = normalize_feature(
        prominence
    )


    scores = (
        0.85 * scores
        +
        0.15 * prominence
    )


    return normalize_feature(
        scores
    )


# ---------------------------------------------------------
# STRONG BEAT SELECTION
# ---------------------------------------------------------

def choose_start_beat(
    accent_scores,
    interval
):

    """
    Choose a strong recurring starting beat.

    We deliberately do NOT simply choose the first
    detected beat.

    The first note of a song may be a pickup,
    intro note or isolated accent.

    We look ahead and favor beats whose strength
    repeats at approximately the requested interval.
    """

    beat_count = len(
        accent_scores
    )

    if beat_count == 0:

        return 0


    search_length = min(
        beat_count,
        max(
            6,
            interval * 2 + 2
        )
    )


    best_index = 0
    best_score = -999.0


    for candidate in range(
        search_length
    ):

        repeated_values = (
            accent_scores[
                candidate
                ::interval
            ][:6]
        )


        if len(
            repeated_values
        ) > 1:

            recurring_median = float(
                np.median(
                    repeated_values
                )
            )

            recurring_mean = float(
                np.mean(
                    repeated_values
                )
            )

        else:

            recurring_median = float(
                accent_scores[
                    candidate
                ]
            )

            recurring_mean = (
                recurring_median
            )


        candidate_score = (

            0.25
            * accent_scores[
                candidate
            ]

            +

            0.45
            * recurring_median

            +

            0.30
            * recurring_mean

            -

            # Small preference for not
            # starting unnecessarily late.
            0.025
            * candidate
        )


        if (
            candidate_score
            > best_score
        ):

            best_score = (
                candidate_score
            )

            best_index = (
                candidate
            )


    return best_index


def select_strong_beat_indices(
    accent_scores,
    interval
):

    """
    Select musically strong editing points while
    keeping approximately regular spacing.

    Every 2 beats:
        normally gap = 2
        can choose 1 or 3 if substantially stronger

    Every 4 beats:
        normally gap = 4
        can choose 3 or 5 if substantially stronger

    This allows DetectTheBeat to follow changes in
    musical accent without becoming completely irregular.
    """

    beat_count = len(
        accent_scores
    )


    if beat_count == 0:

        return []


    # Every beat remains simple and predictable.
    if interval == 1:

        return list(
            range(
                beat_count
            )
        )


    first_index = (
        choose_start_beat(
            accent_scores,
            interval
        )
    )


    selected = [
        first_index
    ]


    current = (
        first_index
    )


    while True:

        target = (
            current
            + interval
        )


        if target >= beat_count:

            break


        minimum_gap = max(
            1,
            interval - 1
        )

        maximum_gap = (
            interval + 1
        )


        candidate_start = (
            current
            + minimum_gap
        )

        candidate_end = min(
            beat_count - 1,
            current
            + maximum_gap
        )


        if (
            candidate_start
            > candidate_end
        ):

            break


        best_candidate = None
        best_value = -999.0


        for candidate in range(
            candidate_start,
            candidate_end + 1
        ):

            gap = (
                candidate
                - current
            )


            strength = float(
                accent_scores[
                    candidate
                ]
            )


            # -------------------------------------------------
            # LOOK AHEAD
            #
            # A beat is more interesting if the same rough
            # pulse continues to be strong afterwards.
            # -------------------------------------------------

            future_values = []

            future_index = (
                candidate
            )

            for _ in range(3):

                if (
                    future_index
                    < beat_count
                ):

                    future_values.append(
                        accent_scores[
                            future_index
                        ]
                    )

                future_index += (
                    interval
                )


            if future_values:

                future_strength = (
                    float(
                        np.mean(
                            future_values
                        )
                    )
                )

            else:

                future_strength = (
                    strength
                )


            # -------------------------------------------------
            # LOCAL PEAK BONUS
            # -------------------------------------------------

            local_start = max(
                0,
                candidate - 1
            )

            local_end = min(
                beat_count,
                candidate + 2
            )


            local_max = float(
                np.max(
                    accent_scores[
                        local_start:
                        local_end
                    ]
                )
            )


            if (
                strength
                >= local_max - 0.01
            ):

                local_peak_bonus = 0.08

            else:

                local_peak_bonus = 0.0


            # -------------------------------------------------
            # SPACING PENALTY
            #
            # A nearby stronger beat is allowed to win,
            # but it has to be clearly better than the
            # normally expected beat.
            # -------------------------------------------------

            distance_from_target = abs(
                gap - interval
            )


            spacing_penalty = (
                0.22
                * distance_from_target
            )


            candidate_value = (

                0.70
                * strength

                +

                0.22
                * future_strength

                +

                local_peak_bonus

                -

                spacing_penalty
            )


            if (
                candidate_value
                > best_value
            ):

                best_value = (
                    candidate_value
                )

                best_candidate = (
                    candidate
                )


        if best_candidate is None:

            break


        selected.append(
            best_candidate
        )


        current = (
            best_candidate
        )


    return selected


# ---------------------------------------------------------
# USER INTERFACE
# ---------------------------------------------------------

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

    # Every 4 beats is now default
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


# ---------------------------------------------------------
# GENERATION
# ---------------------------------------------------------

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


                # ---------------------------------------------
                # STEP 1
                # ---------------------------------------------

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
                    / f"input{extension}"
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
                    > MAX_AUDIO_DURATION
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


                # ---------------------------------------------
                # STEP 2
                # ---------------------------------------------

                status.write(
                    "2/5 · Preparing audio "
                    "for beat detection"
                )


                analysis_path = (
                    work_dir
                    / "analysis.wav"
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
                    str(
                        analysis_path
                    )
                ])


                # ---------------------------------------------
                # STEP 3
                # ---------------------------------------------

                status.write(
                    "3/5 · Detecting beats "
                    "and musical accents"
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
                    == 0
                ):

                    raise RuntimeError(
                        "No reliable beats were "
                        "detected in this track."
                    )


                accent_scores = (
                    get_beat_accent_scores(
                        y=y,
                        sr=sr,
                        beat_frames=(
                            beat_frames
                        ),
                        onset_envelope=(
                            onset_envelope
                        )
                    )
                )


                selected_indices = (
                    select_strong_beat_indices(
                        accent_scores,
                        BEAT_INTERVAL
                    )
                )


                selected_beats = (
                    beat_times[
                        selected_indices
                    ]
                )


                status.write(
                    f"Detected "
                    f"{len(beat_times)} beats"
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
                        f"strong recurring "
                        f"edit points"
                    )


                del y
                del onset_envelope
                del accent_scores


                # ---------------------------------------------
                # FRAME CONVERSION
                # ---------------------------------------------

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
                            FPS_VALUE
                        )
                    )


                    if (
                        frame_number > 0
                        and
                        frame_number
                        < total_video_frames
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


                # ---------------------------------------------
                # STEP 4
                # ---------------------------------------------

                status.write(
                    "4/5 · Building "
                    "frame-accurate video"
                )


                raw_video_path = (
                    work_dir
                    / "reference.rgb"
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
                            < len(
                                scene_change_frames
                            )
                            and
                            frame_number
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


                # ---------------------------------------------
                # STEP 5
                # ---------------------------------------------

                status.write(
                    "5/5 · Rendering video"
                )


                output_path = (
                    work_dir
                    / "detectthebeat_video.mp4"
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
                        f"{INTERNAL_WIDTH}x"
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


            # ---------------------------------------------
            # FINISHED
            # ---------------------------------------------

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
                str(error)
            )
