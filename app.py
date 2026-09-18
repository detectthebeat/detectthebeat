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

MAX_AUDIO_DURATION = 6 * 60

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

FPS_OPTIONS = {
    "25 fps": {"value": 25.0, "ffmpeg": "25"},
    "23.976 fps": {"value": 24000 / 1001, "ffmpeg": "24000/1001"},
}

BEAT_INTERVALS = {
    "Every beat": 1,
    "Every 2 beats": 2,
    "Every 4 beats": 4,
}

INTERNAL_WIDTH = 64
INTERNAL_HEIGHT = 36
HOP_LENGTH = 512


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
        raise RuntimeError(f"FFmpeg error:\n{error_text}")

    return result


def get_audio_duration(audio_path):
    return float(librosa.get_duration(path=audio_path))


def create_solid_frame(width, height, color):
    pixel = bytes(color)
    return pixel * (width * height)


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


def beat_time_to_frame(time_seconds, fps):
    return round(time_seconds * fps)


def normalize_feature(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    low = np.percentile(values, 10)
    high = np.percentile(values, 90)

    if high <= low:
        return np.zeros_like(values)

    normalized = (values - low) / (high - low)

    return np.clip(
        normalized,
        0.0,
        1.0
    )


def get_beat_accent_scores(
    y,
    sr,
    beat_frames,
    onset_envelope
):
    """
    Give every detected beat an accent score.

    The score combines:
    - onset / transient strength
    - low-frequency kick and bass energy

    This helps DetectTheBeat choose the musically stronger
    beat phase instead of blindly starting at the first beat.
    """

    if len(beat_frames) == 0:
        return np.array([])

    valid_beat_frames = np.clip(
        beat_frames,
        0,
        len(onset_envelope) - 1
    )

    onset_at_beats = onset_envelope[
        valid_beat_frames
    ]

    onset_scores = normalize_feature(
        onset_at_beats
    )

    stft = np.abs(
        librosa.stft(
            y,
            n_fft=2048,
            hop_length=HOP_LENGTH
        )
    )

    frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=2048
    )

    bass_mask = (
        (frequencies >= 30)
        & (frequencies <= 180)
    )

    if np.any(bass_mask):
        bass_energy = np.mean(
            stft[bass_mask, :],
            axis=0
        )

        valid_bass_frames = np.clip(
            beat_frames,
            0,
            len(bass_energy) - 1
        )

        bass_at_beats = bass_energy[
            valid_bass_frames
        ]

        bass_scores = normalize_feature(
            bass_at_beats
        )

    else:
        bass_scores = np.zeros_like(
            onset_scores
        )

    accent_scores = (
        0.65 * onset_scores
        + 0.35 * bass_scores
    )

    return accent_scores


def choose_best_phase(
    accent_scores,
    interval
):
    """
    Choose the strongest repeating beat phase.

    For Every 4 Beats, for example:

    Phase 1:
    1, 5, 9, 13...

    Phase 2:
    2, 6, 10, 14...

    Phase 3:
    3, 7, 11, 15...

    Phase 4:
    4, 8, 12, 16...

    DetectTheBeat compares all possible phases and chooses
    the one that most consistently lands on strong accents.
    """

    if (
        interval <= 1
        or len(accent_scores) == 0
    ):
        return 0

    scores = []

    full_groups = (
        len(accent_scores) // interval
    )

    wins = np.zeros(
        interval,
        dtype=float
    )

    if full_groups > 0:

        for group_index in range(
            full_groups
        ):
            start = (
                group_index * interval
            )

            group = accent_scores[
                start:start + interval
            ]

            if len(group) == interval:
                winner = int(
                    np.argmax(group)
                )

                wins[winner] += 1

        wins /= full_groups

    for phase in range(interval):

        phase_values = accent_scores[
            phase::interval
        ]

        if len(phase_values) == 0:
            scores.append(-1.0)
            continue

        mean_strength = float(
            np.mean(phase_values)
        )

        median_strength = float(
            np.median(phase_values)
        )

        if full_groups > 0:
            win_rate = float(
                wins[phase]
            )
        else:
            win_rate = 0.0

        phase_score = (
            0.55 * win_rate
            + 0.30 * mean_strength
            + 0.15 * median_strength
        )

        scores.append(
            phase_score
        )

    return int(
        np.argmax(scores)
    )


def make_output_filename(
    original_name,
    beat_choice
):
    song_name = Path(
        original_name
    ).stem

    # Remove characters that can cause
    # filename problems
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
        f"{song_name}_DetectTheBeat_"
        f"{beat_labels[beat_choice]}.mp4"
    )


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


uploaded_file = st.file_uploader(
    "Upload audio",
    type=["mp3", "wav", "m4a"]
)

st.caption(
    "MP3, WAV or M4A · Maximum length: 6 minutes"
)


st.subheader(
    "Video settings"
)

st.caption(
    "Output format: 16:9 · 1920×1080"
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
    index=2
)


FPS_VALUE = FPS_OPTIONS[
    fps_choice
]["value"]

FPS_FFMPEG = FPS_OPTIONS[
    fps_choice
]["ffmpeg"]

BEAT_INTERVAL = BEAT_INTERVALS[
    beat_choice
]


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
                        str(audio_path)
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
                    f"{minutes}:{seconds:02d}"
                )

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
                    str(analysis_path)
                ])

                status.write(
                    "3/5 · Detecting beats "
                    "and strongest beat phase"
                )

                y, sr = librosa.load(
                    str(analysis_path),
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

                if len(beat_times) == 0:

                    raise RuntimeError(
                        "No reliable beats were "
                        "detected in this track."
                    )

                accent_scores = (
                    get_beat_accent_scores(
                        y=y,
                        sr=sr,
                        beat_frames=beat_frames,
                        onset_envelope=(
                            onset_envelope
                        )
                    )
                )

                best_phase = (
                    choose_best_phase(
                        accent_scores,
                        BEAT_INTERVAL
                    )
                )

                selected_beats = (
                    beat_times[
                        best_phase
                        ::BEAT_INTERVAL
                    ]
                )

                if BEAT_INTERVAL > 1:

                    status.write(
                        f"Selected strongest "
                        f"{BEAT_INTERVAL}-beat "
                        f"phase: "
                        f"{best_phase + 1}/"
                        f"{BEAT_INTERVAL}"
                    )

                del y
                del onset_envelope
                del accent_scores

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
                            float(beat),
                            FPS_VALUE
                        )
                    )

                    if (
                        frame_number > 0
                        and frame_number
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
                    f"Detected "
                    f"{len(beat_times)} beats"
                )

                status.write(
                    f"Creating "
                    f"{len(scene_change_frames)} "
                    f"scene changes"
                )

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
                            pattern_index % 2
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
                    str(raw_video_path),
                    "-i",
                    str(audio_path),
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
                    str(output_path)
                ])

                video_bytes = (
                    output_path.read_bytes()
                )

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
                "Import the MP4 into your "
                "editing software and use "
                "Scene Edit Detection to "
                "create cuts at the "
                "checkerboard changes. "
                "Use your original audio "
                "file for the final edit."
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
