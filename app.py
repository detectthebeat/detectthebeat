import streamlit as st
import librosa
import tempfile
import subprocess
import math
import re
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

VIDEO_SIZES = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
}

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


def make_output_filename(original_name, beat_choice, format_choice):
    # Remove original audio extension
    song_name = Path(original_name).stem

    # Remove characters that can cause filename problems
    song_name = re.sub(r'[<>:"/\\|?*]', '', song_name).strip()

    beat_labels = {
        "Every beat": "EveryBeat",
        "Every 2 beats": "Every2Beats",
        "Every 4 beats": "Every4Beats",
    }

    format_labels = {
        "16:9": "16x9",
        "9:16": "9x16",
        "1:1": "1x1",
    }

    return (
        f"{song_name}_DetectTheBeat_"
        f"{beat_labels[beat_choice]}_"
        f"{format_labels[format_choice]}.mp4"
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

st.caption("MP3, WAV or M4A · Maximum length: 6 minutes")


st.subheader("Video settings")

format_choice = st.radio(
    "Video format",
    options=["16:9", "9:16", "1:1"],
    horizontal=True
)

fps_choice = st.radio(
    "Frame rate",
    options=["25 fps", "23.976 fps"],
    horizontal=True
)

beat_choice = st.radio(
    "Scene change",
    options=["Every beat", "Every 2 beats", "Every 4 beats"]
)


VIDEO_WIDTH, VIDEO_HEIGHT = VIDEO_SIZES[format_choice]
FPS_VALUE = FPS_OPTIONS[fps_choice]["value"]
FPS_FFMPEG = FPS_OPTIONS[fps_choice]["ffmpeg"]
BEAT_INTERVAL = BEAT_INTERVALS[beat_choice]


if uploaded_file is not None:
    st.audio(uploaded_file)

    DOWNLOAD_FILENAME = make_output_filename(
        uploaded_file.name,
        beat_choice,
        format_choice
    )

    if st.button("🚀 Generate Video", type="primary"):
        status = None

        try:
            status = st.status(
                "Preparing audio...",
                expanded=True
            )

            with tempfile.TemporaryDirectory() as work_dir:
                work_dir = Path(work_dir)

                status.write("1/5 · Checking audio")

                extension = Path(
                    uploaded_file.name
                ).suffix.lower()

                if extension not in [".mp3", ".wav", ".m4a"]:
                    extension = ".mp3"

                audio_path = work_dir / f"input{extension}"

                with open(audio_path, "wb") as file:
                    file.write(uploaded_file.getvalue())

                total_duration = get_audio_duration(
                    str(audio_path)
                )

                if total_duration > MAX_AUDIO_DURATION:
                    status.update(
                        label="Audio is too long",
                        state="error"
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

                status.write(
                    "2/5 · Preparing audio for beat detection"
                )

                analysis_path = work_dir / "analysis.wav"

                run_command([
                    FFMPEG_EXE,
                    "-y",
                    "-loglevel", "error",
                    "-i", str(audio_path),
                    "-vn",
                    "-ac", "1",
                    "-ar", "22050",
                    str(analysis_path)
                ])

                status.write("3/5 · Detecting beats")

                y, sr = librosa.load(
                    str(analysis_path),
                    sr=None,
                    mono=True
                )

                onset_envelope = librosa.onset.onset_strength(
                    y=y,
                    sr=sr
                )

                _, beat_frames = librosa.beat.beat_track(
                    onset_envelope=onset_envelope,
                    sr=sr
                )

                beat_times = librosa.frames_to_time(
                    beat_frames,
                    sr=sr
                )

                del y
                del onset_envelope

                selected_beats = beat_times[::BEAT_INTERVAL]

                total_video_frames = math.ceil(
                    total_duration * FPS_VALUE
                )

                scene_change_frames = []

                for beat in selected_beats:
                    frame_number = beat_time_to_frame(
                        float(beat),
                        FPS_VALUE
                    )

                    if (
                        frame_number > 0
                        and frame_number < total_video_frames
                    ):
                        scene_change_frames.append(
                            frame_number
                        )

                scene_change_frames = sorted(
                    set(scene_change_frames)
                )

                status.write(
                    f"Detected {len(beat_times)} beats"
                )

                status.write(
                    f"Creating {len(scene_change_frames)} "
                    "scene changes"
                )

                status.write(
                    "4/5 · Building frame-accurate video"
                )

                raw_video_path = (
                    work_dir / "reference.rgb"
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
                            < len(scene_change_frames)
                            and frame_number
                            >= scene_change_frames[
                                change_index
                            ]
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

                status.write("5/5 · Rendering video")

                output_path = (
                    work_dir / "detectthebeat_video.mp4"
                )

                run_command([
                    FFMPEG_EXE,
                    "-y",
                    "-loglevel", "error",
                    "-f", "rawvideo",
                    "-pix_fmt", "rgb24",
                    "-s:v",
                    f"{INTERNAL_WIDTH}x{INTERNAL_HEIGHT}",
                    "-r", FPS_FFMPEG,
                    "-i", str(raw_video_path),
                    "-i", str(audio_path),
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-vf",
                    (
                        f"scale={VIDEO_WIDTH}:"
                        f"{VIDEO_HEIGHT}:"
                        f"flags=neighbor,"
                        f"format=yuv420p"
                    ),
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "18",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    "-movflags", "+faststart",
                    str(output_path)
                ])

                video_bytes = output_path.read_bytes()

            status.update(
                label="Video ready!",
                state="complete",
                expanded=False
            )

            st.success(
                f"Created {len(scene_change_frames)} "
                "scene changes."
            )

            st.video(video_bytes)

            st.download_button(
                label="📥 Download Video",
                data=video_bytes,
                file_name=DOWNLOAD_FILENAME,
                mime="video/mp4"
            )

            st.info(
                "Import the MP4 into your editing software "
                "and use Scene Edit Detection to create cuts "
                "at the checkerboard changes."
            )

        except Exception as error:
            if status is not None:
                try:
                    status.update(
                        label="Something went wrong",
                        state="error"
                    )
                except Exception:
                    pass

            st.error("Video generation failed.")
            st.code(str(error))
