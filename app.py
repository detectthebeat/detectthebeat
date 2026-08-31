import streamlit as st
import librosa
import tempfile
import subprocess
import os
from pathlib import Path


# --------------------------------------------------
# APP SETTINGS
# --------------------------------------------------

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

MAX_AUDIO_DURATION = 6 * 60  # 6 minutes

# High-contrast colors for reliable scene detection
COLOR_PALETTE = [
    ("red", (255, 0, 0)),
    ("blue", (0, 0, 255)),
    ("yellow", (255, 255, 0)),
    ("green", (0, 255, 0)),
]

VIDEO_SIZES = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
}

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


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def run_command(command, cwd=None):
    """
    Run a command and raise a useful error if it fails.
    """

    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        error_text = result.stderr[-4000:]

        raise RuntimeError(
            f"FFmpeg error:\n{error_text}"
        )

    return result


def get_audio_duration(audio_path):
    """
    Use ffprobe instead of MoviePy to read duration.
    """

    result = run_command([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ])

    return float(result.stdout.strip())


def create_ppm(path, color):
    """
    Create a tiny 2x2 solid-color image.

    FFmpeg later scales this to 1080p.
    Using 2x2 images means almost no memory usage.
    """

    width = 2
    height = 2

    header = f"P6\n{width} {height}\n255\n".encode()

    pixel = bytes(color)
    pixels = pixel * (width * height)

    with open(path, "wb") as file:
        file.write(header)
        file.write(pixels)


def quantize_time_to_frame(time_seconds, fps):
    """
    Move beat times onto actual video frame boundaries.

    This is useful because Premiere can only make cuts
    on video frames anyway.
    """

    frame_number = round(time_seconds * fps)

    return frame_number / fps


# --------------------------------------------------
# UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload audio",
    type=["mp3", "wav", "m4a"]
)

st.caption(
    "MP3, WAV or M4A · Maximum length: 6 minutes"
)


# --------------------------------------------------
# VIDEO SETTINGS
# --------------------------------------------------

st.subheader("Video settings")

format_choice = st.radio(
    "Video format",
    options=[
        "16:9",
        "9:16",
        "1:1"
    ],
    horizontal=True
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
    "Color change",
    options=[
        "Every beat",
        "Every 2 beats",
        "Every 4 beats"
    ]
)

VIDEO_WIDTH, VIDEO_HEIGHT = VIDEO_SIZES[
    format_choice
]

FPS_VALUE = FPS_OPTIONS[
    fps_choice
]["value"]

FPS_FFMPEG = FPS_OPTIONS[
    fps_choice
]["ffmpeg"]

BEAT_INTERVAL = BEAT_INTERVALS[
    beat_choice
]


# --------------------------------------------------
# GENERATE VIDEO
# --------------------------------------------------

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button(
        "🚀 Generate Video",
        type="primary"
    ):

        try:

            status = st.status(
                "Preparing audio...",
                expanded=True
            )

            # Everything happens inside a temporary folder.
            # It is automatically deleted when we're finished.

            with tempfile.TemporaryDirectory() as work_dir:

                work_dir = Path(work_dir)

                # ------------------------------------------
                # 1. SAVE UPLOADED FILE
                # ------------------------------------------

                status.write(
                    "1/4 · Checking audio"
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
                    work_dir /
                    f"input{extension}"
                )

                with open(
                    audio_path,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getvalue()
                    )

                # ------------------------------------------
                # CHECK AUDIO LENGTH
                # ------------------------------------------

                total_duration = get_audio_duration(
                    str(audio_path)
                )

                if total_duration > MAX_AUDIO_DURATION:

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

                # ------------------------------------------
                # 2. CREATE LIGHTWEIGHT ANALYSIS AUDIO
                # ------------------------------------------

                status.write(
                    "2/4 · Preparing audio for beat detection"
                )

                analysis_path = (
                    work_dir /
                    "analysis.wav"
                )

                # Convert to:
                # mono
                # 22050 Hz
                #
                # This keeps beat analysis lightweight
                # while leaving the ORIGINAL audio untouched
                # for the final video.

                run_command([
                    "ffmpeg",
                    "-y",
                    "-loglevel", "error",
                    "-i", str(audio_path),
                    "-vn",
                    "-ac", "1",
                    "-ar", "22050",
                    str(analysis_path)
                ])

                # ------------------------------------------
                # 3. DETECT BEATS
                # ------------------------------------------

                status.write(
                    "3/4 · Detecting beats"
                )

                y, sr = librosa.load(
                    str(analysis_path),
                    sr=None,
                    mono=True
                )

                onset_envelope = (
                    librosa.onset.onset_strength(
                        y=y,
                        sr=sr
                    )
                )

                tempo, beat_frames = (
                    librosa.beat.beat_track(
                        onset_envelope=onset_envelope,
                        sr=sr
                    )
                )

                beat_times = (
                    librosa.frames_to_time(
                        beat_frames,
                        sr=sr
                    )
                )

                # Free the large audio arrays immediately

                del y
                del onset_envelope

                # Use every 1st / 2nd / 4th beat

                selected_beats = (
                    beat_times[
                        ::BEAT_INTERVAL
                    ]
                )

                # ------------------------------------------
                # SNAP BEATS TO VIDEO FRAMES
                # ------------------------------------------

                frame_aligned_beats = []

                for beat in selected_beats:

                    frame_time = (
                        quantize_time_to_frame(
                            float(beat),
                            FPS_VALUE
                        )
                    )

                    if (
                        frame_time > 0
                        and
                        frame_time < total_duration
                    ):

                        frame_aligned_beats.append(
                            frame_time
                        )

                # Remove duplicates caused by two detected
                # beats landing on the same video frame.

                frame_aligned_beats = sorted(
                    set(frame_aligned_beats)
                )

                status.write(
                    f"Detected {len(beat_times)} beats"
                )

                status.write(
                    f"Creating "
                    f"{len(frame_aligned_beats)} "
                    f"scene changes"
                )

                # ------------------------------------------
                # 4. CREATE TINY COLOR IMAGES
                # ------------------------------------------

                black_path = (
                    work_dir /
                    "black.ppm"
                )

                create_ppm(
                    black_path,
                    (0, 0, 0)
                )

                color_files = []

                for (
                    color_name,
                    rgb
                ) in COLOR_PALETTE:

                    color_path = (
                        work_dir /
                        f"{color_name}.ppm"
                    )

                    create_ppm(
                        color_path,
                        rgb
                    )

                    color_files.append(
                        color_path
                    )

                # ------------------------------------------
                # BUILD VIDEO TIMELINE
                # ------------------------------------------

                segments = []

                if len(
                    frame_aligned_beats
                ) == 0:

                    segments.append(
                        (
                            black_path,
                            total_duration
                        )
                    )

                else:

                    first_beat = (
                        frame_aligned_beats[0]
                    )

                    # Black before first detected beat

                    if first_beat > 0:

                        segments.append(
                            (
                                black_path,
                                first_beat
                            )
                        )

                    # Colors from beat to beat

                    for index, start in enumerate(
                        frame_aligned_beats
                    ):

                        if (
                            index + 1
                            <
                            len(
                                frame_aligned_beats
                            )
                        ):

                            end = (
                                frame_aligned_beats[
                                    index + 1
                                ]
                            )

                        else:

                            end = total_duration

                        duration = (
                            end - start
                        )

                        if duration <= 0:
                            continue

                        color_file = (
                            color_files[
                                index
                                %
                                len(color_files)
                            ]
                        )

                        segments.append(
                            (
                                color_file,
                                duration
                            )
                        )

                # ------------------------------------------
                # CREATE FFMPEG CONCAT FILE
                # ------------------------------------------

                timeline_path = (
                    work_dir /
                    "timeline.txt"
                )

                with open(
                    timeline_path,
                    "w"
                ) as timeline:

                    for (
                        image_path,
                        duration
                    ) in segments:

                        timeline.write(
                            f"file "
                            f"'{image_path.name}'\n"
                        )

                        timeline.write(
                            f"duration "
                            f"{duration:.9f}\n"
                        )

                    # FFmpeg concat requires the final
                    # image to be listed again so the
                    # previous duration is respected.

                    if segments:

                        timeline.write(
                            f"file "
                            f"'{segments[-1][0].name}'\n"
                        )

                # ------------------------------------------
                # RENDER FINAL VIDEO
                # ------------------------------------------

                status.write(
                    "4/4 · Rendering video"
                )

                output_path = (
                    work_dir /
                    "detectthebeat_video.mp4"
                )

                run_command(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel", "error",

                        # Color timeline
                        "-f", "concat",
                        "-safe", "0",
                        "-i", "timeline.txt",

                        # Original uploaded audio
                        "-i", str(audio_path),

                        # Video
                        "-map", "0:v:0",

                        # Audio
                        "-map", "1:a:0",

                        # Scale tiny images to requested format
                        "-vf",
                        (
                            f"scale="
                            f"{VIDEO_WIDTH}:"
                            f"{VIDEO_HEIGHT}:"
                            f"flags=neighbor,"
                            f"format=yuv420p"
                        ),

                        # Constant frame rate
                        "-r", FPS_FFMPEG,

                        # H.264
                        "-c:v", "libx264",

                        # Fast rendering
                        "-preset", "ultrafast",

                        # Good quality
                        "-crf", "18",

                        # AAC audio
                        "-c:a", "aac",
                        "-b:a", "192k",

                        # Stop at end of audio
                        "-shortest",

                        # Better browser compatibility
                        "-movflags", "+faststart",

                        str(output_path)
                    ],
                    cwd=str(work_dir)
                )

                # Read finished file into memory.
                # Temporary working files can then disappear.

                video_bytes = (
                    output_path.read_bytes()
                )

            # ------------------------------------------
            # DONE
            # ------------------------------------------

            status.update(
                label="Video ready!",
                state="complete",
                expanded=False
            )

            st.success(
                f"Created "
                f"{len(frame_aligned_beats)} "
                f"scene changes."
            )

            st.video(
                video_bytes
            )

            st.download_button(
                label="📥 Download Video",
                data=video_bytes,
                file_name=(
                    "detectthebeat_video.mp4"
                ),
                mime="video/mp4"
            )

            st.info(
                "Import the MP4 into your editing software "
                "and use Scene Edit Detection to create cuts "
                "at the color changes."
            )

        except Exception as error:

            try:
                status.update(
                    label="Something went wrong",
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
