import streamlit as st
import librosa
import tempfile
import os
import uuid

from moviepy.editor import (
    ColorClip,
    AudioFileClip,
    concatenate_videoclips
)

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
    "generate a beat-synced video, and use Scene Edit Detection in "
    "Premiere Pro or similar editing software."
)

MAX_AUDIO_DURATION = 6 * 60

COLOR_PALETTE = [
    (255, 0, 0),       # Red
    (0, 0, 255),       # Blue
    (255, 255, 0),     # Yellow
    (0, 255, 0),       # Green
]

VIDEO_SIZES = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
}

FPS_OPTIONS = {
    "25 fps": 25,
    "23.976 fps": 24000 / 1001,
}

BEAT_INTERVALS = {
    "Every beat": 1,
    "Every 2 beats": 2,
    "Every 4 beats": 4,
}

# --------------------------------------------------
# UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload audio",
    type=["mp3", "wav", "m4a"]
)

st.caption("MP3, WAV or M4A · Maximum length: 6 minutes")

# --------------------------------------------------
# VIDEO SETTINGS
# --------------------------------------------------

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
    "Color change",
    options=[
        "Every beat",
        "Every 2 beats",
        "Every 4 beats"
    ]
)

VIDEO_SIZE = VIDEO_SIZES[format_choice]
FPS = FPS_OPTIONS[fps_choice]
BEAT_INTERVAL = BEAT_INTERVALS[beat_choice]

# --------------------------------------------------
# GENERATE VIDEO
# --------------------------------------------------

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("🚀 Generate Video", type="primary"):

        audio_path = None
        output_path = None
        audio_clip = None
        final_video = None

        try:

            # Save uploaded audio
            file_extension = os.path.splitext(
                uploaded_file.name
            )[1].lower()

            if not file_extension:
                file_extension = ".mp3"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension
            ) as tmp_audio:

                tmp_audio.write(uploaded_file.getvalue())
                audio_path = tmp_audio.name

            # Check audio duration
            with st.spinner("Checking audio..."):

                audio_clip = AudioFileClip(audio_path)
                total_duration = audio_clip.duration

                if total_duration > MAX_AUDIO_DURATION:
                    st.error(
                        "Audio is too long. Please upload a track "
                        "of 6 minutes or less."
                    )
                    st.stop()

            # Detect beats
            with st.spinner("Detecting beats..."):

                y, sr = librosa.load(
                    audio_path,
                    sr=None,
                    mono=True
                )

                _, beat_frames = librosa.beat.beat_track(
                    y=y,
                    sr=sr
                )

                beat_times = librosa.frames_to_time(
                    beat_frames,
                    sr=sr
                )

                selected_beats = beat_times[::BEAT_INTERVAL]

            # Create video
            clips = []

            if len(selected_beats) == 0:

                clips.append(
                    ColorClip(
                        size=VIDEO_SIZE,
                        color=(0, 0, 0),
                        duration=total_duration
                    )
                )

            else:

                first_beat = float(selected_beats[0])

                # Black before first detected beat
                if first_beat > 0:

                    clips.append(
                        ColorClip(
                            size=VIDEO_SIZE,
                            color=(0, 0, 0),
                            duration=min(
                                first_beat,
                                total_duration
                            )
                        )
                    )

                # Generate one color section per selected beat
                for i, start in enumerate(selected_beats):

                    start = float(start)

                    if start >= total_duration:
                        break

                    if i + 1 < len(selected_beats):
                        end = float(selected_beats[i + 1])
                    else:
                        end = total_duration

                    end = min(end, total_duration)

                    duration = end - start

                    if duration <= 0:
                        continue

                    color = COLOR_PALETTE[
                        i % len(COLOR_PALETTE)
                    ]

                    clips.append(
                        ColorClip(
                            size=VIDEO_SIZE,
                            color=color,
                            duration=duration
                        )
                    )

            if not clips:
                raise RuntimeError(
                    "Could not create video sections."
                )

            final_video = concatenate_videoclips(
                clips,
                method="chain"
            )

            final_video = final_video.set_audio(
                audio_clip
            )

            # Unique temporary output
            output_filename = (
                f"detectthebeat_{uuid.uuid4().hex[:8]}.mp4"
            )

            output_path = os.path.join(
                tempfile.gettempdir(),
                output_filename
            )

            # Render
            with st.spinner("Rendering video..."):

                final_video.write_videofile(
                    output_path,
                    fps=FPS,
                    codec="libx264",
                    audio_codec="aac",
                    preset="medium",
                    threads=4,
                    logger=None
                )

            # Result
            st.success(
                f"Done! Detected {len(beat_times)} beats and "
                f"created {len(selected_beats)} color changes."
            )

            st.video(output_path)

            with open(output_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.download_button(
                label="📥 Download Video",
                data=video_bytes,
                file_name="detectthebeat_video.mp4",
                mime="video/mp4"
            )

            st.info(
                "Import the video into your editing software and "
                "use Scene Edit Detection to create cuts at the "
                "color changes."
            )

        except Exception as e:

            st.error(
                f"Something went wrong while generating the video: {e}"
            )

        finally:

            if final_video is not None:
                final_video.close()

            if audio_clip is not None:
                audio_clip.close()

            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
