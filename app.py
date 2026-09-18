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

# Keep one rhythm interpretation for roughly a phrase.
PHRASE_BEATS = 16

# How much better a new phrase alignment must be
# before DetectTheBeat changes rhythm interpretation.
SWITCH_MARGIN = 0.10

# A detected hit must be reasonably close to a musical beat.
MAX_BEAT_ALIGNMENT_SECONDS = 0.14

# Minimum strength for something to be treated as
# an interesting editorial hit.
MIN_HIT_STRENGTH = 0.24

# If an earlier hit is at least this fraction as strong
# as the strongest hit in the same small cluster,
# give the earlier hit a chance.
EARLY_HIT_RATIO = 0.72


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
# AUDIO HELPERS
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


def sample_curve_at_time(
    curve,
    time_seconds,
    sr,
    radius=1
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
# MUSICAL STRENGTH
# =========================================================

def build_musical_strength_curve(
    y,
    sr,
    onset_envelope
):

    """
    Continuous estimate of how useful a moment might
    be as an editorial musical hit.

    Onset strength remains the most important part.

    We also include:
    - bass / kick energy
    - higher-frequency attacks such as piano, guitar,
      claps and cymbals
    - overall energy
    """

    onset_curve = normalize_feature(
        onset_envelope
    )


    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=HOP_LENGTH,
        n_mels=48,
        fmin=30,
        fmax=9000,
        power=1.0
    )


    mel_frequencies = (
        librosa.mel_frequencies(
            n_mels=48,
            fmin=30,
            fmax=9000
        )
    )


    # -----------------------------------------------------
    # BASS / KICK
    # -----------------------------------------------------

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
    # MID / HIGH ATTACKS
    #
    # Important for piano notes like the Heron Vale
    # examples.
    # -----------------------------------------------------

    upper_mask = (
        mel_frequencies >= 600
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


    strength_curve = (

        0.58
        * onset_curve

        +

        0.18
        * bass_curve

        +

        0.16
        * upper_curve

        +

        0.08
        * rms_curve

    )


    return normalize_feature(
        strength_curve
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
    Allow the beat grid to shift slightly so a pickup
    note does not automatically determine the whole
    song's rhythm.
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
        maximum_shift
        + analysis_step / 2,
        analysis_step
    )


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
# PHRASE-LOCKED RHYTHM
# =========================================================

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
    strength_curve,
    sr,
    block_start,
    block_end,
    interval,
    phase,
    offset,
    ignore_first_beats=0
):

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


        strength = sample_curve_at_time(
            strength_curve,
            shifted_time,
            sr,
            radius=1
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
        * median_strength

        +

        0.25
        * mean_strength

        +

        0.25
        * lower_strength

    )


    beat_period = get_median_beat_period(
        beat_times
    )


    if beat_period > 0:

        score -= (
            0.025
            * abs(
                offset
            )
            / beat_period
        )


    return score


def find_best_phrase_state(
    beat_times,
    strength_curve,
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
                strength_curve=strength_curve,
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


def create_phrase_locked_slots(
    beat_times,
    strength_curve,
    sr,
    interval
):

    """
    This is the rhythm system from the previous version.

    It first establishes a stable repeating rhythm,
    then only changes interpretation at phrase boundaries
    when the alternative is clearly better.
    """

    beat_count = len(
        beat_times
    )


    if beat_count == 0:

        return [], []


    if interval == 1:


        slots = []


        for index, beat in enumerate(
            beat_times
        ):


            slots.append(
                {
                    "time": float(
                        beat
                    ),
                    "strength": sample_curve_at_time(
                        strength_curve,
                        float(
                            beat
                        ),
                        sr
                    ),
                    "beat_index": index,
                    "block": 0
                }
            )


        return (
            slots,
            []
        )


    beat_period = get_median_beat_period(
        beat_times
    )


    offset_candidates = (
        create_offset_candidates(
            beat_period,
            sr
        )
    )


    slots = []
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
        # FIRST PHRASE
        #
        # Ignore the first two detected beats while
        # deciding alignment. This prevents a pickup note
        # from dominating.
        # -------------------------------------------------

        if block_number == 0:


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
                strength_curve=strength_curve,
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
                strength_curve=strength_curve,
                sr=sr,
                block_start=block_start,
                block_end=block_end,
                interval=interval,
                offset_candidates=(
                    offset_candidates
                ),
                ignore_first_beats=0
            )


            previous_score = (
                score_phrase_state(
                    beat_times=beat_times,
                    strength_curve=strength_curve,
                    sr=sr,
                    block_start=block_start,
                    block_end=block_end,
                    interval=interval,
                    phase=previous_phase,
                    offset=previous_offset,
                    ignore_first_beats=0
                )
            )


            offset_change = abs(
                best_offset
                -
                previous_offset
            )


            offset_change_penalty = (

                0.05
                * offset_change

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


            slots.append(
                {
                    "time": shifted_time,
                    "strength": sample_curve_at_time(
                        strength_curve,
                        shifted_time,
                        sr
                    ),
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


    slots = sorted(
        slots,
        key=lambda item: item[
            "time"
        ]
    )


    return (
        slots,
        phrase_states
    )


# =========================================================
# DETECT MUSICAL HIT CANDIDATES
# =========================================================

def detect_musical_hits(
    onset_envelope,
    strength_curve,
    sr
):

    """
    Find actual musical events independently of the
    planned editing rhythm.

    We intentionally use a fairly sensitive peak detector
    so smaller build-up hits are not automatically ignored.
    """

    normalized_onset = normalize_feature(
        onset_envelope
    )


    peak_frames = librosa.util.peak_pick(
        normalized_onset,
        pre_max=2,
        post_max=2,
        pre_avg=4,
        post_avg=4,
        delta=0.04,
        wait=2
    )


    hits = []


    for frame in peak_frames:


        time_seconds = float(
            librosa.frames_to_time(
                frame,
                sr=sr,
                hop_length=HOP_LENGTH
            )
        )


        onset_strength = float(
            normalized_onset[
                frame
            ]
        )


        musical_strength = (
            sample_curve_at_time(
                strength_curve,
                time_seconds,
                sr,
                radius=1
            )
        )


        combined_strength = (

            0.65
            * musical_strength

            +

            0.35
            * onset_strength

        )


        if (
            combined_strength
            >=
            MIN_HIT_STRENGTH
        ):


            hits.append(
                {
                    "time": time_seconds,
                    "strength": combined_strength,
                    "onset_strength": onset_strength
                }
            )


    return hits


# =========================================================
# FILTER HITS TO MUSICAL BEAT GRID
# =========================================================

def get_phrase_state_for_beat(
    beat_index,
    phrase_states
):

    if len(
        phrase_states
    ) == 0:

        return None


    block_number = (
        beat_index
        //
        PHRASE_BEATS
    )


    block_number = min(
        block_number,
        len(
            phrase_states
        )
        - 1
    )


    return phrase_states[
        block_number
    ]


def filter_hits_near_beat_grid(
    hits,
    beat_times,
    phrase_states,
    beat_period,
    interval
):

    """
    Keep musical hits only when they are reasonably
    close to the beat grid.

    This prevents a random off-beat sound from becoming
    an edit point just because it is loud.
    """

    if interval == 1:

        return hits


    valid_hits = []


    tolerance = min(
        MAX_BEAT_ALIGNMENT_SECONDS,
        beat_period * 0.35
    )


    beat_times_array = np.asarray(
        beat_times
    )


    for hit in hits:


        raw_nearest_index = int(
            np.argmin(
                np.abs(
                    beat_times_array
                    -
                    hit["time"]
                )
            )
        )


        state = (
            get_phrase_state_for_beat(
                raw_nearest_index,
                phrase_states
            )
        )


        if state is None:

            shifted_beat_time = float(
                beat_times[
                    raw_nearest_index
                ]
            )

        else:

            shifted_beat_time = (

                float(
                    beat_times[
                        raw_nearest_index
                    ]
                )

                +

                state[
                    "offset"
                ]

            )


        distance = abs(
            hit["time"]
            -
            shifted_beat_time
        )


        if distance <= tolerance:


            hit_copy = dict(
                hit
            )


            hit_copy[
                "beat_distance"
            ] = distance


            hit_copy[
                "nearest_beat_index"
            ] = raw_nearest_index


            valid_hits.append(
                hit_copy
            )


    return valid_hits


# =========================================================
# CHOOSE BETTER HIT FOR EACH PLANNED CUT
# =========================================================

def choose_hit_for_slot(
    slot,
    candidates,
    beat_period,
    interval
):

    """
    One planned cut may have several legitimate musical
    hits nearby.

    We first find the strongest candidate.

    If an earlier candidate in the same local hit cluster
    is nearly as strong, we slightly prefer that earlier hit.

    Example:

        16.3 HIT
        17.0 BIGGER HIT

    If both are good, 16.3 can win.

    If the first event is genuinely weak, 17.0 still wins.
    """

    if len(
        candidates
    ) == 0:

        return slot


    strongest = max(
        candidates,
        key=lambda item: item[
            "strength"
        ]
    )


    chosen = strongest


    # -----------------------------------------------------
    # EARLIER-MEANINGFUL-HIT RULE
    # -----------------------------------------------------

    earlier_candidates = []


    cluster_lookback = min(
        beat_period * 2.0,
        0.95
    )


    for candidate in candidates:


        if (
            candidate["time"]
            <
            strongest["time"]
        ):


            time_difference = (

                strongest[
                    "time"
                ]

                -

                candidate[
                    "time"
                ]

            )


            strong_enough = (

                candidate[
                    "strength"
                ]

                >=

                max(
                    0.32,
                    strongest[
                        "strength"
                    ]
                    *
                    EARLY_HIT_RATIO
                )

            )


            if (
                time_difference
                <=
                cluster_lookback
                and
                strong_enough
            ):


                earlier_candidates.append(
                    candidate
                )


    if earlier_candidates:


        # Prefer the earliest clearly meaningful event
        # within the local cluster.
        chosen = min(
            earlier_candidates,
            key=lambda item: item[
                "time"
            ]
        )


    # -----------------------------------------------------
    # DON'T MOVE FOR A WORSE RESULT
    # -----------------------------------------------------

    # If the existing phrase-locked cut is already strong,
    # the new candidate needs to be reasonably convincing.
    if (
        chosen[
            "strength"
        ]
        <
        slot[
            "strength"
        ]
        -
        0.05
    ):


        return slot


    return {
        "time": chosen[
            "time"
        ],
        "strength": chosen[
            "strength"
        ],
        "beat_index": chosen.get(
            "nearest_beat_index",
            slot[
                "beat_index"
            ]
        ),
        "block": slot[
            "block"
        ],
        "moved_to_hit": True
    }


def refine_slots_with_musical_hits(
    slots,
    valid_hits,
    beat_period,
    interval
):

    """
    Keep the stable phrase-based rhythm, but let each
    planned cut move onto a more useful musical hit.

    Every 2 remains deliberately conservative.

    Every 4 is allowed to search more broadly because
    there is more space between cuts.
    """

    if (
        interval == 1
        or
        len(
            slots
        ) == 0
        or
        len(
            valid_hits
        ) == 0
    ):

        return (
            slots,
            0
        )


    refined = []

    move_count = 0


    slot_times = [
        slot[
            "time"
        ]
        for slot in slots
    ]


    for index, slot in enumerate(
        slots
    ):


        # -------------------------------------------------
        # DEFINE THIS CUT'S SEARCH REGION
        # -------------------------------------------------

        if index == 0:


            previous_midpoint = (
                slot[
                    "time"
                ]
                -
                beat_period
                *
                interval
                /
                2
            )

        else:


            previous_midpoint = (

                slot_times[
                    index - 1
                ]

                +

                slot[
                    "time"
                ]

            ) / 2


        if (
            index
            ==
            len(
                slots
            )
            -
            1
        ):


            next_midpoint = (
                slot[
                    "time"
                ]
                +
                beat_period
                *
                interval
                /
                2
            )

        else:


            next_midpoint = (

                slot[
                    "time"
                ]

                +

                slot_times[
                    index + 1
                ]

            ) / 2


        # -------------------------------------------------
        # HOW FAR ARE WE WILLING TO MOVE?
        # -------------------------------------------------

        if interval == 2:


            # Every 2 was already working well.
            # Keep this fairly conservative.
            max_shift = (
                beat_period
                *
                0.70
            )


        else:


            # Every 4 can search much more broadly for
            # editorially useful hits.
            max_shift = (
                beat_period
                *
                1.80
            )


        search_start = max(
            previous_midpoint,
            slot[
                "time"
            ]
            -
            max_shift
        )


        search_end = min(
            next_midpoint,
            slot[
                "time"
            ]
            +
            max_shift
        )


        candidates = []


        for hit in valid_hits:


            if (
                hit[
                    "time"
                ]
                >=
                search_start
                and
                hit[
                    "time"
                ]
                <
                search_end
            ):


                candidates.append(
                    hit
                )


        chosen = choose_hit_for_slot(
            slot=slot,
            candidates=candidates,
            beat_period=beat_period,
            interval=interval
        )


        if (
            abs(
                chosen[
                    "time"
                ]
                -
                slot[
                    "time"
                ]
            )
            >
            0.015
        ):


            move_count += 1


        refined.append(
            chosen
        )


    return (
        refined,
        move_count
    )


# =========================================================
# CLEAN PHRASE BOUNDARIES
# =========================================================

def clean_cut_records(
    records,
    beat_period,
    interval
):

    if len(
        records
    ) == 0:

        return []


    records = sorted(
        records,
        key=lambda item: item[
            "time"
        ]
    )


    expected_gap = (
        beat_period
        *
        interval
    )


    # This catches only genuinely accidental collisions.
    minimum_gap = (
        expected_gap
        *
        0.42
    )


    cleaned = []


    for record in records:


        if not cleaned:


            cleaned.append(
                record
            )

            continue


        gap = (

            record[
                "time"
            ]

            -

            cleaned[
                -1
            ][
                "time"
            ]

        )


        if gap < minimum_gap:


            # If two chosen cuts accidentally collide,
            # keep whichever musical event is stronger.
            if (
                record[
                    "strength"
                ]
                >
                cleaned[
                    -1
                ][
                    "strength"
                ]
            ):


                cleaned[
                    -1
                ] = record


        else:


            cleaned.append(
                record
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

    # Every 4 beats is the default.
    index=2
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


# =========================================================
# GENERATE VIDEO
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
                    "and musical hits"
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


                beat_period = (
                    get_median_beat_period(
                        beat_times
                    )
                )


                strength_curve = (
                    build_musical_strength_curve(
                        y=y,
                        sr=sr,
                        onset_envelope=(
                            onset_envelope
                        )
                    )
                )


                # -------------------------------------------------
                # STABLE PHRASE-LOCKED RHYTHM
                # -------------------------------------------------

                (
                    base_slots,
                    phrase_states
                ) = (
                    create_phrase_locked_slots(
                        beat_times=beat_times,
                        strength_curve=(
                            strength_curve
                        ),
                        sr=sr,
                        interval=(
                            BEAT_INTERVAL
                        )
                    )
                )


                # -------------------------------------------------
                # ACTUAL MUSICAL HITS
                # -------------------------------------------------

                musical_hits = (
                    detect_musical_hits(
                        onset_envelope=(
                            onset_envelope
                        ),
                        strength_curve=(
                            strength_curve
                        ),
                        sr=sr
                    )
                )


                valid_hits = (
                    filter_hits_near_beat_grid(
                        hits=musical_hits,
                        beat_times=beat_times,
                        phrase_states=phrase_states,
                        beat_period=beat_period,
                        interval=(
                            BEAT_INTERVAL
                        )
                    )
                )


                # -------------------------------------------------
                # MOVE PLANNED CUTS TO BETTER MUSICAL HITS
                # -------------------------------------------------

                (
                    refined_slots,
                    moved_count
                ) = (
                    refine_slots_with_musical_hits(
                        slots=base_slots,
                        valid_hits=valid_hits,
                        beat_period=beat_period,
                        interval=(
                            BEAT_INTERVAL
                        )
                    )
                )


                refined_slots = (
                    clean_cut_records(
                        records=refined_slots,
                        beat_period=beat_period,
                        interval=(
                            BEAT_INTERVAL
                        )
                    )
                )


                selected_times = [
                    record[
                        "time"
                    ]
                    for record
                    in refined_slots
                ]


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
                        f"Found "
                        f"{len(valid_hits)} "
                        f"on-beat musical hits"
                    )


                    status.write(
                        f"Moved "
                        f"{moved_count} "
                        f"planned cuts onto "
                        f"stronger musical hits"
                    )


                del y
                del onset_envelope
                del strength_curve


                # =================================================
                # VIDEO FRAME POSITIONS
                # =================================================

                total_video_frames = (
                    math.ceil(
                        total_duration
                        *
                        FPS_VALUE
                    )
                )


                scene_change_frames = []


                for cut_time in selected_times:


                    frame_number = (
                        beat_time_to_frame(
                            float(
                                cut_time
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
