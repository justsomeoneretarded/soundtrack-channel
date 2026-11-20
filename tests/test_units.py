import random

import numpy as np
import pytest

from soundtrack.audio import Passage, write_mix
from soundtrack.common import RATE, length_label, timestamp
from soundtrack.metadata import chapters, hashtag, tags, video_title
from soundtrack.sources import cue_labels
from soundtrack.visuals import scene_plan


def test_cue_labels_drop_work_title_and_repeated_artist():
    names = [
        "Hans Zimmer - Cornfield Chase (Interstellar Official Soundtrack)",
        "Hans Zimmer - Day One | Interstellar OST",
        "Interstellar Soundtrack | No Time for Caution – Hans Zimmer | WaterTower",
    ]
    assert cue_labels(names, "Interstellar (2014)") == ["Cornfield Chase", "Day One", "No Time for Caution"]


def test_cue_labels_use_composer_and_fall_back_to_name():
    assert cue_labels(["Ramin Djawadi - Light of the Seven"], "Game of Thrones", "Ramin Djawadi") == [
        "Light of the Seven"]
    assert cue_labels(["Dune Official Soundtrack"], "Dune") == ["Dune Official Soundtrack"]


def test_timestamps_and_lengths():
    assert timestamp(0) == "0:00"
    assert timestamp(605) == "10:05"
    assert timestamp(3725) == "1:02:05"
    assert length_label(3600) == "1 Hour"
    assert length_label(7200) == "2 Hours"
    assert length_label(1800) == "30 Minute"
    assert length_label(45) == "45-Second Preview"


def test_mix_chapters_are_valid_for_youtube():
    result = chapters("mix", [(0.0, 0)], ["A"], 3600)
    assert result[0][0] == 0 and len(result) == 6
    assert all(b[0] - a[0] >= 10 for a, b in zip(result, result[1:]))
    assert chapters("mix", [(0.0, 0)], ["A"], 30) == []


def test_album_chapters_merge_short_and_repeat():
    timeline = [(0.0, 0), (5.0, 1), (100.0, 2), (200.0, 0), (3595.0, 1)]
    result = chapters("album", timeline, ["One", "Two", "Three"], 3600)
    assert result == [(0.0, "Two"), (100.0, "Three"), (200.0, "One")]


def test_titles_stay_within_youtube_limit():
    long_title = "A" * 150
    assert len(video_title(long_title, 3600, "mix")) <= 100
    assert video_title("Dune", 3600, "mix") == "Dune Soundtrack Mix | 1 Hour Cinematic Score"
    assert "Hans Zimmer" in video_title("Dune", 3600, "album", "Hans Zimmer")


def test_tags_and_hashtags():
    values = tags("Dune (2021)", "Hans Zimmer", ["Paul's Dream"], 3600)
    assert values[0] == "Dune" and "Hans Zimmer" in values
    assert sum(len(v) + 1 for v in values) <= 480
    assert hashtag("The Lord of the Rings (2001)") == "#TheLordOfTheRings"


def test_scene_plan_counts():
    footage = [("a.mp4", 120.0, ("a", ""))]
    assert len(scene_plan(footage, 3600)) == 4
    assert len(scene_plan(footage, 3600, 8)) == 8
    assert scene_plan(footage, 3600, 0) == []
    for _, start, length, when in scene_plan(footage, 3600, 8):
        assert 0 <= start and start + length <= 120 and 0 < when < 3600


def _sources():
    rng = np.random.default_rng(0)
    sources, passages = [], []
    for source in range(3):
        audio = (rng.normal(0, 0.05 * (source + 1), (RATE * 40, 2))).astype(np.float32)
        sources.append(audio)
        for start in range(0, 40, 10):
            passages.append(Passage(source, start * RATE, (start + 10) * RATE, 0.05 * (source + 1),
                                    brightness=10.0, level=0.05 * (source + 1)))
    return sources, passages


@pytest.mark.parametrize("style", ["mix", "album"])
def test_write_mix_is_exact_and_bounded(tmp_path, style):
    sources, passages = _sources()
    timeline = write_mix(tmp_path / "mix.pcm", sources, passages, 100, random.Random(1), style)
    audio = np.fromfile(tmp_path / "mix.pcm", dtype="<f4")
    assert len(audio) == 100 * RATE * 2
    assert np.max(np.abs(audio)) <= 0.97
    assert timeline[0][0] == 0
    assert all(b[0] > a[0] for a, b in zip(timeline, timeline[1:]))
    if style == "album":
        # Sources play in order and each runs its full length before the next.
        assert [source for _, source in timeline][:3] == [0, 1, 2]


def test_loudness_gain_caps_limiting():
    from soundtrack.audio import loudness_filter, loudness_gain

    quiet = {"input_i": -24.0, "input_tp": -12.0, "input_lra": 8, "input_thresh": -34, "target_offset": 0}
    assert loudness_gain(quiet) == 10.0
    assert "alimiter" not in loudness_filter(quiet)
    peaky = dict(quiet, input_tp=-2.0)
    assert loudness_gain(peaky) == 3.5  # -1.5 dBTP ceiling plus 3 dB of limiting, not the full 10 dB
    assert "alimiter" in loudness_filter(peaky)
    assert loudness_filter(None) == "anull"
