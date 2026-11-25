import random

import numpy as np
import pytest

from soundtrack.audio import Passage, write_mix
from soundtrack.common import RATE, length_label, timestamp
from soundtrack.metadata import chapters, hashtag, tags, video_title
from soundtrack.sources import cue_labels


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


# --------------------------------------------------------------------------- footage

from pathlib import Path  # noqa: E402

from soundtrack.footage import (Analysis, Face, Video, character_ids, choose, cluster,  # noqa: E402
                                candidate_windows, pick_work)
from soundtrack.visuals import write_timeline  # noqa: E402


def _claim(value):
    return {"mainsnak": {"datavalue": {"value": value}}}


def test_wikidata_work_and_characters():
    entities = {
        "Q1": {"descriptions": {"en": {"value": "album by Hans Zimmer"}}},
        "Q2": {"descriptions": {"en": {"value": "2014 film directed by Christopher Nolan"}},
               "claims": {"P577": [_claim({"time": "+2014-10-26T00:00:00Z"})],
                          "P161": [{"qualifiers": {"P453": [{"datavalue": {"value": {"id": "Q10"}}}]}},
                                   {"qualifiers": {}},
                                   {"qualifiers": {"P453": [{"datavalue": {"value": {"id": "Q11"}}}]}}],
                          "P674": [_claim({"id": "Q10"}), _claim({"id": "Q12"})]}},
    }
    assert pick_work(["Q1", "Q2"], entities) == "Q2"
    assert pick_work(["Q1", "Q2"], entities, "2014") == "Q2"
    assert pick_work(["Q1", "Q2"], entities, "1999") is None
    assert character_ids(entities["Q2"]) == ["Q10", "Q11", "Q12"]


def _embedding(seed):
    vector = np.random.default_rng(seed).normal(size=128).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _face(video, time, identity, area=0.05, noise=0):
    vector = _embedding(identity) + np.random.default_rng(noise).normal(0, 0.05, 128).astype(np.float32)
    return Face(video, time, (200, 60, 120, 120), area, 0.9, 0.9, vector / np.linalg.norm(vector))


def test_cluster_ranks_recurring_people_first():
    faces = [_face(0, t, 1, noise=t) for t in range(3)]
    faces += [_face(v, t, 2, area=0.1, noise=100 + t) for v in (0, 1) for t in range(5)]
    people = cluster(faces)
    assert [count for _, _, count in people] == [10, 3]
    assert {f.person for f in faces[3:]} == {0} and {f.person for f in faces[:3]} == {1}


def _analysis():
    times = np.arange(0, 60, 0.5)
    videos = [Video(Path(f"v{i}.mp4"), 60.0, (f"Video {i}", ""), (1280, 720),
                    [0.0, 6.0, 12.0, 20.0, 30.0, 38.0, 45.0, 60.0], times, np.full(times.shape, 0.6))
              for i in range(2)]
    faces = []
    for t in np.arange(12, 20, 0.5):  # Lead close-ups in video 0, shot 12-20.
        faces.append(Face(0, float(t), (200, 60, 150, 150), 0.1, 0.9, 0.9, None, 0))
    for t in np.arange(30, 38, 0.5):  # Second character in video 1, shot 30-38.
        faces.append(Face(1, float(t), (200, 60, 100, 100), 0.05, 0.9, 0.9, None, 1))
    return Analysis(videos, faces, [(0, 10.0, 16), (1, 5.0, 16)], True)


def test_scene_windows_follow_cuts_and_favour_characters():
    windows = candidate_windows(_analysis())
    assert all(w.end - w.start <= 10.0 for w in windows)
    best = max(windows, key=lambda w: w.score)
    assert (best.video, best.start, best.person) == (0, 12.0, 0)
    picks = choose(windows, 2)
    assert {w.person for w in picks} == {0, 1}  # Variety across characters.


def test_reel_alternates_videos_and_fills_the_video(tmp_path, monkeypatch):
    from soundtrack import reel

    shots = reel.plan_reel(_analysis(), seed=1)
    assert len(shots) >= 4
    counts = [sum(shot.video == video for shot in shots) for video in (0, 1)]
    repeats = sum(a.video == b.video for a, b in zip(shots, shots[1:]))
    assert repeats <= abs(counts[0] - counts[1])  # Neighbours come from different videos.
    assert reel.best_shot(shots).person == 0
    monkeypatch.setattr(reel, "_chunk", lambda group, output, draft, mask: (output.touch(), 40.0)[1])
    monkeypatch.setattr(reel.subprocess, "run", lambda *args, **kwargs: None)
    picture, unique = reel.build_reel(shots * 3, tmp_path, 600, seed=1)
    assert picture == tmp_path / "reel.mp4"
    files = [line for line in (tmp_path / "reel.ffconcat").read_text().splitlines() if line.startswith("file")]
    assert unique == 40.0 * len(set(files)) and len(files) * 40 >= 605
    assert all(a != b for a, b in zip(files, files[1:]))  # No chunk plays twice in a row.


def test_text_timeline_fades_title_and_cues(tmp_path):
    from soundtrack.visuals import split_title, text_timeline

    assert split_title("Pirates of the Caribbean: The Curse of the Black Pearl (2003)") == (
        "PIRATES OF THE CARIBBEAN", "THE CURSE OF THE BLACK PEARL")
    assert split_title("Dune (2021)") == ("DUNE", "")
    listing = text_timeline("Dune (2021)", "Hans Zimmer", [(0.0, "Paul's Dream"), (60.0, "Ripples")],
                            300, tmp_path, (320, 180))
    text = listing.read_text()
    assert text.count("text-") == 3 * 12  # Title and two cues, six steps in and six out.
    assert write_timeline([], 10, tmp_path / "x") is None


def test_cookie_secret_formats(tmp_path):
    import base64
    import json as jsonlib

    from soundtrack.cookies import normalise

    netscape = ("# Netscape HTTP Cookie File\n"
                ".youtube.com\tTRUE\t/\tTRUE\t1999999999\tSAPISID\tabc\n"
                "#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t1999999999\tLOGIN_INFO\txyz\n"
                ".example.com\tTRUE\t/\tFALSE\t0\tother\t1\n")
    text, names = normalise(netscape)
    assert names == {"SAPISID", "LOGIN_INFO"} and text.startswith("# Netscape")
    assert normalise(base64.b64encode(netscape.encode()).decode())[1] == names
    exported = [{"domain": ".youtube.com", "name": "SID", "value": "v", "path": "/", "secure": True,
                 "expirationDate": 1999999999.5}]
    text, names = normalise(jsonlib.dumps(exported))
    assert names == {"SID"} and ".youtube.com\tTRUE\t/\tTRUE\t1999999999\tSID\tv" in text
    assert normalise("not cookies")[1] == set()


def test_search_interleaves_queries(monkeypatch):
    import json as jsonlib
    import subprocess as sp

    from soundtrack import sources

    results = {
        "official trailer": [("aaaaaaaaaa1", "Dune Official Trailer"), ("aaaaaaaaaa2", "Dune Official Trailer 2")],
        "Paul scene": [("bbbbbbbbbb1", "Dune Paul Atreides Scene"), ("aaaaaaaaaa1", "Dune Official Trailer")],
    }

    def fake_run(command, **_):
        query = command[-1].split(" ", 1)[1]
        lines = [jsonlib.dumps({"id": vid, "title": name, "duration": 120}) for vid, name in results[query]]
        return sp.CompletedProcess(command, 0, stdout="\n".join(lines))

    monkeypatch.setattr(sources.subprocess, "run", fake_run)
    found = sources.matching_uploads("Dune", list(results), {"scene", "trailer"}, set(), (20, 900),
                                     interleave=True)
    assert [vid for vid, _ in found] == ["aaaaaaaaaa1", "bbbbbbbbbb1", "aaaaaaaaaa2"]


def test_download_paths_run_with_mocked_ytdlp(monkeypatch, tmp_path):
    """Exercise the YouTube download code that live runs use, without the network."""
    import subprocess as sp

    from soundtrack import footage, sources

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs.get("timeout")))
        target = command[command.index("-o") + 1].replace("%(ext)s", "m4a")
        open(target, "w").close()
        return sp.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(sources.subprocess, "run", fake_run)
    monkeypatch.setattr(sources, "media_length", lambda path: 120.0)
    candidates = [("aaaaaaaaaa1", (5, 120, "Dune Official Soundtrack")),
                  ("aaaaaaaaaa2", (4, 90, "Dune Main Theme"))]
    files, credits = sources.download_uploads(candidates, tmp_path / "a", "bestaudio/best", 8, 2400, 12)
    assert len(files) == 2 and credits[0][1].endswith("aaaaaaaaaa1") and calls[0][1] == 240

    monkeypatch.setattr(footage, "matching_uploads", lambda *args, **kwargs: candidates)
    files, _ = footage.gather("Dune", tmp_path / "v", ["Paul Atreides"])
    assert len(files) == 2 and calls[-1][1] == 420 and "700M" in calls[-1][0]

    monkeypatch.setattr(sources, "matching_uploads", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(sources, "audible_span", lambda path: 120.0)
    files, _ = sources.sources_for("Dune", tmp_path / "s", "Hans Zimmer")
    assert len(files) == 2


def test_clean_composer():
    from soundtrack.common import clean_composer

    assert clean_composer("Composed primarily by Klaus Badelt, with Hans Zimmer") == "Klaus Badelt & Hans Zimmer"
    assert clean_composer("Music by John Williams") == "John Williams"
    assert clean_composer("Hans Zimmer") == "Hans Zimmer"


def test_download_stops_when_youtube_blocks(monkeypatch, tmp_path):
    import subprocess as sp

    from soundtrack import sources

    calls = []

    def blocked(command, **kwargs):
        calls.append(command)
        return sp.CompletedProcess(command, 1, stdout="",
                                   stderr="ERROR: [youtube] x: Sign in to confirm you're not a bot.")

    monkeypatch.setattr(sources.subprocess, "run", blocked)
    candidates = [(f"aaaaaaaaa{i:02d}", (5, 120, f"Dune Soundtrack Part {i}")) for i in range(10)]
    files, _ = sources.download_uploads(candidates, tmp_path, "bestaudio/best", 8, 2400, 10)
    assert files == [] and len(calls) == 3


def test_render_graph_has_no_memory_floods(monkeypatch, tmp_path):
    """Image timelines must not pass through fps, and no input may loop endlessly.

    An fps filter after a sparse image timeline emits minutes of frames at once; on a long
    render that exhausted the runner's memory.
    """
    from soundtrack import video

    captured = {}

    def fake_run(command, **_):
        captured["command"] = command
        raise SystemExit  # Stop before encoding.

    monkeypatch.setattr(video.subprocess, "run", fake_run)
    for reel in (tmp_path / "reel.mp4", None):
        with pytest.raises(SystemExit):
            video.render_video(tmp_path / "mix.pcm", reel, tmp_path / "b.png", tmp_path / "text.ffconcat",
                               tmp_path / "out.mp4", 3600, False, "anull", None)
        command = captured["command"]
        graph = command[command.index("-filter_complex") + 1]
        assert "-loop" not in command
        chain = graph.split("[2:v]", 1)[1].split(";", 1)[0]  # The text timeline.
        assert "fps=" not in chain


def _fake_search(monkeypatch, items):
    import json as jsonlib
    import subprocess as sp

    from soundtrack import sources

    def fake_run(command, **_):
        lines = [jsonlib.dumps({"id": vid, "title": name, "duration": 150, "channel": channel})
                 for vid, name, channel in items]
        return sp.CompletedProcess(command, 0, stdout="\n".join(lines))

    monkeypatch.setattr(sources.subprocess, "run", fake_run)
    return sources


def test_audio_search_accepts_franchise_titles_and_official_artist_tracks(monkeypatch):
    sources = _fake_search(monkeypatch, [
        ("aaaaaaaaaa1", "Pirates of the Caribbean - He's a Pirate (Soundtrack)", "Some Uploader"),
        ("aaaaaaaaaa2", "Pirates of the Caribbean: The Curse of the Black Pearl - Main Theme", "Some Uploader"),
        ("aaaaaaaaaa3", "The Medallion Calls", "Klaus Badelt - Topic"),
        ("aaaaaaaaaa4", "Time", "Some Other Artist - Topic"),
        ("aaaaaaaaaa5", "Pirates of the Caribbean, arr. Bernotas - Concert Band score", "Band Music"),
        ("aaaaaaaaaa6", "He's a Pirate - Piano Tutorial (Pirates of the Caribbean soundtrack)", "Piano"),
    ])
    found = sources.matching_uploads(
        "Pirates of the Caribbean: The Curse of the Black Pearl (2003)", ["official soundtrack"],
        {"soundtrack", "score", "ost", "theme", "music"},
        {"piano", "tutorial", "arr", "band", "concert"}, (30, 7200),
        artists=["Klaus Badelt", "Hans Zimmer"])
    ids = [vid for vid, _ in found]
    assert ids[0] == "aaaaaaaaaa2"  # The subtitle match ranks first.
    assert set(ids) == {"aaaaaaaaaa1", "aaaaaaaaaa2", "aaaaaaaaaa3"}


def test_unavailable_download_retries_signed_out(monkeypatch, tmp_path):
    import subprocess as sp

    from soundtrack import sources

    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("SOUNDTRACK_COOKIES", str(cookies))
    calls = []

    def fake_run(command, **_):
        calls.append("--cookies" in command)
        if "--cookies" in command:
            return sp.CompletedProcess(command, 1, stdout="",
                                       stderr="ERROR: [youtube] x: Video unavailable. This content isn’t available.")
        target = command[command.index("-o") + 1].replace("%(ext)s", "m4a")
        open(target, "w").close()
        return sp.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(sources.subprocess, "run", fake_run)
    monkeypatch.setattr(sources, "media_length", lambda path: 120.0)
    files, _ = sources.download_uploads([("aaaaaaaaaa1", (5, 120, "Dune Soundtrack"))],
                                        tmp_path / "a", "bestaudio/best", 8, 2400, 12)
    assert calls == [True, False] and len(files) == 1


def test_download_routes(monkeypatch, tmp_path):
    from soundtrack.common import download_routes, youtube

    cookies = tmp_path / "cookies.txt"
    cookies.write_text("x")
    monkeypatch.setenv("SOUNDTRACK_COOKIES", str(cookies))
    monkeypatch.setenv("SOUNDTRACK_PROXY", "socks5://127.0.0.1:40000")
    assert download_routes() == [(True, True), (False, True), (True, False)]
    assert youtube()[-2:] == ["--proxy", "socks5://127.0.0.1:40000"]
    assert "--proxy" not in youtube(proxy=False)
    monkeypatch.delenv("SOUNDTRACK_PROXY")
    assert download_routes() == [(True, False), (False, False)]
    monkeypatch.delenv("SOUNDTRACK_COOKIES")
    assert download_routes() == [(False, False)]


def test_truncated_download_is_retried_and_counted_by_real_length(monkeypatch, tmp_path):
    import subprocess as sp

    from soundtrack import sources

    cookies = tmp_path / "cookies.txt"
    cookies.write_text("x")
    monkeypatch.setenv("SOUNDTRACK_COOKIES", str(cookies))
    lengths = iter([360.0, 2600.0, 300.0])  # First copy cut short, retry complete.

    def fake_run(command, **_):
        target = command[command.index("-o") + 1].replace("%(ext)s", "webm")
        open(target, "w").close()
        return sp.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(sources.subprocess, "run", fake_run)
    monkeypatch.setattr(sources, "media_length", lambda path: next(lengths))
    files, _ = sources.download_uploads([("aaaaaaaaaa1", (5, 2627, "Full Official Soundtrack")),
                                         ("aaaaaaaaaa2", (5, 300, "Main Theme"))],
                                        tmp_path / "a", "bestaudio/best", 8, 4800, 12)
    assert files[0].name == "aaaaaaaaaa1.webm"
    assert not list((tmp_path / "a").glob("*.partial.*"))


def test_other_installments_are_left_out():
    from soundtrack.common import terms
    from soundtrack.sources import installment_terms, other_installment, title_parts

    main, sub = title_parts("Pirates of the Caribbean: The Curse of the Black Pearl (2003)")
    siblings = installment_terms(["Pirates of the Caribbean: Dead Man's Chest",
                                  "Pirates of the Caribbean: At World's End"], main)
    cases = {
        "Pirates of the Caribbean: At World's End - Love Theme (Revised Suite)": True,
        "Pirates of the Caribbean Dead Man's Chest Soundtrack - Jack Sparrow": True,
        "Pirates of the Caribbean: The Curse of the Black Pearl - Full Official Soundtrack": False,
        'Pirates of the Caribbean: Curse of the Black Pearl (2003) Scene: "...Take a walk!"': False,
        "Pirates of the Caribbean - He's a Pirate (Soundtrack)": False,
    }
    for name, expected in cases.items():
        assert other_installment(name, terms(name), main, sub, siblings) is expected, name
    # Without a subtitle there is nothing to tell installments apart by name alone.
    main, sub = title_parts("Interstellar (2014)")
    assert not other_installment("Interstellar: Main Theme", terms("Interstellar: Main Theme"), main, sub, [])


def test_wikidata_series_parsing():
    from soundtrack.footage import related_ids, series_parts

    entity = {"claims": {"P179": [_claim({"id": "Q100"})], "P156": [_claim({"id": "Q2"})]}}
    assert related_ids(entity) == (["Q100"], ["Q2"])
    assert series_parts({"claims": {"P527": [_claim({"id": "Q1"}), _claim({"id": "Q2"})]}}) == ["Q1", "Q2"]


def test_audio_search_skips_covers_and_prefers_full_albums(monkeypatch):
    import json as jsonlib
    import subprocess as sp

    from soundtrack import sources

    items = [
        ("aaaaaaaaaa1", "Pirates of the Caribbean Suite - Film Symphony Orchestra (Soundtrack)", 379),
        ("aaaaaaaaaa2", "Pirates of the Caribbean: The Curse of the Black Pearl - Main Theme", 200),
        ("aaaaaaaaaa3", "Pirates of the Caribbean: The Curse of the Black Pearl - Full Soundtrack", 2627),
    ]

    def fake_run(command, **_):
        lines = [jsonlib.dumps({"id": vid, "title": name, "duration": length, "channel": "x"})
                 for vid, name, length in items]
        return sp.CompletedProcess(command, 0, stdout="\n".join(lines))

    seen = {}

    def fake_download(candidates, *args, **kwargs):
        seen["ids"] = [vid for vid, _ in candidates]
        return [object()], [("x", "y")]

    monkeypatch.setattr(sources.subprocess, "run", fake_run)
    monkeypatch.setattr(sources, "download_uploads", fake_download)
    sources.sources_for("Pirates of the Caribbean: The Curse of the Black Pearl (2003)", None)
    assert seen["ids"] == ["aaaaaaaaaa3", "aaaaaaaaaa2"]


def test_media_length_sees_truncated_files(tmp_path):
    import shutil
    import subprocess as sp

    from soundtrack.common import media_duration, media_length

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required")
    full = tmp_path / "full.webm"
    sp.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
            "sine=frequency=220:duration=120", "-c:a", "libopus", "-b:a", "64k", str(full)], check=True)
    cut = tmp_path / "cut.webm"
    cut.write_bytes(full.read_bytes()[:len(full.read_bytes()) // 6])
    assert media_duration(cut) > 100  # The header still claims the full length.
    assert media_length(cut) < 40 and abs(media_length(full) - 120) < 1
    # Timestamps that jump to the end (as in a real Actions download) must not count as audio.
    gap = tmp_path / "gap.webm"
    sp.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
            "sine=frequency=220:duration=40", "-af", "asetpts='if(gte(T,20),PTS+600/TB,PTS)'",
            "-c:a", "libopus", "-b:a", "64k", str(gap)], check=True)
    assert media_duration(gap) > 600 and abs(media_length(gap) - 40) < 1


def test_audible_span_ignores_a_silent_tail(tmp_path):
    import shutil
    import subprocess as sp

    from soundtrack.common import audible_span, media_length

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required")
    path = tmp_path / "silent-tail.webm"
    sp.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
            "sine=frequency=220:duration=60", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=240",
            "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1", "-c:a", "libopus", "-b:a", "64k",
            str(path)], check=True)
    assert abs(media_length(path) - 300) < 2  # All the data is there...
    assert abs(audible_span(path) - 60) < 3   # ...but only a minute of it is sound.


def test_distinct_cues_drops_second_upload_of_same_track():
    from soundtrack import sources

    title = "Pirates of the Caribbean: The Curse of the Black Pearl (2003)"
    names = [
        "Pirates of the Caribbean (2003) - Full Official Soundtrack",
        "01. Fog Bound (Pirates of the Caribbean: The Curse of the Black Pearl Complete Score)",
        "Pirates Of The Caribbean (Complete Score) - Fog Bound",
        "Pirates of the Caribbean- Original  Soundtrack",
        "Pirates Of The Caribbean (Complete Score) - Sacking The Village (Part 1)",
        "Pirates Of The Caribbean (Complete Score) - Sacking The Village (Part 2)",
    ]
    kept = sources.distinct_cues([(str(i), (0, 60, n)) for i, n in enumerate(names)], title, "Klaus Badelt")
    assert [video_id for video_id, _ in kept] == ["0", "1", "3", "4", "5"]


def test_title_options_follow_the_style_rules():
    from soundtrack.titles import title_options

    title = "Pirates of the Caribbean: The Curse of the Black Pearl (2003)"
    popular = title_options(title, "Jack Sparrow", popular=True, seed=3)
    assert len(popular) == len(set(popular)) == 5
    assert popular[0] == "Pirates of the Caribbean music to work with Jack Sparrow"
    assert sum(" | " in option for option in popular) == 3
    assert all(len(option) <= 100 for option in popular)
    niche = title_options(title, "Jack Sparrow", popular=False, seed=3)
    assert niche.index("Pirates of the Caribbean music to work with Jack Sparrow") == 3
    assert title_options("Dune (2021)", "", seed=1)[0] != title_options("Dune (2021)", "", seed=2)[0]


def test_names_match_faces_through_video_titles():
    from soundtrack.thumbnails import named_people

    analysis = _analysis()
    analysis.videos[0].credit = ("Ava Stone best scenes | Test Film", "")
    analysis.videos[1].credit = ("Max Kade entrance scene", "")
    assert named_people(analysis, ["Ava Stone", "Max Kade", "Nobody Here"]) == {"Ava Stone": 0, "Max Kade": 1}


def test_thumbnails_keep_text_off_the_face(tmp_path):
    from PIL import Image

    from soundtrack.thumbnails import SIZE, crop_bars, make_thumbnails, text_region

    frame = Image.new("RGB", (1280, 720), (90, 110, 130))
    frame.paste((0, 0, 0), (0, 0, 1280, 90))
    frame.paste((0, 0, 0), (0, 630, 1280, 720))
    cropped, offset = crop_bars(frame)
    assert cropped.size == (1280, 540) and offset == (0, 90)
    face = (760, 140, 180, 200)  # Face on the right: text must not land on it.
    (x, y), cost = text_region(Image.new("RGB", SIZE, (90, 110, 130)), face, 420, 90)
    assert cost < 1000
    assert x + 420 < face[0] - face[2] * 0.7 or y > face[1] + face[3] * 1.3 or y + 90 < face[1] - face[3] * 1.2
    paths = make_thumbnails([(cropped, (900, 60, 150, 170), 0)], "Test Film (2024)", "Ava Stone", tmp_path,
                            lead_person=0)
    assert [path.name for path in paths] == ["thumbnail.jpg"] + [f"thumbnail-{n}.jpg" for n in range(2, 11)]
    assert all(Image.open(path).size == SIZE for path in paths)


def test_the_name_goes_only_on_the_lead_characters_frame(tmp_path, monkeypatch):
    from PIL import Image

    from soundtrack import thumbnails

    named_on = []
    monkeypatch.setattr(thumbnails, "style_dedication",
                        lambda canvas, face, name, character: named_on.append((canvas.getpixel((5, 5)), character)) or canvas)
    other = (Image.new("RGB", (1280, 720), (200, 30, 30)), None, -1)
    lead = (Image.new("RGB", (1280, 720), (30, 200, 30)), None, 0)
    thumbnails.make_thumbnails([other, lead], "Test Film (2024)", "Ava Stone", tmp_path, lead_style="headline",
                               lead_person=0)
    assert len(named_on) == 1  # One frame of the lead: one named version, not three copies.
    assert all(character == "Ava Stone" and pixel[1] > pixel[0] for pixel, character in named_on)
    named_on.clear()  # Only the green (lead) frame carries the name, at most three times.
    paths = thumbnails.make_thumbnails([other], "Test Film (2024)", "Ava Stone", tmp_path, lead_person=0)
    assert not named_on and len(paths) == 10  # No frame of the lead: no one is named.


def test_ten_thumbnail_plans_vary_style_and_words():
    from soundtrack.thumbnails import plan_styles

    plan = plan_styles(10, "headline", 0, True)
    worded = [item for item in plan if item[0] != "dedication"]
    assert len(plan) == 10 and len(worded) == len(set(worded)) and plan[0][0] == "headline"
    assert sum(style == "dedication" for style, _ in plan) == 3
    assert not any(style == "dedication" for style, _ in plan_styles(10, "dedication", 0, False))


def test_tmdb_summary_prefers_text_free_stills_and_billing_order():
    from soundtrack.thumbnails import short_character
    from soundtrack.tmdb import clean_character, details_summary, pick_result

    details = {
        "id": 22, "title": "Pirates of the Caribbean: The Curse of the Black Pearl", "vote_count": 21000,
        "credits": {"cast": [{"order": 1, "name": "Geoffrey Rush", "character": "Captain Barbossa"},
                             {"order": 0, "name": "Johnny Depp", "character": "Captain Jack Sparrow",
                              "profile_path": "/depp.jpg"}]},
        "images": {"backdrops": [
            {"file_path": "/titled.jpg", "width": 3840, "iso_639_1": "en", "vote_average": 9},
            {"file_path": "/small.jpg", "width": 780, "iso_639_1": None, "vote_average": 9},
            {"file_path": "/clean.jpg", "width": 1920, "iso_639_1": None, "vote_average": 5}]},
    }
    summary = details_summary(details, "movie")
    assert summary["backdrops"] == ["/clean.jpg", "/titled.jpg"] and summary["textless"] == 1
    assert summary["cast"][0] == {"name": "Johnny Depp", "character": "Captain Jack Sparrow", "profile": "/depp.jpg"}
    assert short_character("Captain Jack Sparrow") == "Jack Sparrow"
    assert short_character("Captain America") == "Captain America"
    assert clean_character("Gollum / Sméagol (voice)") == "Gollum"
    results = [{"id": 1, "release_date": "2017-05-24"}, {"id": 2, "release_date": "2003-07-09"}]
    assert pick_result(results, "2003", "movie")["id"] == 2


def test_tmdb_is_skipped_without_a_token(monkeypatch):
    from soundtrack import tmdb

    monkeypatch.delenv("TMDB_READ_TOKEN", raising=False)
    assert tmdb.lookup("Dune (2021)") is None


def test_text_prefers_the_darker_side_of_the_frame():
    from PIL import Image, ImageDraw

    from soundtrack.thumbnails import SIZE, text_region

    canvas = Image.new("RGB", SIZE, (235, 235, 230))  # Bright sky on top...
    ImageDraw.Draw(canvas).rectangle((0, 360, 1280, 720), fill=(40, 45, 50))  # ...dark ground below.
    (_, y), _ = text_region(canvas, None, 500, 90)
    assert y > 360
