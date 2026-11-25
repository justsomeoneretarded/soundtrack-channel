"""End-to-end smoke test: synthetic audio and footage through the full renderer."""
import json
import shutil
import subprocess

import pytest

from soundtrack.cli import main
from soundtrack.common import media_duration

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")


def _tone(path, frequency, seconds):
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
        f"aevalsrc=0.3*sin(2*PI*{frequency}*t)*(0.55+0.45*sin(t*0.7))|"
        f"0.3*sin(2*PI*{frequency * 1.5}*t)*(0.55+0.45*sin(t*0.5)):s=48000:d={seconds}",
        str(path),
    ], check=True)


@pytest.mark.parametrize("style", ["mix", "album"])
def test_full_render(tmp_path, style):
    audio = []
    for index, frequency in enumerate((220, 330, 440)):
        path = tmp_path / f"Composer - Cue {index + 1} (Test Film Soundtrack).flac"
        _tone(path, frequency, 40)
        audio.append(path)
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=s=640x360:r=24:d=20", "-c:v", "libx264", str(clip)], check=True)
    out = tmp_path / "dist"
    main(["--title", "Test Film (2024)", "--style", style, "--audio", *map(str, audio),
          "--clips", str(clip), "--seconds", "100", "--short-seconds", "15", "--draft", "--out", str(out)])
    for name in ("soundtrack.mp4", "short.mp4", "thumbnail.jpg", "thumbnail-2.jpg", "thumbnail-3.jpg",
                 "youtube-title.txt", "youtube-title-options.txt",
                 "youtube-description.txt", "youtube-tags.txt", "manifest.json"):
        assert (out / name).stat().st_size > 0, name
    assert abs(media_duration(out / "soundtrack.mp4") - 100) <= 0.1
    assert abs(media_duration(out / "short.mp4") - 15) <= 0.1
    channels = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=channels",
        "-of", "csv=p=0", str(out / "soundtrack.mp4")], text=True).strip()
    assert channels == "2"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["style"] == style
    assert len(manifest["title_options"]) == 5
    assert len(manifest["thumbnails"]) == 10
    assert {source["cue"] for source in manifest["audio_sources"]} == {"Cue 1", "Cue 2", "Cue 3"}
    assert manifest["loudness"] is not None
    assert manifest["reel_shots"], "shots should be chosen even from faceless footage"
    description = (out / "youtube-description.txt").read_text()
    assert "0:00" in description
    chapter_count = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_chapters", "-of", "csv=p=0", str(out / "soundtrack.mp4")],
        text=True).strip().count("\n") + 1
    assert chapter_count == len(manifest["chapters"]) >= 3


def test_reel_fills_video_when_source_ends_mid_shot(tmp_path):
    from soundtrack.reel import Shot, build_reel
    from soundtrack.video import render_video

    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=s=320x180:r=24:d=3", "-c:v", "libx264", str(clip)], check=True)
    shots = [Shot(clip, 2, 4.5, 0, -1, 1.0)] * 8
    audio = tmp_path / "mix.pcm"
    audio.write_bytes(bytes(18 * 48000 * 2 * 4))
    output = tmp_path / "soundtrack.mp4"
    reel, unique = build_reel(shots, tmp_path / "reel", 18, seed=0, draft=True)

    assert 0 < unique < 3
    render_video(audio, reel, None, None, output, 18, draft=True)
    assert abs(media_duration(output, stream="v:0") - 18) <= 0.1


def test_video_validation_rejects_sustained_blackout(tmp_path):
    from soundtrack.video import _check_black_picture

    output = tmp_path / "blackout.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=black:s=64x36:r=12:d=40", "-g", "24", "-c:v", "libx264", str(output)],
                   check=True)
    with pytest.raises(RuntimeError, match="Video is black"):
        _check_black_picture(output, 40)
