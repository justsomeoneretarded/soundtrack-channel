"""Shared constants and small helpers."""
import os
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

RATE = 48000
MAX_SOURCE_SECONDS = 2700  # Longest stretch decoded from one upload (45 minutes).
WIDTH, HEIGHT = 1280, 720
FFMPEG = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
YOUTUBE = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
           "--socket-timeout", "15", "--retries", "1"]

FONT_ROOT = Path(os.environ.get("SOUNDTRACK_FONT_DIR", "/usr/share/fonts/truetype/dejavu"))
YEAR = re.compile(r"(?:19|20)\d{2}")
WAVE_COLOR = (0x74, 0xEF, 0xB7)


def youtube(signed_in=True, proxy=True):
    """yt-dlp command.

    Signs in with a cookies file when SOUNDTRACK_COOKIES points to one, and sends traffic
    through SOUNDTRACK_PROXY when set. YouTube refuses media to data-center addresses such
    as GitHub's runners; the workflow routes through Cloudflare WARP (or a YOUTUBE_PROXY
    secret) so YouTube sees a different exit address.
    """
    command = list(YOUTUBE)
    cookies = os.environ.get("SOUNDTRACK_COOKIES", "")
    if signed_in and cookies and Path(cookies).is_file():
        # Gentle pacing keeps a signed-in account from looking automated.
        command += ["--cookies", cookies, "--sleep-requests", "1"]
    if proxy and proxy_url():
        command += ["--proxy", proxy_url()]
    return command


def proxy_url():
    return os.environ.get("SOUNDTRACK_PROXY", "").strip()


def signed_in():
    cookies = os.environ.get("SOUNDTRACK_COOKIES", "")
    return bool(cookies and Path(cookies).is_file())


def download_routes():
    """(signed_in, proxy) combinations to try, most likely to succeed first."""
    routes = [(True, True), (False, True)] if proxy_url() else [(True, False), (False, False)]
    if proxy_url():
        routes.append((True, False))
    if not signed_in():
        routes = [route for route in routes if not route[0]] or [(False, bool(proxy_url()))]
    return list(dict.fromkeys(routes))


def terms(value):
    value = unicodedata.normalize("NFKD", value.casefold())
    return set(re.findall(r"[^\W_]+", value))


def clean_composer(text):
    """'Composed primarily by Klaus Badelt, with Hans Zimmer' -> 'Klaus Badelt & Hans Zimmer'."""
    text = " ".join(text.split())
    text = re.sub(r"(?i)\b(?:original\s+)?(?:music|score|soundtrack)?\s*(?:composed|written|scored|"
                  r"music|score)\s+(?:primarily\s+|mainly\s+|mostly\s+|largely\s+)?by\s+", "", text)
    text = re.sub(r"(?i)^(?:by|music|composer|composers)\s*:?\s+", "", text)
    text = re.sub(r"(?i)\s*(?:,\s*(?:with|and)\s+|\s+with\s+|\s+and\s+|,\s+|\s*/\s*|\s*\+\s*)",
                  " & ", text)
    text = re.sub(r"(?i)\s*\((?:with|additional)[^)]*\)", "", text)
    return text.strip(" ,;&.")


def font(name, size):
    from PIL import ImageFont

    return ImageFont.truetype(str(FONT_ROOT / name), size)


def media_duration(path, stream=None):
    """Return the duration of a container or selected stream, in seconds."""
    selection = ["-select_streams", stream] if stream else []
    entry = "stream=duration" if stream else "format=duration"
    result = subprocess.run(
        ["ffprobe", "-v", "error", *selection, "-show_entries", entry,
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def media_length(path):
    """Seconds of media actually present: the sum of the first stream's packet durations.

    A download that was cut short can keep the full length in its header, and its timestamps
    can jump straight to the end, so neither the header nor the last timestamp is reliable.
    """
    for stream in ("a:0", "v:0"):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries",
             "packet=duration_time", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True,
        )
        total = 0.0
        for line in result.stdout.splitlines():
            try:
                total += float(line.strip().rstrip(","))
            except ValueError:
                continue
        if total > 0:
            return total
    return None


def audible_span(path, limit=MAX_SOURCE_SECONDS):
    """Seconds from the first to the last audible moment of the first audio stream.

    YouTube can serve a file whose packets run the full length but fall silent partway, so
    this listens (8 kHz mono, about three seconds per 45 minutes) instead of counting data.
    """
    import numpy as np

    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-t", str(limit), "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
        capture_output=True,
    )
    samples = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768
    seconds = len(samples) // 8000
    if seconds < 1:
        return 0.0
    rms = np.sqrt(np.mean(np.square(samples[:seconds * 8000].reshape(seconds, 8000)), axis=1))
    active = np.flatnonzero(rms > max(0.0005, float(np.percentile(rms, 80)) * 0.06))
    return float(active[-1] - active[0] + 1) if active.size else 0.0


def timestamp(seconds):
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def length_label(seconds):
    if seconds % 3600 == 0:
        hours = seconds // 3600
        return f"{hours} Hour" if hours == 1 else f"{hours} Hours"
    if seconds >= 600 and seconds % 60 == 0:
        return f"{seconds // 60} Minute"
    return f"{seconds}-Second Preview"
