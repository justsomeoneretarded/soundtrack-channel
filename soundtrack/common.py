"""Shared constants and small helpers."""
import os
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

RATE = 48000
WIDTH, HEIGHT = 1280, 720
FFMPEG = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
YOUTUBE = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
           "--socket-timeout", "15", "--retries", "1"]

FONT_ROOT = Path(os.environ.get("SOUNDTRACK_FONT_DIR", "/usr/share/fonts/truetype/dejavu"))
YEAR = re.compile(r"(?:19|20)\d{2}")
WAVE_COLOR = (0x74, 0xEF, 0xB7)


def youtube(signed_in=True):
    """yt-dlp command, signed in with a cookies file when SOUNDTRACK_COOKIES points to one."""
    command = list(YOUTUBE)
    cookies = os.environ.get("SOUNDTRACK_COOKIES", "")
    if signed_in and cookies and Path(cookies).is_file():
        # Gentle pacing keeps a signed-in account from looking automated.
        command += ["--cookies", cookies, "--sleep-requests", "1"]
    return command


def signed_in():
    cookies = os.environ.get("SOUNDTRACK_COOKIES", "")
    return bool(cookies and Path(cookies).is_file())


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


def media_duration(path):
    """Return a container duration in seconds, or None when it cannot be read."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


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
