"""Find and download candidate audio and footage, and derive readable cue names."""
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.parse

from .common import YEAR, terms, youtube

# Words that describe an upload rather than name a cue.
GENERIC = {
    "official", "soundtrack", "score", "original", "ost", "music", "audio", "video", "hd", "hq",
    "4k", "full", "album", "motion", "picture", "from", "the", "lyrics", "lyric", "visualizer",
    "version", "movie", "film", "series", "season", "netflix", "hbo", "disney", "amazon",
    "prime", "tv", "show", "of", "by", "and", "complete", "remastered", "deluxe", "edition", "mp3",
    "flac", "high", "quality", "presents", "records", "recordings", "topic",
}
SEPARATORS = re.compile(r"\s*(?:\||｜|•|·|/|:|\s[-–—]\s|[\[\]()【】{}])\s*")


def matching_uploads(title, queries, required, excluded, duration_limits, interleave=False):
    """Search YouTube and rank matching uploads.

    With interleave, results alternate between queries so each query contributes its best
    matches instead of the strongest query crowding out the rest.
    """
    words = terms(title)
    core = {word for word in words if not YEAR.fullmatch(word)} or words
    requested_year = words - core
    candidates, per_query = {}, []
    for query in queries:
        found = {}
        per_query.append(found)
        try:
            result = subprocess.run(
                youtube() + ["--extractor-retries", "1", "--flat-playlist", "--dump-json",
                           f"ytsearch12:{title} {query}"],
                stdout=subprocess.PIPE, text=True, timeout=85,
            )
        except subprocess.TimeoutExpired:
            continue
        for line in result.stdout.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = item.get("title") or ""
            tokens = terms(name)
            duration = item.get("duration") or 0
            video_id = item.get("id") or ""
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                continue
            if not core <= tokens or not duration_limits[0] <= duration <= duration_limits[1]:
                continue
            if tokens & excluded or not tokens & required:
                continue
            years = {token for token in tokens if YEAR.fullmatch(token)}
            if requested_year and years and not requested_year & years:
                continue
            official = "official" in tokens or (item.get("channel") or "").endswith(" - Topic")
            rank = 5 * bool(requested_year & years) + 4 * official
            rank += 2 * bool(tokens & {"soundtrack", "score", "ost", "clip", "scene"})
            if video_id not in candidates:
                candidates[video_id] = found[video_id] = (rank, duration, name)

    def ordered(group):
        return sorted(group.items(), key=lambda pair: (-pair[1][0], pair[1][1], pair[0]))

    if not interleave:
        return ordered(candidates)
    queues = [ordered(group) for group in per_query]
    merged = []
    while any(queues):
        for queue in queues:
            if queue:
                merged.append(queue.pop(0))
    return merged


BLOCKED = re.compile(r"sign in to confirm|not a bot|http error 429|too many requests|"
                     r"this content isn.t available|po token", re.IGNORECASE)


def download_uploads(candidates, scratch, stream_format, limit, budget, attempts, timeout=240,
                     deadline_minutes=12):
    """Download the best candidates within a time limit.

    Stops early when YouTube keeps answering with sign-in or rate-limit checks, so a blocked
    runner fails in minutes instead of waiting out every timeout.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    files, credits = [], []
    used_names = set()
    seconds, blocked = 0, 0
    stop_at = time.monotonic() + deadline_minutes * 60
    for video_id, (_, duration, name) in candidates[:attempts]:
        signature = frozenset(terms(name))
        if signature in used_names:
            continue
        remaining = stop_at - time.monotonic()
        if remaining < 20:
            print(f"Stopped downloading after {deadline_minutes} minutes.", file=sys.stderr, flush=True)
            break
        url = f"https://www.youtube.com/watch?v={video_id}"
        began = time.monotonic()
        print(f"Downloading {name!r} ({duration:.0f}s)…", flush=True)
        try:
            download = subprocess.run(
                youtube() + ["--no-progress", "-f", stream_format, "--max-filesize", "200M",
                             "-o", str(scratch / f"{video_id}.%(ext)s"), url],
                stderr=subprocess.PIPE, text=True, timeout=min(timeout, remaining),
            )
        except subprocess.TimeoutExpired:
            print(f"  timed out after {time.monotonic() - began:.0f}s.", file=sys.stderr, flush=True)
            continue
        errors = (download.stderr or "").strip()
        matches = [path for path in scratch.glob(f"{video_id}.*")
                   if path.suffix not in {".part", ".ytdl"} and not path.name.endswith(".temp")]
        if download.returncode or len(matches) != 1:
            reason = next((line for line in reversed(errors.splitlines()) if "ERROR" in line),
                          errors.splitlines()[-1] if errors else f"exit code {download.returncode}")
            print(f"  failed: {reason[:300]}", file=sys.stderr, flush=True)
            blocked = blocked + 1 if BLOCKED.search(errors) else 0
            if blocked >= 3:
                print("YouTube is refusing this runner (sign-in or rate-limit check on three uploads "
                      "in a row). Refresh the YOUTUBE_COOKIES secret or use audio_urls.",
                      file=sys.stderr, flush=True)
                break
            continue
        blocked = 0
        print(f"  done in {time.monotonic() - began:.0f}s.", flush=True)
        files.append(matches[0])
        credits.append((name, url))
        used_names.add(signature)
        seconds += duration
        if len(files) >= limit or seconds >= budget:
            break
    return files, credits


def sources_for(title, scratch, composer=""):
    queries = ("official soundtrack", "original score")
    if composer:
        queries += (f"{composer} score",)
    candidates = matching_uploads(
        title, queries,
        {"soundtrack", "score", "ost", "theme", "music", "suite"},
        {"trailer", "reaction", "cover", "karaoke", "remake", "slowed", "extended", "remix", "loop",
         "piano", "tutorial", "sped", "8d", "nightcore", "lofi"},
        (30, 7200),
    )
    files, credits = download_uploads(candidates, scratch, "bestaudio/best", 8, 2400, 12)
    if not files:
        signed_in = bool(os.environ.get("SOUNDTRACK_COOKIES"))
        hint = ("The YOUTUBE_COOKIES secret was used; if the log shows a sign-in or bot check, "
                "export fresh cookies (see README) and update the secret."
                if signed_in else
                "Add a YOUTUBE_COOKIES repository secret so the runner can sign in (see README).")
        raise SystemExit(f"YouTube did not return downloadable matching audio ({len(candidates)} "
                         f"matching uploads found). {hint} You can also pass audio links with the "
                         "audio_urls workflow input or --audio-url.")
    return files, credits


def download_links(urls, scratch):
    """Download audio from direct links, Google Drive, Dropbox, or any site yt-dlp supports."""
    scratch.mkdir(parents=True, exist_ok=True)
    files, credits = [], []
    for index, url in enumerate(urls):
        if not re.match(r"https?://", url):
            print(f"Skipping {url!r}: links must start with http:// or https://.", file=sys.stderr)
            continue
        stem = f"link-{index}"
        try:
            result = subprocess.run(
                youtube() + ["--no-progress", "-f", "bestaudio/best", "--max-filesize", "500M",
                             "--print", "after_move:title", "-o", str(scratch / f"{stem}.%(ext)s"), url],
                stdout=subprocess.PIPE, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            print(f"Timed out downloading {url}.", file=sys.stderr)
            continue
        matches = [path for path in scratch.glob(f"{stem}.*") if path.suffix not in {".part", ".ytdl"}]
        if result.returncode or len(matches) != 1:
            print(f"Could not download {url}.", file=sys.stderr)
            continue
        name = (result.stdout.strip().splitlines() or [""])[-1].strip()
        if not name or name.casefold() in {"download", "file", "audio"}:
            name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).stem or f"Track {index + 1}"
        files.append(matches[0])
        credits.append((name, url))
    if not files:
        raise SystemExit("None of the audio links could be downloaded.")
    return files, credits


def _parts(name):
    return [part.strip(" -–—.,'\"") for part in SEPARATORS.split(name) if part and part.strip(" -–—.,'\"")]


def cue_labels(names, title, composer=""):
    """Turn upload titles into short cue names for chapters and the on-screen card.

    Parts that only restate the work, describe the upload, name the composer, or repeat
    across several uploads (usually an artist or channel) are dropped.
    """
    noise = terms(title) | GENERIC | terms(composer)
    split = [_parts(name) for name in names]
    seen = Counter(frozenset(terms(part)) for parts in split for part in {p.casefold(): p for p in parts}.values())
    labels = []
    for name, parts in zip(names, split):
        keep = []
        for part in parts:
            words = terms(part)
            if not words or words <= noise or all(YEAR.fullmatch(w) or w in noise for w in words):
                continue
            if len(names) > 1 and seen[frozenset(words)] > 1:
                continue
            keep.append(part)
        label = keep[0] if keep else name.strip()
        if len(label) > 60:
            label = label[:59].rstrip() + "…"
        labels.append(label)
    return labels
