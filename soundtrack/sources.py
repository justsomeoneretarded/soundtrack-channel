"""Find and download candidate audio and footage, and derive readable cue names."""
from collections import Counter
import json
import re
import subprocess

from .common import YEAR, YOUTUBE, terms

# Words that describe an upload rather than name a cue.
GENERIC = {
    "official", "soundtrack", "score", "original", "ost", "music", "audio", "video", "hd", "hq",
    "4k", "full", "album", "motion", "picture", "from", "the", "lyrics", "lyric", "visualizer",
    "version", "movie", "film", "series", "season", "netflix", "hbo", "disney", "amazon",
    "prime", "tv", "show", "of", "by", "and", "complete", "remastered", "deluxe", "edition", "mp3",
    "flac", "high", "quality", "presents", "records", "recordings", "topic",
}
SEPARATORS = re.compile(r"\s*(?:\||｜|•|·|/|:|\s[-–—]\s|[\[\]()【】{}])\s*")


def matching_uploads(title, queries, required, excluded, duration_limits):
    words = terms(title)
    core = {word for word in words if not YEAR.fullmatch(word)} or words
    requested_year = words - core
    candidates = {}
    for query in queries:
        try:
            result = subprocess.run(
                YOUTUBE + ["--extractor-retries", "1", "--flat-playlist", "--dump-json",
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
            rank += 2 * bool(tokens & {"soundtrack", "score", "ost", "clip"})
            candidates[video_id] = (rank, duration, name)
    return sorted(candidates.items(), key=lambda pair: (-pair[1][0], pair[1][1], pair[0]))


def download_uploads(candidates, scratch, stream_format, limit, budget, attempts):
    scratch.mkdir(parents=True, exist_ok=True)
    files, credits = [], []
    used_names = set()
    seconds = 0
    for video_id, (_, duration, name) in candidates[:attempts]:
        signature = frozenset(terms(name))
        if signature in used_names:
            continue
        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            download = subprocess.run(
                YOUTUBE + ["--no-progress", "-f", stream_format, "--max-filesize", "200M",
                           "-o", str(scratch / f"{video_id}.%(ext)s"), url],
                timeout=150,
            )
        except subprocess.TimeoutExpired:
            continue
        matches = [path for path in scratch.glob(f"{video_id}.*")
                   if path.suffix not in {".part", ".ytdl"} and not path.name.endswith(".temp")]
        if download.returncode or len(matches) != 1:
            continue
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
        raise SystemExit("YouTube did not return downloadable matching audio from this runner. "
                         "Check the search and downloader errors above. "
                         "For local rendering, supply downloadable audio with --audio.")
    return files, credits


def footage_for(title, scratch):
    candidates = matching_uploads(
        title, ("official movie clip", "official trailer"), {"scene", "clip", "trailer"},
        {"reaction", "fan", "remake", "recap", "breakdown", "concept", "soundtrack", "gameplay",
         "review", "parody", "compilation", "explained", "ai"}, (20, 480),
    )
    return download_uploads(candidates, scratch, "bestvideo[height<=720]/best[height<=720]", 2, 480, 4)


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
