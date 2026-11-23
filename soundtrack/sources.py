"""Find and download candidate audio and footage, and derive readable cue names."""
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unicodedata
import urllib.parse

from .common import MAX_SOURCE_SECONDS, YEAR, download_routes, media_length, terms, youtube

# Words that describe an upload rather than name a cue.
GENERIC = {
    "official", "soundtrack", "score", "original", "ost", "music", "audio", "video", "hd", "hq",
    "4k", "full", "album", "motion", "picture", "from", "the", "lyrics", "lyric", "visualizer",
    "version", "movie", "film", "series", "season", "netflix", "hbo", "disney", "amazon",
    "prime", "tv", "show", "of", "by", "and", "complete", "remastered", "deluxe", "edition", "mp3",
    "flac", "high", "quality", "presents", "records", "recordings", "topic",
}
SEPARATORS = re.compile(r"\s*(?:\||｜|•|·|/|:|\s[-–—]\s|[\[\]()【】{}])\s*")


LINKING_WORDS = {"the", "of", "a", "an", "and", "in", "on", "at", "to", "for"}


def title_parts(title):
    """('Pirates of the Caribbean', 'The Curse of the Black Pearl') from a full title.

    Uploads usually name the franchise but often drop the subtitle, so only the main part is
    required to match; the subtitle raises the rank.
    """
    bare = re.sub(r"[(\[]?\b(?:19|20)\d{2}\b[)\]]?", " ", title)
    parts = re.split(r"\s*(?::|\s[-–—]\s)\s*", bare.strip(), maxsplit=1)
    main, subtitle = parts[0], parts[1] if len(parts) > 1 else ""
    main_terms = terms(main) or terms(title)
    sub_terms = terms(subtitle) - LINKING_WORDS
    return main_terms, sub_terms


def installment_terms(others, main_terms):
    """Distinctive words of each other work in the series, e.g. {'dead', 'man', 's', 'chest'}."""
    found = []
    for other in others:
        distinct = terms(other) - main_terms - LINKING_WORDS - GENERIC
        distinct = {word for word in distinct if not YEAR.fullmatch(word)}
        if distinct:
            found.append(distinct)
    return found


def other_installment(name, tokens, main_terms, sub_terms, sibling_terms):
    """True when an upload names a different film or season of the same franchise."""
    for distinct in sibling_terms:
        if distinct <= tokens and not distinct <= sub_terms:
            return True
    if not sub_terms:
        return False
    # "Franchise: Other Subtitle - Track" names another installment when the subtitle differs.
    for match in re.finditer(r":\s*([^:|\-–—(\[]+)", name):
        before = re.findall(r"[^\W_]+", unicodedata.normalize("NFKD", name[:match.start()].casefold()))
        if not main_terms <= set(before[-len(main_terms):]):
            continue  # The colon doesn't follow the franchise name (e.g. 'Scene: "..."').
        words = terms(match.group(1)) - LINKING_WORDS - GENERIC
        if words and not words & sub_terms:
            return True
    return False


def matching_uploads(title, queries, required, excluded, duration_limits, interleave=False,
                     artists=(), others=(), prefer_long=False):
    """Search YouTube and rank matching uploads.

    An upload must name the work's main title (the subtitle is a bonus). Uploads from an
    artist's auto-generated "Topic" channel are accepted for the given artists even when the
    track name doesn't mention the work, since those are official releases.
    With interleave, results alternate between queries so each query contributes its best
    matches instead of the strongest query crowding out the rest.
    """
    words = terms(title)
    requested_year = {word for word in words if YEAR.fullmatch(word)}
    main_terms, sub_terms = title_parts(title)
    artist_terms = [terms(artist) for artist in artists if terms(artist)]
    sibling_terms = installment_terms(others, main_terms)
    skipped = 0
    candidates, per_query = {}, []
    for query in queries:
        found = {}
        per_query.append(found)
        try:
            result = subprocess.run(
                youtube() + ["--extractor-retries", "1", "--flat-playlist", "--dump-json",
                             f"ytsearch15:{title} {query}"],
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
            channel = item.get("channel") or item.get("uploader") or ""
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                continue
            if not duration_limits[0] <= duration <= duration_limits[1] or tokens & excluded:
                continue
            topic = channel.endswith(" - Topic")
            channel_terms = terms(channel[:-len(" - Topic")]) if topic else set()
            artist_topic = bool(channel_terms) and any(
                artist <= channel_terms or channel_terms <= artist for artist in artist_terms)
            names_work = main_terms <= tokens
            if not (names_work and tokens & required) and not artist_topic:
                continue
            years = {token for token in tokens if YEAR.fullmatch(token)}
            if requested_year and years and not requested_year & years:
                continue
            if other_installment(name, tokens, main_terms, sub_terms, sibling_terms):
                skipped += 1
                continue
            official = "official" in tokens or topic
            rank = 5 * bool(requested_year & years) + 4 * official
            rank += 2 * bool(tokens & {"soundtrack", "score", "ost", "clip", "scene"})
            if sub_terms:
                rank += 6 if sub_terms <= tokens else 2 * (len(sub_terms & tokens) * 2 >= len(sub_terms))
            if not names_work:
                rank -= 3  # Official artist track without the work's name: plausible but unconfirmed.
            if video_id not in candidates:
                candidates[video_id] = found[video_id] = (rank, duration, name)
    print(f"Found {len(candidates)} matching uploads"
          + (f" (left out {skipped} from other films in the series)." if skipped else "."), flush=True)

    def ordered(group):
        # Among equal ranks, longer uploads first when collecting music (full albums and suites).
        length = (lambda d: -d) if prefer_long else (lambda d: d)
        return sorted(group.items(), key=lambda pair: (-pair[1][0], length(pair[1][1]), pair[0]))

    if not interleave:
        return ordered(candidates)
    queues = [ordered(group) for group in per_query]
    merged = []
    while any(queues):
        for queue in queues:
            if queue:
                merged.append(queue.pop(0))
    return merged


BLOCKED = re.compile(r"sign in to confirm|not a bot|http error 429|too many requests|po token",
                     re.IGNORECASE)
ALTERNATE_CLIENTS = ("web_safari", "mweb")
UNAVAILABLE = re.compile(r"content isn.t available|video unavailable", re.IGNORECASE)


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
        command = ["--no-progress", "-f", stream_format, "--max-filesize", "200M",
                   "-o", str(scratch / f"{video_id}.%(ext)s"), url]
        best = None  # (seconds, path) of the longest copy seen, in case every copy is short.
        try:
            routes = download_routes()
            # After the usual routes, ask YouTube as other apps: a stream served cut short to
            # one client is often complete for another.
            attempts = [(cookies, proxy, None) for cookies, proxy in routes]
            attempts += [(routes[0][0], routes[0][1], client) for client in ALTERNATE_CLIENTS]
            for attempt, (with_cookies, with_proxy, client) in enumerate(attempts):
                if client and best is None:
                    break  # Alternate clients only help with truncated copies.
                if attempt:
                    route = (("signed in" if with_cookies else "signed out")
                             + (" through the proxy" if with_proxy else " directly")
                             + (f" as {client}" if client else ""))
                    print(f"  retrying {route}.", file=sys.stderr, flush=True)
                extra = ["--extractor-args", f"youtube:player_client={client}"] if client else []
                download = subprocess.run(
                    youtube(signed_in=with_cookies, proxy=with_proxy) + extra + command,
                    stderr=subprocess.PIPE, text=True,
                    timeout=min(timeout, max(20, stop_at - time.monotonic())))
                if not download.returncode:
                    got = _downloaded(scratch, video_id)
                    length = (media_length(got) or 0) if got else 0
                    if got and (duration <= 30 or length >= duration * 0.85):
                        if best:
                            best[1].unlink(missing_ok=True)  # Drop the earlier, shorter copy.
                        best = (length, got)
                        break
                    # yt-dlp can report success for a stream that was cut short.
                    print(f"  incomplete download: {length:.0f}s of {duration:.0f}s.", file=sys.stderr, flush=True)
                    if got and (best is None or length > best[0]):
                        if best:
                            best[1].unlink(missing_ok=True)
                        kept = got.with_name(f"{video_id}.partial{got.suffix}")
                        got.replace(kept)
                        best = (length, kept)
                    elif got:
                        got.unlink(missing_ok=True)
                    continue
                if not (UNAVAILABLE.search(download.stderr or "") or BLOCKED.search(download.stderr or "")):
                    break
        except subprocess.TimeoutExpired:
            print(f"  timed out after {time.monotonic() - began:.0f}s.", file=sys.stderr, flush=True)
            continue
        errors = (download.stderr or "").strip()
        if best and best[0] < 45:
            best[1].unlink(missing_ok=True)
            best = None
        if best is None:
            reason = next((line for line in reversed(errors.splitlines()) if "ERROR" in line),
                          errors.splitlines()[-1] if errors else f"exit code {download.returncode}")
            print(f"  failed: {reason[:300]}", file=sys.stderr, flush=True)
            for warning in [line for line in errors.splitlines() if line.startswith("WARNING")][:4]:
                print(f"    {warning[:300]}", file=sys.stderr, flush=True)
            blocked = blocked + 1 if BLOCKED.search(errors) else 0
            if blocked >= 3:
                print("YouTube is refusing this runner (sign-in or rate-limit check on three uploads "
                      "in a row). Refresh the YOUTUBE_COOKIES secret or use audio_urls.",
                      file=sys.stderr, flush=True)
                break
            continue
        blocked = 0
        length, path = best
        print(f"  done in {time.monotonic() - began:.0f}s ({length:.0f}s of media).", flush=True)
        files.append(path)
        credits.append((name, url))
        used_names.add(signature)
        seconds += min(length, MAX_SOURCE_SECONDS)
        if len(files) >= limit or seconds >= budget:
            break
    return files, credits


def _downloaded(scratch, video_id):
    matches = [path for path in scratch.glob(f"{video_id}.*")
               if path.suffix not in {".part", ".ytdl"} and not path.name.endswith(".temp")
               and ".partial." not in path.name]
    return matches[0] if len(matches) == 1 else None


COVERS = {"orchestra", "orchestral", "symphony", "symphonic", "philharmonic", "choir", "choral",
          "tribute", "rendition", "performed", "performance", "reimagined", "reimagining", "acapella",
          "vocal", "vocals", "lyrics", "singing", "sing", "metal", "rock", "trap", "edm", "dubstep"}
# Mood compilations mix several films and loop material, so they are not this work's album.
COMPILATIONS = {"focus", "study", "studying", "sleep", "sleeping", "relax", "relaxing", "ambience",
                "ambient", "asmr", "chill", "calm", "meditation", "hour", "hours", "hrs", "hr",
                "playlist", "compilation", "collection", "best", "greatest", "epic", "mix", "megamix",
                "mashup", "medley", "workout", "gaming"}


def sources_for(title, scratch, composer="", seconds=3600, others=()):
    queries = ("official soundtrack", "full soundtrack", "original score", "complete score")
    if composer:
        queries += (f"{composer} score",)
    candidates = matching_uploads(
        title, queries,
        {"soundtrack", "score", "ost", "theme", "music", "suite"},
        {"trailer", "reaction", "cover", "karaoke", "remake", "slowed", "extended", "remix", "loop",
         "piano", "tutorial", "sped", "8d", "nightcore", "lofi", "arr", "arranged", "arrangement",
         "band", "concert", "live", "guitar", "violin", "cello", "ukulele", "flute", "ocarina", "drum",
         "drums", "marimba", "kalimba", "synthesia", "midi", "8bit", "orgel", "accordion", "harp",
         "recorder", "trumpet", "saxophone", "sax", "lesson", "sheet", "easy", "beginner"} | COVERS | COMPILATIONS,
        (30, 7200), artists=[part for part in composer.split(" & ") if part], others=others,
        prefer_long=True,
    )
    # Gather at least as much distinct music as the video is long (capped), from several uploads.
    budget = min(2 * MAX_SOURCE_SECONDS, max(1800, seconds))
    files, credits = download_uploads(candidates, scratch, "bestaudio/best", 8, budget, 16)
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
