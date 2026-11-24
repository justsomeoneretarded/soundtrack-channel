"""Official stills and cast from The Movie Database, when a TMDB_READ_TOKEN is available.

TMDB backdrops are the studios' own stills: sharp, at least 1280 pixels wide, and (for those
with no language tag) free of text, so they make far better thumbnails than frames grabbed
from fan uploads. The cast list gives the lead character in billing order, and the lead
actor's headshot lets a thumbnail name the character only over a still that shows them.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

from PIL import Image

from .common import YEAR

API = "https://api.themoviedb.org/3"
IMAGES = "https://image.tmdb.org/t/p/"
POPULAR_VOTES = 3000  # TMDB vote count above which a film or series is widely known.


def token():
    return os.environ.get("TMDB_READ_TOKEN", "").strip()


def _get(path, **params):
    url = f"{API}/{path}?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token()}",
                                                    "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def clean_character(name):
    """'Captain Jack Sparrow / Narrator (voice)' -> 'Captain Jack Sparrow'."""
    name = re.split(r"\s*/\s*|\s*\|\s*", name or "")[0]
    name = re.sub(r"\s*\((?:voice|uncredited|archive footage)[^)]*\)", "", name, flags=re.IGNORECASE)
    return name.strip(" \"'")


def pick_result(results, year, kind):
    """First search result in the requested year, else the first result."""
    key = "release_date" if kind == "movie" else "first_air_date"
    if year:
        for result in results:
            if str(result.get(key, ""))[:4] == year:
                return result
    return results[0] if results else None


def details_summary(details, kind):
    cast = sorted(details.get("credits", {}).get("cast", []), key=lambda person: person.get("order", 99))
    images = details.get("images", {}) if isinstance(details.get("images"), dict) else {}
    backdrops = [b for b in images.get("backdrops", []) if b.get("file_path") and (b.get("width") or 0) >= 1280]
    # Text-free stills first, then the best-rated and largest.
    backdrops.sort(key=lambda b: (b.get("iso_639_1") is None, b.get("vote_average") or 0, b.get("width") or 0),
                   reverse=True)
    return {
        "kind": kind,
        "id": details.get("id"),
        "title": details.get("title") or details.get("name") or "",
        "votes": int(details.get("vote_count") or 0),
        "cast": [{"name": person.get("name", ""), "character": clean_character(person.get("character", "")),
                  "profile": person.get("profile_path")} for person in cast[:6] if person.get("character")],
        "backdrops": [b["file_path"] for b in backdrops[:16]],
        "textless": sum(1 for b in backdrops[:16] if b.get("iso_639_1") is None),
    }


def lookup(title):
    """Summary of the work on TMDB, or None without a token or a match."""
    if not token():
        return None
    year = next(iter(YEAR.findall(title)), None)
    name = re.sub(r"[\s(\[]*\b(?:19|20)\d{2}\b[)\]]*", " ", title).strip() or title
    try:
        for kind in ("movie", "tv"):
            params = {"query": name, "include_adult": "false", "language": "en-US"}
            results = _get(f"search/{kind}", **params).get("results", [])
            match = pick_result(results, year, kind)
            if match and (not year or str(match.get("release_date") or match.get("first_air_date") or "")[:4] == year
                          or kind == "tv"):
                details = _get(f"{kind}/{match['id']}", append_to_response="credits,images",
                               include_image_language="en,null", language="en-US")
                summary = details_summary(details, kind)
                lead = summary["cast"][0]["character"] if summary["cast"] else "none"
                print(f"TMDB: {summary['title']} ({kind}), {summary['votes']} votes, lead character {lead}, "
                      f"{len(summary['backdrops'])} stills ({summary['textless']} without text).", flush=True)
                return summary
    except Exception as error:  # Network, token, or format problems: fall back to footage frames.
        print(f"TMDB lookup skipped: {error}", file=sys.stderr)
    return None


def fetch(path, cache, size="original", longest=1920):
    """Download a TMDB image once (cached) and return it as RGB, scaled to `longest` pixels."""
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (hashlib.sha256(f"{size}{path}".encode()).hexdigest()[:24] + ".jpg")
    if not target.exists():
        try:
            request = urllib.request.Request(IMAGES + size + path, headers={"User-Agent": "soundtrack-video/1.0"})
            with urllib.request.urlopen(request, timeout=40) as response:
                data = response.read()
            Image.open(io.BytesIO(data)).verify()
            target.write_bytes(data)
        except Exception:
            return None
    try:
        image = Image.open(target).convert("RGB")
    except OSError:
        return None
    if max(image.size) > longest:
        image.thumbnail((longest, longest), Image.Resampling.LANCZOS)
    return image
