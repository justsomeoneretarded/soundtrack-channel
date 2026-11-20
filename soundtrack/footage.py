"""Find the memorable characters in trailers and clips, and pick their best scenes.

Pipeline:
1. Ask Wikidata for the work's characters so YouTube can be searched for their scenes.
2. Decode each video once: detect shot cuts, and sample two frames a second for exposure,
   sharpness, and faces (OpenCV YuNet detector + SFace recogniser, pinned by hash).
3. Group faces by identity across all videos. Characters who appear most often and largest,
   across the most videos, rank as the main characters.
4. Score each shot by how well it shows a main character, build 5-10 second windows that
   start and end on shot cuts, and choose a varied set across characters and sources.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageOps

from .common import FFMPEG, YEAR
from .sources import download_uploads, matching_uploads

ZOO_COMMIT = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODELS = {
    "detector": ("face_detection_yunet/face_detection_yunet_2023mar.onnx",
                 "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "recognizer": ("face_recognition_sface/face_recognition_sface_2021dec.onnx",
                   "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}
MODEL_DIR = Path(os.environ.get("SOUNDTRACK_MODEL_DIR") or Path.home() / ".cache" / "soundtrack-models")
SAMPLE_FPS = 2
FRAME_W, FRAME_H = 640, 360
SAME_PERSON = 0.40  # SFace cosine similarity; the model authors suggest 0.363.
PERSON_WEIGHTS = (1.0, 0.88, 0.78, 0.7)
ANALYSE_SECONDS = 900
USER_AGENT = "soundtrack-video/1.0 (open-source renderer)"

SCENE_WORDS = {"scene", "scenes", "clip", "clips", "trailer", "teaser", "moment", "moments"}
NOT_SCENES = {
    "reaction", "reacts", "fan", "remake", "recap", "breakdown", "concept", "soundtrack", "gameplay",
    "review", "parody", "explained", "explain", "ai", "interview", "featurette", "behind", "bts",
    "making", "premiere", "podcast", "press", "junket", "cast", "edit", "amv", "shorts", "lego",
    "animatic", "cosplay", "theory", "theories", "ranking", "ranked", "analysis", "deleted",
}


# --------------------------------------------------------------------------- characters

def _wikidata(params):
    url = "https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode({**params, "format": "json"})
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def pick_work(ids, entities, year=None):
    """First search hit described as a film or series, matching the year when one is given."""
    kinds = re.compile(r"\b(film|movie|series|television|tv|anime|miniseries|show|sitcom|drama)\b")
    for item in ids:
        entity = entities.get(item, {})
        description = entity.get("descriptions", {}).get("en", {}).get("value", "").casefold()
        if not kinds.search(description):
            continue
        if year:
            dates = [claim.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("time", "")
                     for claim in entity.get("claims", {}).get("P577", [])]
            if year not in description and not any(date[1:5] == year for date in dates):
                continue
        return item
    return None


def character_ids(entity):
    """Character items from cast-member role qualifiers, then the work's character list."""
    found = []
    claims = entity.get("claims", {})
    for claim in claims.get("P161", []):
        for qualifier in claim.get("qualifiers", {}).get("P453", []):
            value = qualifier.get("datavalue", {}).get("value", {})
            if isinstance(value, dict) and value.get("id"):
                found.append(value["id"])
    for claim in claims.get("P674", []):
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        if isinstance(value, dict) and value.get("id"):
            found.append(value["id"])
    return list(dict.fromkeys(found))


def characters_for(title, limit=4):
    """Main character names for a work, best effort. Returns [] when Wikidata has none."""
    year = next(iter(YEAR.findall(title)), None)
    name = re.sub(r"[\s(\[]*\b(?:19|20)\d{2}\b[)\]]*", " ", title).strip() or title
    try:
        found = _wikidata({"action": "wbsearchentities", "search": name, "language": "en",
                           "type": "item", "limit": 8}).get("search", [])
        ids = [item["id"] for item in found]
        if not ids:
            return []
        entities = _wikidata({"action": "wbgetentities", "ids": "|".join(ids),
                              "props": "claims|descriptions", "languages": "en"}).get("entities", {})
        work = pick_work(ids, entities, year)
        roles = character_ids(entities.get(work, {})) if work else []
        if not roles:
            return []
        labels = _wikidata({"action": "wbgetentities", "ids": "|".join(roles[:30]), "props": "labels",
                            "languages": "en"}).get("entities", {})
    except Exception as error:  # Network or format problems only cost the character searches.
        print(f"Character lookup skipped: {error}", file=sys.stderr)
        return []
    names = []
    for role in roles:
        label = labels.get(role, {}).get("labels", {}).get("en", {}).get("value", "")
        label = re.sub(r"\s*\(.*?\)\s*", " ", label).strip()
        if label and label.casefold() != name.casefold() and label not in names:
            names.append(label)
    print(f"Characters from Wikidata: {', '.join(names[:limit]) or 'none'}.", flush=True)
    return names[:limit]


def gather(title, scratch, characters=()):
    """Download trailers, official clips, and scenes featuring the named characters."""
    queries = ["official trailer", "official clip"]
    queries += [f"{character} scene" for character in characters[:4]]
    queries += ["best scenes"]
    candidates = matching_uploads(title, queries, SCENE_WORDS, NOT_SCENES, (20, 900), interleave=True)
    return download_uploads(
        candidates, scratch,
        "bestvideo[height<=720][vcodec^=avc1]/bestvideo[height<=720]/best[height<=720]",
        6, 2400, 12, timeout=300,
    )


# --------------------------------------------------------------------------- analysis

@dataclass
class Face:
    video: int
    time: float
    box: tuple  # x, y, w, h in the 640x360 analysis frame
    area: float
    quality: float
    frontal: float
    embedding: object = None
    person: int = -1


@dataclass
class Video:
    path: Path
    duration: float
    credit: tuple
    size: tuple
    cuts: list
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    quality: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class Analysis:
    videos: list
    faces: list
    people: list  # [(person id, importance, face count)] most important first
    recognised: bool


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_models():
    """YuNet detector and SFace recogniser, downloaded once and verified. (None, None) on failure."""
    if os.environ.get("SOUNDTRACK_FACE_MODELS") == "0":
        return None, None
    import cv2

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        paths = {}
        for key, (relative, digest) in MODELS.items():
            path = MODEL_DIR / Path(relative).name
            if not path.exists() or _sha256(path) != digest:
                url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{ZOO_COMMIT}/models/{relative}"
                partial = path.with_suffix(".part")
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
                    shutil.copyfileobj(response, out)
                if _sha256(partial) != digest:
                    partial.unlink()
                    raise ValueError(f"checksum mismatch for {path.name}")
                partial.replace(path)
            paths[key] = path
        detector = cv2.FaceDetectorYN.create(str(paths["detector"]), "", (FRAME_W, FRAME_H), 0.7, 0.3, 5000)
        recognizer = cv2.FaceRecognizerSF.create(str(paths["recognizer"]), "")
        return detector, recognizer
    except Exception as error:
        print(f"Face recognition models unavailable ({error}); using basic face detection.", file=sys.stderr)
        return None, None


def _probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode:
        return None
    details = json.loads(result.stdout or "{}")
    streams = details.get("streams") or []
    try:
        duration = float(details["format"]["duration"])
        width, height = int(streams[0]["width"]), int(streams[0]["height"])
    except (KeyError, ValueError, IndexError, TypeError):
        return None
    if not math.isfinite(duration) or duration < 4 or width < 32 or height < 32:
        return None
    return duration, (width, height)


def _frontal(row):
    """1 for a face looking at the camera, falling toward 0 in profile (YuNet landmarks)."""
    right_eye, left_eye, nose = row[4:6], row[6:8], row[8:10]
    span = abs(left_eye[0] - right_eye[0])
    if span < 1:
        return 0.0
    offset = abs(nose[0] - (left_eye[0] + right_eye[0]) / 2) / span
    return float(max(0.0, 1 - offset * 2.2))


def _faces_in(frame, gray, detector, recognizer, cascade):
    found = []
    if detector is not None:
        detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, rows = detector.detect(frame)
        for row in rows if rows is not None else []:
            x, y, w, h = (float(v) for v in row[:4])
            if w < 28 or h < 28:
                continue
            embedding = recognizer.feature(recognizer.alignCrop(frame, row)).flatten().astype(np.float32)
            embedding /= max(float(np.linalg.norm(embedding)), 1e-6)
            found.append(((x, y, w, h), float(row[-1]), _frontal(row), embedding))
    else:
        for x, y, w, h in cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(36, 36)):
            found.append(((float(x), float(y), float(w), float(h)), 0.9, 0.8, None))
    return found


def _analyse_video(index, path, credit, detector, recognizer, cascade, log):
    import cv2

    probe = _probe(path)
    if probe is None:
        return None, []
    duration, size = probe
    duration = min(duration, ANALYSE_SECONDS)
    scale = (f"fps={SAMPLE_FPS},scale={FRAME_W}:{FRAME_H}:force_original_aspect_ratio=decrease,"
             f"pad={FRAME_W}:{FRAME_H}:(ow-iw)/2:(oh-ih)/2,format=bgr24")
    graph = f"[0:v]split[cuts][frames];[cuts]scale=320:-2,select='gt(scene,0.30)',showinfo,nullsink;[frames]{scale}[out]"
    with log.open("w") as errors:
        process = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info", "-t", str(duration), "-i", str(path),
             "-filter_complex", graph, "-map", "[out]", "-f", "rawvideo", "pipe:"],
            stdout=subprocess.PIPE, stderr=errors,
        )
        frame_bytes = FRAME_W * FRAME_H * 3
        times, quality, faces = [], [], []
        sample = 0
        while True:
            buffer = process.stdout.read(frame_bytes)
            if len(buffer) < frame_bytes:
                break
            frame = np.frombuffer(buffer, np.uint8).reshape(FRAME_H, FRAME_W, 3)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            moment = sample / SAMPLE_FPS
            sample += 1
            exposure = float(gray.mean())
            sharpness = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 250)
            usable = 1.0 if 32 < exposure < 215 else 0.25
            times.append(moment)
            quality.append(usable * (0.35 + 0.65 * sharpness))
            for (x, y, w, h), confidence, frontal, embedding in _faces_in(frame, gray, detector, recognizer, cascade):
                x0, y0 = max(0, int(x)), max(0, int(y))
                crop = gray[y0:int(y + h), x0:int(x + w)]
                crisp = min(1.0, cv2.Laplacian(crop, cv2.CV_64F).var() / 180) if crop.size else 0.0
                lit = 1.0 if 40 < float(crop.mean() if crop.size else 0) < 225 else 0.4
                faces.append(Face(index, moment, (x, y, w, h), w * h / (FRAME_W * FRAME_H),
                                  confidence * (0.3 + 0.7 * crisp) * lit, frontal, embedding))
        process.stdout.close()
        process.wait()
    cuts = sorted({round(float(t), 3) for t in re.findall(r"pts_time:([0-9.]+)", log.read_text(errors="replace"))
                   if 0.3 < float(t) < duration - 0.3})
    if not times:
        return None, []
    video = Video(path, duration, credit, size, [0.0] + cuts + [duration],
                  np.array(times), np.array(quality))
    return video, faces


def cluster(faces):
    """Assign each face a person id by greedy cosine clustering; return people by importance."""
    centroids, members = [], []
    for face in sorted((f for f in faces if f.embedding is not None), key=lambda f: -f.quality * f.area):
        if centroids:
            similarity = np.stack(centroids) @ face.embedding
            best = int(np.argmax(similarity))
            if similarity[best] >= SAME_PERSON:
                members[best].append(face)
                total = centroids[best] * (len(members[best]) - 1) + face.embedding
                centroids[best] = total / max(float(np.linalg.norm(total)), 1e-6)
                face.person = best
                continue
        centroids.append(face.embedding.copy())
        members.append([face])
        face.person = len(centroids) - 1
    ranked = []
    for person, group in enumerate(members):
        videos = len({face.video for face in group})
        importance = sum(math.sqrt(face.area) * (0.4 + 0.6 * face.quality) for face in group)
        ranked.append((person, importance * (1 + 0.5 * (videos - 1)), len(group)))
    ranked.sort(key=lambda item: -item[1])
    # Renumber so person 0 is the most important.
    order = {old: new for new, (old, _, _) in enumerate(ranked)}
    for face in faces:
        if face.person >= 0:
            face.person = order[face.person]
    return [(order[old], importance, count) for old, importance, count in ranked]


def analyse(files, credits, scratch):
    if not files:
        return Analysis([], [], [], False)
    import cv2

    detector, recognizer = load_models()
    cascade = None
    if detector is None:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    videos, faces = [], []
    scratch.mkdir(parents=True, exist_ok=True)
    for path, credit in zip(files, credits):
        video, found = _analyse_video(len(videos), Path(path), credit, detector, recognizer, cascade,
                                      scratch / f"analysis-{len(videos)}.log")
        if video is None:
            print(f"Skipping unreadable footage {credit[0]!r}.", file=sys.stderr)
            continue
        videos.append(video)
        faces += found
    people = cluster(faces) if detector is not None else []
    main = [person for person, _, count in people if count >= 3][:len(PERSON_WEIGHTS)]
    print(f"Analysed {len(videos)} videos: {len(faces)} faces, {len(people)} people, "
          f"{len(main)} main characters.", flush=True)
    return Analysis(videos, faces, [p for p in people if p[0] in main], detector is not None)


# --------------------------------------------------------------------------- scenes

@dataclass
class Window:
    video: int
    start: float
    end: float
    score: float
    person: int


def _weight(person, recognised):
    if not recognised:
        return 0.75  # Without identities every face counts the same.
    return PERSON_WEIGHTS[person] if 0 <= person < len(PERSON_WEIGHTS) else 0.25


def candidate_windows(analysis, shortest=5.0, longest=10.0):
    main = {person for person, _, _ in analysis.people}
    by_video = {}
    for face in analysis.faces:
        by_video.setdefault(face.video, []).append(face)
    windows = []
    for index, video in enumerate(analysis.videos):
        # Trailers open on logos and close on title cards and credits.
        lo, hi = max(1.0, video.duration * 0.03), min(video.duration - 2.0, video.duration * 0.92)
        shots = [(a, b) for a, b in zip(video.cuts, video.cuts[1:]) if b > lo and a < hi]
        if not shots:
            continue
        faces = by_video.get(index, [])
        scores, owners = [], []
        for a, b in shots:
            inside = video.quality[(video.times >= a) & (video.times < b)]
            base = float(inside.mean()) if inside.size else 0.0
            best, owner = 0.0, -1
            for face in faces:
                if a <= face.time < b:
                    person = face.person if face.person in main else -1
                    value = (_weight(person, analysis.recognised) * min(1.0, math.sqrt(face.area) * 3.2)
                             * (0.45 + 0.55 * face.quality) * (0.7 + 0.3 * face.frontal))
                    if value > best:
                        best, owner = value, person
            scores.append(0.15 * base + best)
            owners.append(owner)
        for first in range(len(shots)):
            start = max(shots[first][0], lo)
            if shots[first][1] - start < 0.8:
                continue  # Don't open a scene on a flash cut.
            total, weight, credit = 0.0, 0.0, {}
            end = start
            for shot in range(first, len(shots)):
                a, b = max(shots[shot][0], start), min(shots[shot][1], hi, start + longest)
                if b <= a:
                    break
                total += scores[shot] * (b - a)
                weight += b - a
                credit[owners[shot]] = credit.get(owners[shot], 0) + scores[shot] * (b - a)
                end = b
                if end - start >= shortest:
                    break  # End on a cut once the scene is long enough.
            length = end - start
            if length < min(shortest, 3.0):
                continue
            cut_count = sum(1 for cut in video.cuts if start < cut < end)
            pace = 1.0 if cut_count / length < 0.9 else 0.85  # Rapid montages read as noise.
            person = max(credit, key=credit.get) if credit else -1
            windows.append(Window(index, round(start, 3), round(end, 3),
                                  float(total / weight * pace), int(person)))
    return windows


def choose(windows, count):
    """Greedy pick of strong, non-overlapping windows, varied across characters and sources."""
    chosen, people, sources = [], {}, {}
    pool = list(windows)
    while pool and len(chosen) < count:
        def value(window):
            return (window.score * 0.72 ** people.get(window.person, 0)
                    * 0.85 ** sources.get(window.video, 0))

        best = max(pool, key=value)
        chosen.append(best)
        people[best.person] = people.get(best.person, 0) + 1
        sources[best.video] = sources.get(best.video, 0) + 1
        pool = [w for w in pool if w.video != best.video or w.end + 2 <= best.start or w.start >= best.end + 2]
    return chosen


def default_count(seconds):
    if seconds < 12:
        return 0
    if seconds < 60:
        return 2
    return max(4, min(16, round(seconds / 400)))


def plan_scenes(analysis, seconds, count=None):
    """Place the best scenes through the video; the strongest lands near the climax.

    Returns (scenes, details): scenes as (path, source_start, length, output_start).
    """
    count = default_count(seconds) if count is None else max(0, min(count, 24))
    if count == 0 or not analysis.videos:
        return [], []
    longest = min(10.0, max(3.0, seconds * 0.1))
    shortest = min(5.0, longest)
    picks = choose(candidate_windows(analysis, shortest, longest), count)
    if not picks:
        return [], []
    # Leave out filler that is far weaker than the best scene, keeping at least two.
    strongest = max(window.score for window in picks)
    picks = [w for w in picks if w.score >= strongest * 0.3] or picks
    if len(picks) < 2:
        picks = sorted(choose(candidate_windows(analysis, shortest, longest), 2), key=lambda w: -w.score)
    moments = [(index + 0.6) / (len(picks) + 0.2) * 0.92 for index in range(len(picks))]
    slots = sorted(range(len(moments)), key=lambda i: abs(moments[i] - 0.8))
    ordered = sorted(picks, key=lambda w: -w.score)
    placed = sorted(zip((moments[i] for i in slots), ordered), key=lambda pair: pair[0])
    scenes, details = [], []
    for fraction, window in placed:
        video = analysis.videos[window.video]
        when = round(seconds * fraction, 3)
        length = round(window.end - window.start, 3)
        scenes.append((video.path, window.start, length, when))
        details.append({"start": when, "length": length, "source": video.credit[0],
                        "source_start": window.start, "character": window.person,
                        "score": round(window.score, 3)})
    return scenes, details


# --------------------------------------------------------------------------- portraits

def _still(path, moment):
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / "still.png"
        subprocess.run(FFMPEG + ["-ss", f"{moment:.3f}", "-i", str(path), "-map", "0:v:0",
                                 "-frames:v", "1", str(target)])
        if not target.exists():
            return None
        with Image.open(target) as image:
            return image.convert("RGB")


def _portrait_crop(image, box):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    crop_w = min(image.width, image.height * 0.91, max(image.width * 0.26, w * 3.5))
    crop_h = crop_w / 0.91
    left = max(0, min(image.width - crop_w, cx - crop_w / 2))
    top = max(0, min(image.height - crop_h, cy - crop_h * 0.37))
    crop = image.crop((int(left), int(top), int(left + crop_w), int(top + crop_h)))
    return ImageOps.fit(crop, (420, 462), Image.Resampling.LANCZOS)


def _framing(box):
    """Prefer faces that sit in the upper part of the frame at a size a portrait crop can hold."""
    _, y, _, h = box
    centre, height = (y + h / 2) / FRAME_H, h / FRAME_H
    placement = 1.0 if centre < 0.55 else max(0.2, 1 - (centre - 0.55) * 3)
    size = 1.0 if 0.12 <= height <= 0.42 else 0.55
    return placement * size


def portraits(analysis, limit=4):
    """{person: portrait image} for the main characters (or the single best face without models)."""
    if not analysis.faces:
        return {}
    import cv2

    people = [person for person, _, _ in analysis.people][:limit] if analysis.recognised else []
    people = people or [-1]
    detector, _ = load_models() if analysis.recognised else (None, None)
    result = {}
    for person in people:
        candidates = [f for f in analysis.faces if (person == -1 or f.person == person) and f.area > 0.012]
        candidates.sort(key=lambda f: -(f.quality * (0.4 + 0.6 * f.frontal) * min(1.0, math.sqrt(f.area) * 4)
                                        * _framing(f.box)))
        for face in candidates[:4]:
            video = analysis.videos[face.video]
            image = _still(video.path, face.time)
            if image is None:
                continue
            width, height = image.size
            scale = min(FRAME_W / width, FRAME_H / height)
            ox, oy = (FRAME_W - width * scale) / 2, (FRAME_H - height * scale) / 2
            x, y, w, h = face.box
            box = ((x - ox) / scale, (y - oy) / scale, w / scale, h / scale)
            if detector is not None:
                # Re-detect at full resolution so the crop centres on the face in this exact frame.
                bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                detector.setInputSize((width, height))
                _, rows = detector.detect(bgr)
                detector.setInputSize((FRAME_W, FRAME_H))
                if rows is None or not len(rows):
                    continue
                centre = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])
                row = min(rows, key=lambda r: np.linalg.norm(r[:2] + r[2:4] / 2 - centre))
                if np.linalg.norm(row[:2] + row[2:4] / 2 - centre) > max(box[2], 40):
                    continue
                box = tuple(float(v) for v in row[:4])
            result[person] = _portrait_crop(image, box)
            break
    return result
