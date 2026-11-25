"""A continuous film reel: the best shots from the downloaded footage, graded and dissolved.

The reel is the picture for the whole video. Shots are chosen for sharpness, exposure and
how clearly they show the main characters, joined with cross-dissolves in short chunks,
graded to one look, framed with cinema bars, and repeated in shuffled order to fill the
mix.
"""
from dataclasses import dataclass
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from PIL import Image

from .common import FFMPEG, HEIGHT, WIDTH, media_duration
from .footage import candidate_windows, choose

DISSOLVE = 0.8         # Seconds of cross-dissolve between shots.
EDGE_FADE = 0.4        # Fade from and to black at chunk edges.
SHOTS_PER_CHUNK = 8
BAR = 0.128            # Cinema bars, as a fraction of the frame height (about 2.39:1).
GRADE = "eq=contrast=1.06:saturation=0.86:brightness=-0.015"


@dataclass
class Shot:
    path: Path
    start: float
    length: float
    video: int
    person: int
    score: float


def plan_reel(analysis, seed, limit=360, shortest=4.5, longest=8.0):
    """Pick strong, non-overlapping shots and order them so neighbours come from different videos."""
    if not analysis.videos:
        return []
    windows = candidate_windows(analysis, shortest, longest, base_weight=1.0)
    for window in windows:
        window.score *= resolution_weight(analysis.videos[window.video].size)
    picks = choose(windows, limit)
    if not picks:
        return []
    strongest = max(window.score for window in picks)
    picks = [w for w in picks if w.score >= strongest * 0.3 and w.end - w.start >= 2.5]
    rng = random.Random(seed)
    rng.shuffle(picks)
    ordered = []
    while picks:
        last = ordered[-1].video if ordered else None
        index = next((i for i, w in enumerate(picks) if w.video != last), 0)
        ordered.append(picks.pop(index))
    return [Shot(analysis.videos[w.video].path, w.start, round(w.end - w.start, 3), w.video, w.person,
                 round(w.score, 3)) for w in ordered]


def resolution_weight(size):
    """Low-resolution uploads look soft and blocky next to HD footage."""
    height = min(size) if size and min(size) > 0 else 0
    if height >= 700:
        return 1.0
    if height >= 470:
        return 0.75
    return 0.45


def best_shot(shots):
    """The strongest shot that shows a main character."""
    if not shots:
        return None
    with_people = [shot for shot in shots if shot.person >= 0]
    return max(with_people or shots, key=lambda shot: shot.score)


def frame_size(draft):
    return (640, 360, 12) if draft else (WIDTH, HEIGHT, 24)


def bar_height(height):
    return round(height * BAR / 2) * 2


def frame_mask(path, draft):
    """Vignette and cinema bars in one transparent image.

    Overlaying a still is several times faster than FFmpeg's vignette and drawbox filters,
    which recompute every frame.
    """
    width, height, _ = frame_size(draft)
    y, x = np.mgrid[0:height, 0:width]
    radius = np.sqrt(((x - width / 2) / (width / 2)) ** 2 + ((y - height / 2) / (height / 2)) ** 2)
    alpha = np.clip((radius - 0.55) / 0.9, 0, 1) ** 1.6 * 150
    bar = bar_height(height)
    alpha[:bar] = 255
    alpha[height - bar:] = 255
    pixels = np.zeros((height, width, 4), np.uint8)
    pixels[..., 3] = alpha.astype(np.uint8)
    Image.fromarray(pixels, "RGBA").save(path)
    return path


def _chunk(shots, output, draft, mask):
    width, height, fps = frame_size(draft)
    inputs, graph = [], []
    for index, shot in enumerate(shots):
        inputs += ["-ss", f"{shot.start:.3f}", "-t", f"{shot.length:.3f}", "-i", str(shot.path)]
        graph.append(
            f"[{index}:v]setpts=PTS-STARTPTS,fps={fps},"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},"
            f"setsar=1,format=yuv420p,settb=1/{fps}[s{index}]")
    layer, elapsed = "s0", shots[0].length
    for index in range(1, len(shots)):
        graph.append(f"[{layer}][s{index}]xfade=transition=fade:duration={DISSOLVE}:"
                     f"offset={elapsed - DISSOLVE:.3f}[x{index}]")
        layer = f"x{index}"
        elapsed += shots[index].length - DISSOLVE
    inputs += ["-i", str(mask)]
    graph += [
        f"[{len(shots)}:v]format=yuva420p[mask]",
        f"[{layer}]{GRADE}[graded]",
        f"[graded][mask]overlay=0:0:format=yuv420,"
        f"fade=t=in:st=0:d={EDGE_FADE},fade=t=out:st={elapsed - EDGE_FADE:.3f}:d={EDGE_FADE}[v]"]
    result = subprocess.run(FFMPEG + [
        *inputs, "-filter_complex", ";".join(graph), "-map", "[v]", "-an",
        "-t", f"{elapsed:.3f}", "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "17", "-g", str(fps * 2), "-pix_fmt", "yuv420p", str(output),
    ])
    if result.returncode != 0 or not output.exists():
        return None
    # A source can end mid-shot and FFmpeg can still successfully write a shorter chunk.
    return media_duration(output, stream="v:0") or None


def build_reel(shots, scratch, seconds, seed, draft=False):
    """Render the shots in chunks and assemble a reel that fills `seconds`.

    Returns (reel_path, unique_seconds), or (None, 0) when no chunk could be rendered.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    chunks = []
    mask = frame_mask(scratch / "frame.png", draft)
    for start in range(0, len(shots), SHOTS_PER_CHUNK):
        group = shots[start:start + SHOTS_PER_CHUNK]
        if len(group) < 2 and chunks:
            continue  # A lone leftover shot would play without dissolves.
        output = scratch / f"chunk-{len(chunks):03d}.mp4"
        length = _chunk(group, output, draft, mask)
        if length is None:
            print(f"Skipping a reel chunk that failed to render ({output.name}).", file=sys.stderr)
            continue
        chunks.append((output, length))
    if not chunks:
        return None, 0.0
    unique = sum(length for _, length in chunks)
    # Repeat in a fresh shuffled order each pass, never playing the same chunk twice in a row.
    rng, order, total = random.Random(seed), [], 0.0
    while total < seconds + 5:
        batch = list(range(len(chunks)))
        if order:
            rng.shuffle(batch)
            if len(batch) > 1 and batch[0] == order[-1]:
                batch.append(batch.pop(0))
        for index in batch:
            order.append(index)
            total += chunks[index][1]
            if total >= seconds + 5:
                break
    listing = scratch / "reel.ffconcat"
    lines = ["ffconcat version 1.0"] + [f"file '{chunks[i][0].as_posix()}'" for i in order]
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Give the compositor and teaser one continuous video input with a seekable timeline.
    reel = scratch / "reel.mp4"
    subprocess.run(FFMPEG + ["-f", "concat", "-safe", "0", "-i", str(listing),
                             "-map", "0:v:0", "-c:v", "copy", "-t", str(seconds + 2),
                             str(reel)], check=True)
    print(f"Built a {unique / 60:.1f}-minute reel from {len(shots)} shots in {len(chunks)} chunks.",
          flush=True)
    return reel, unique
