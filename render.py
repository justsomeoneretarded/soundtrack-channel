import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RATE = 48000
WIDTH, HEIGHT = 1280, 720
FFMPEG = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")


@dataclass
class Passage:
    source: int
    start: int
    end: int
    energy: float


def terms(value):
    value = unicodedata.normalize("NFKD", value.casefold())
    return set(re.findall(r"[^\W_]+", value))


def sources_for(title, scratch):
    yt = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
          "--socket-timeout", "15", "--retries", "1"]
    words = terms(title)
    core = {word for word in words if not re.fullmatch(r"(?:19|20)\d{2}", word)} or words
    requested_year = words - core
    candidates = {}
    for query in (f"{title} official soundtrack", f"{title} original score"):
        try:
            result = subprocess.run(
                yt + ["--flat-playlist", "--dump-json", f"ytsearch12:{query}"],
                stdout=subprocess.PIPE, text=True, timeout=150,
            )
        except subprocess.TimeoutExpired:
            continue
        for line in result.stdout.splitlines():
            item = json.loads(line)
            name = item.get("title") or ""
            tokens = terms(name)
            duration = item.get("duration") or 0
            video_id = item.get("id") or ""
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                continue
            if not core <= tokens or not 30 <= duration <= 7200:
                continue
            if tokens & {"trailer", "reaction", "cover", "karaoke", "remake", "slowed", "extended", "remix", "loop"}:
                continue
            if not tokens & {"soundtrack", "score", "ost", "theme", "music"}:
                continue
            years = {token for token in tokens if re.fullmatch(r"(?:19|20)\d{2}", token)}
            if requested_year and years and not requested_year & years:
                continue
            official = "official" in tokens or (item.get("channel") or "").endswith(" - Topic")
            rank = 5 * bool(requested_year & years) + 4 * official + 2 * bool(tokens & {"soundtrack", "score", "ost"})
            candidates[video_id] = (rank, duration, name)
    files, credits = [], []
    ranked = sorted(candidates.items(), key=lambda pair: (-pair[1][0], pair[1][1], pair[0]))
    used_names = set()
    seconds = 0
    for video_id, (_, duration, name) in ranked[:10]:
        signature = frozenset(terms(name))
        if signature in used_names:
            continue
        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            download = subprocess.run(
                yt + ["--no-progress", "-f", "bestaudio/best", "--max-filesize", "200M",
                      "-o", str(scratch / f"{video_id}.%(ext)s"), url],
                timeout=200,
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
        if len(files) >= 6 or seconds >= 1800:
            break
    if not files:
        raise SystemExit("No matching YouTube audio could be downloaded. Try adding the release year. "
                         "For local rendering, supply downloadable audio with --audio.")
    return files, credits


def prepare(files, credits, scratch):
    sources, passages, usable_credits = [], [], []
    for path, credit in zip(files, credits):
        raw = scratch / f"source-{len(sources)}.pcm"
        decoded = subprocess.run(FFMPEG + ["-i", str(path), "-map", "0:a:0", "-t", "1800",
                                          "-ac", "2", "-ar", str(RATE), "-f", "f32le", str(raw)])
        if decoded.returncode or raw.stat().st_size < RATE * 8 * 2:
            print(f"Skipping unavailable or empty audio from {credit[0]!r}.", file=sys.stderr)
            continue
        samples = np.memmap(raw, dtype="<f4", mode="r").reshape(-1, 2)
        # Sample every 12th frame to estimate RMS for each quarter second.
        frames = RATE // 4
        windows = len(samples) // frames
        probe = samples[:windows * frames:12].reshape(windows, -1, 2)
        envelope = np.sqrt(np.mean(np.square(probe), axis=(1, 2)))
        active = np.flatnonzero(envelope > max(0.0005, np.percentile(envelope, 80) * 0.06))
        if active.size < 8:
            continue
        first, last = int(active[0]), int(active[-1]) + 1
        source = len(sources)
        sources.append(samples)
        usable_credits.append(credit)
        # Use quiet points close to the target length to avoid cutting through accents.
        start = first
        target_length = min(76, max(5, (last - first) // 20))
        radius = min(9, max(1, target_length // 7))
        while start < last:
            end = last
            if last - start > 4 * (target_length + 2 * radius + 4):
                target = start + 4 * target_length
                lo, hi = target - 4 * radius, target + 4 * radius
                end = lo + int(np.argmin(envelope[lo:hi]))
            energy = float(np.percentile(envelope[start:end], 65))
            passages.append(Passage(source, start * frames, end * frames, energy))
            start = end
    if not passages:
        raise SystemExit("No usable audio passages were found.")
    return sources, passages, usable_credits


def write_mix(path, sources, passages, secs, rng):
    goal = secs * RATE
    position, previous = 0, None
    recent = []
    energies = [passage.energy for passage in passages]
    low, high = np.percentile(energies, [15, 85])
    floor = max(float(high - low), 0.001)
    # Opening, ascent, release, peak, and resolution.
    arc = [0.18, 0.56, 0.78, 0.38, 0.58, 0.94, 0.25]
    with path.open("wb") as output:
        while position < goal:
            target = float(np.interp(position / goal, np.linspace(0, 1, len(arc)), arc))
            pool = [index for index in range(len(passages)) if index not in recent] or list(range(len(passages)))

            def cost(index):
                passage = passages[index]
                energy = np.clip((passage.energy - low) / floor, 0, 1)
                same_source = bool(recent and passage.source == passages[recent[-1]].source)
                return abs(energy - target) + 0.22 * same_source + rng.uniform(0, 0.45)

            index = min(pool, key=cost)
            passage = passages[index]
            recent = (recent + [index])[-min(8, max(1, len(passages) - 1)):]
            clip = np.array(sources[passage.source][passage.start:passage.end])
            rms = max(passage.energy, 0.005)
            gain = min((0.09 / rms) ** 0.65, 3.0, 0.85 / max(float(np.max(np.abs(clip))), 0.001))
            clip *= gain
            overlap = min(len(clip) // 4, int(rng.uniform(1.4, 3.8) * RATE))
            if previous is not None:
                overlap = min(len(previous), overlap)
                theta = np.linspace(0, math.pi / 2, overlap, dtype=np.float32)[:, None]
                blend = previous[-overlap:] * np.cos(theta) + clip[:overlap] * np.sin(theta)
                clip = np.concatenate((previous[:-overlap], blend, clip[overlap:]))
            else:
                fade = min(len(clip), RATE // 5)
                clip[:fade] *= np.linspace(0.1, 1, fade, dtype=np.float32)[:, None]
            remaining = goal - position
            if len(clip) >= remaining:
                clip = clip[:remaining]
                fade = min(len(clip), 6 * RATE)
                clip[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
                previous = None
            else:
                # Keep the tail in memory so the next clip can blend into it.
                tail = min(4 * RATE, len(clip) // 3, max(1, remaining - len(clip) + RATE))
                previous, clip = clip[-tail:], clip[:-tail]
            np.clip(clip, -0.97, 0.97).astype("<f4", copy=False).tofile(output)
            position += len(clip)
    print(f"Arranged {position / RATE:.3f}s from {len(passages)} passages.", flush=True)


def make_art(title, seed, thumbnail, background):
    rng = random.Random(seed)
    hue = rng.randrange(170, 300)
    from colorsys import hsv_to_rgb

    accent = np.array(hsv_to_rgb(hue / 360, 0.55, 1)) * 255
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    center_x, horizon = int(WIDTH * 0.74), int(HEIGHT * 0.59)
    light = np.exp(-((x - center_x) / 390) ** 2 - ((y - horizon) / 210) ** 2)
    pixels = np.array([6, 10, 24]) + light[..., None] * accent * 0.44
    pixels += np.random.default_rng(seed).normal(0, 0.55, (HEIGHT, WIDTH, 1))
    art = Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))
    draw = ImageDraw.Draw(art, "RGBA")
    for _ in range(310):
        px, py = rng.randrange(WIDTH), rng.randrange(horizon)
        radius = rng.choices([0.6, 1, 1.6], [75, 22, 3])[0]
        draw.ellipse((px-radius, py-radius, px+radius, py+radius),
                     fill=(213, 232, 255, rng.randrange(25, 155)))
    for radius, alpha in [(145, 175), (156, 55), (177, 25), (201, 17)]:
        cy = horizon - 95
        bounds = (center_x-radius, cy-radius, center_x+radius, cy+radius)
        if radius == 145:
            draw.ellipse(bounds, fill=(8, 12, 37, 224))
        draw.ellipse(bounds, outline=(*map(int, accent), alpha), width=2)

    for layer in range(4):
        base = (10, 16, 34)
        color = tuple(int(v + (3-layer) * 0.085 * a) for v, a in zip(base, accent))
        heights = []
        phase = rng.uniform(0, 6)
        for px in range(0, WIDTH + 1, 8):
            curve = 35 * math.sin(px / 157 + phase) + 17 * math.sin(px / 59 + phase)
            ridge = (6 + layer * 2) * math.sin(px / 18 + 2 * phase)
            heights.append((px, int(horizon + 26 + layer * 47 + curve + ridge + rng.randrange(0, 8))))
        draw.polygon(heights + [(WIDTH, HEIGHT), (0, HEIGHT)], fill=color)

    draw.line((64, 632, 1216, 632), fill=(*map(int, accent), 67))
    draw.line((64, 96, 154, 96), fill=(*map(int, accent), 255), width=3)
    mono = ImageFont.truetype(str(FONT_ROOT / "DejaVuSansMono.ttf"), 21)
    draw.text((64, 116), "CINEMATIC SCORE MIX", font=mono, fill=tuple(map(int, accent)))

    for size in range(83, 21, -3):
        font = ImageFont.truetype(str(FONT_ROOT / "DejaVuSans-Bold.ttf"), size)
        lines = [""]
        for word in title.split():
            trial = (lines[-1] + " " + word).strip()
            if lines[-1] and font.getlength(trial) > 715:
                lines.append(word)
            else:
                lines[-1] = trial
        if len(lines) <= 3 and max(map(font.getlength, lines)) <= 715:
            break
    while font.getlength(lines[-1]) > 715:
        lines[-1] = lines[-1][:-2] + "…"
    for index, line in enumerate(lines[:3]):
        draw.text((60, 160 + index * (font.size + 4)), line, font=font,
                  fill=(244, 245, 255), stroke_width=2, stroke_fill=(10, 13, 33))
    art.save(background)
    draw.text((64, 646), "CONTINUOUS SOUNDTRACK MIX", font=mono, fill=(192, 207, 241))
    art.save(thumbnail, quality=92)
    return "".join(f"{int(channel):02x}" for channel in accent)


def render_video(audio, backdrop, output, accent, seconds, draft):
    width, height, fps = (640, 360, 12) if draft else (WIDTH, HEIGHT, 24)
    wave_width, wave_height = round(width * 0.9), round(height * 0.083)
    graph = (
        f"[0:v]scale={width}:{height},format=yuv420p,loop=-1:1:0,setpts=N/({fps}*TB)[sky];"
        "[1:a]asplit[waveaudio][mix];"
        f"[waveaudio]aformat=channel_layouts=mono,showwaves=s={wave_width}x{wave_height}:r={fps}:mode=cline:"
        f"scale=sqrt:colors=0x{accent},format=rgba,colorkey=0x000000:0.08:0.12[wave];"
        f"[sky][wave]overlay=x=(W-w)/2:y=H-h-H*0.043:eof_action=pass,"
        f"fade=t=in:st=0:d=1,fade=t=out:st={max(0, seconds-3)}:d=3,setsar=1[v];"
        f"[mix]afade=t=out:st={max(0, seconds-5)}:d=5[a]"
    )
    subprocess.run(FFMPEG + [
        "-filter_complex_threads", "2",
        "-framerate", str(fps), "-i", str(backdrop),
        "-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i", str(audio),
        "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-t", str(seconds),
        "-c:v", "libx264", "-threads", "4", "-preset", "veryfast", "-tune", "stillimage",
        "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(output),
    ], check=True)
    duration = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(output),
    ], text=True)
    if abs(float(duration) - seconds) > 0.1:
        raise RuntimeError(f"Expected {seconds}s, rendered {duration.strip()}s")


def write_metadata(out, title, seconds, credits):
    length = "1 Hour" if seconds == 3600 else f"{seconds}-Second Preview"
    suffix = f" Soundtrack Mix | {length} Cinematic Score"
    video_title = title[:100 - len(suffix)].rstrip() + suffix
    opening = (
        f"Listen to a continuous one-hour {title} soundtrack mix."
        if seconds == 3600 else f"Preview the continuous {title} soundtrack mix in {seconds} seconds."
    )
    description = (
        f"{opening} "
        "The edit moves through quieter passages, fuller arrangements and a final resolution, "
        "paired with an original nightscape and audio-reactive visualizer.\n\n"
        "Independent mix.\n\n"
        "Audio sources\n" + "\n".join(f"{name} — {link}" if link else name for name, link in credits)
        + "\n\n#Soundtrack #FilmScore #CinematicMusic\n"
    )
    (out / "youtube-title.txt").write_text(video_title + "\n", encoding="utf-8")
    (out / "youtube-description.txt").write_text(description, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--audio", nargs="+", type=Path, help="Local audio files for offline runs")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--draft", action="store_true", help="Render a smaller, faster preview")
    parser.add_argument("--out", type=Path, default=Path("dist"))
    args = parser.parse_args()
    title = " ".join(args.title.split())
    if not terms(title) or len(title) > 160 or not 5 <= args.seconds <= 3600:
        parser.error("Provide a title of 1–160 characters and a duration of 5–3600 seconds.")
    seed = int.from_bytes(hashlib.sha256(title.encode()).digest()[:8])
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="soundtrack-") as temp:
        scratch = Path(temp)
        if args.audio:
            files, credits = args.audio, [(path.stem, "") for path in args.audio]
        else:
            files, credits = sources_for(title, scratch)
        sources, passages, credits = prepare(files, credits, scratch)
        if len(passages) < 3 and args.seconds >= 600:
            raise SystemExit("Only one or two usable passages were found. Add more source audio for an hour with variation.")
        mix = scratch / "mix.pcm"
        backdrop = scratch / "backdrop.png"
        write_mix(mix, sources, passages, args.seconds, random.Random(seed))
        accent = make_art(title, seed, args.out / "thumbnail.jpg", backdrop)
        render_video(mix, backdrop, args.out / "soundtrack.mp4", accent, args.seconds, args.draft)
        write_metadata(args.out, title, args.seconds, credits)


if __name__ == "__main__":
    main()
