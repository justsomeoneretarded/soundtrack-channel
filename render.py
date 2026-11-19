import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

RATE = 48000
WIDTH, HEIGHT = 1280, 720
FFMPEG = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
YOUTUBE = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
           "--socket-timeout", "15", "--retries", "1"]
FONT_ROOT = Path(os.environ.get("SOUNDTRACK_FONT_DIR", "/usr/share/fonts/truetype/dejavu"))


@dataclass
class Passage:
    source: int
    start: int
    end: int
    energy: float


def terms(value):
    value = unicodedata.normalize("NFKD", value.casefold())
    return set(re.findall(r"[^\W_]+", value))


def matching_uploads(title, queries, required, excluded, duration_limits):
    words = terms(title)
    core = {word for word in words if not re.fullmatch(r"(?:19|20)\d{2}", word)} or words
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
            item = json.loads(line)
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
            years = {token for token in tokens if re.fullmatch(r"(?:19|20)\d{2}", token)}
            if requested_year and years and not requested_year & years:
                continue
            official = "official" in tokens or (item.get("channel") or "").endswith(" - Topic")
            rank = 5 * bool(requested_year & years) + 4 * official
            rank += 2 * bool(tokens & {"soundtrack", "score", "ost", "clip"})
            candidates[video_id] = (rank, duration, name)
    return sorted(candidates.items(), key=lambda pair: (-pair[1][0], pair[1][1], pair[0]))


def download_uploads(candidates, scratch, stream_format, limit, budget, attempts):
    scratch.mkdir()
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


def sources_for(title, scratch):
    candidates = matching_uploads(
        title, ("official soundtrack", "original score"),
        {"soundtrack", "score", "ost", "theme", "music"},
        {"trailer", "reaction", "cover", "karaoke", "remake", "slowed", "extended", "remix", "loop"},
        (30, 7200),
    )
    files, credits = download_uploads(candidates, scratch, "bestaudio/best", 6, 1800, 10)
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


def inspect_footage(files, credits, scratch):
    if not files:
        return None, []
    import cv2

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    best_portrait, best_score = None, -1
    footage = []
    for index, (path, credit) in enumerate(zip(files, credits)):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
             "-of", "json", str(path)], capture_output=True, text=True,
        )
        if probe.returncode:
            continue
        details = json.loads(probe.stdout)
        if not any(stream["codec_type"] == "video" for stream in details.get("streams", [])):
            continue
        try:
            duration = float(details["format"]["duration"])
        except (KeyError, ValueError):
            continue
        if not math.isfinite(duration) or duration < 4:
            continue
        footage.append((path, duration, credit))
        for fraction in (0.16, 0.30, 0.44, 0.56, 0.68, 0.79):
            still = scratch / f"frame-{index}.jpg"
            extracted = subprocess.run(FFMPEG + [
                "-ss", str(duration * fraction), "-i", str(path), "-map", "0:v:0", "-frames:v", "1",
                "-vf", "scale=800:-2", "-update", "1", str(still),
            ])
            if extracted.returncode or not still.exists():
                continue
            with Image.open(still) as image:
                rgb = image.convert("RGB")
            gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
            for x, y, w, h in faces:
                cx, cy = x + w / 2, y + h / 2
                score = float(w * h) * (1 - abs(cx / rgb.width - 0.5) * 0.65)
                if score <= best_score:
                    continue
                crop_w = min(rgb.width, rgb.height * 0.91, max(210, w * 3.5))
                crop_h = crop_w / 0.91
                left = max(0, min(rgb.width - crop_w, cx - crop_w / 2))
                top = max(0, min(rgb.height - crop_h, cy - crop_h * 0.37))
                crop = rgb.crop((int(left), int(top), int(left + crop_w), int(top + crop_h)))
                best_portrait = ImageOps.fit(crop, (420, 462), Image.Resampling.LANCZOS)
                best_score = score
    return best_portrait, footage


def create_eclipse(accent):
    eclipse = Image.new("RGBA", (410, 410))
    draw = ImageDraw.Draw(eclipse)
    for radius, alpha in [(201, 35), (177, 64), (156, 110), (145, 230)]:
        bounds = (205-radius, 205-radius, 205+radius, 205+radius)
        if radius == 145:
            draw.ellipse(bounds, fill=(8, 12, 37, 194))
        draw.ellipse(bounds, outline=(*accent, alpha), width=2)
    draw.ellipse((61, 168, 68, 175), fill=(*accent, 246))
    draw.ellipse((357, 250, 364, 257), fill=(234, 247, 255, 224))
    return eclipse


def character_layer(portrait, accent):
    if portrait is None:
        return Image.new("RGBA", (420, 462))
    mask = Image.new("L", portrait.size)
    ImageDraw.Draw(mask).rounded_rectangle((7, 7, 412, 454), radius=96, fill=255)
    feather = np.asarray(mask.filter(ImageFilter.GaussianBlur(9)), dtype=np.float32)
    vertical = np.linspace(1, 0, 122)
    feather[-122:] *= vertical[:, None]
    portrait = portrait.convert("RGBA")
    portrait.putalpha(Image.fromarray(feather.astype(np.uint8)))
    edge = Image.new("RGBA", portrait.size)
    ImageDraw.Draw(edge).rounded_rectangle((9, 9, 410, 452), radius=94,
                                           outline=(*accent, 110), width=2)
    edge_alpha = np.array(edge.getchannel("A"))
    edge_alpha[-122:] = (edge_alpha[-122:] * vertical[:, None]).astype(np.uint8)
    edge.putalpha(Image.fromarray(edge_alpha))
    return Image.alpha_composite(portrait, edge)


def scene_plan(footage, seconds):
    if not footage or seconds < 12:
        return []
    length = min(10, max(3, seconds * 0.10))
    moments = (0.18, 0.41, 0.65, 0.84) if seconds >= 60 else (0.23, 0.64)
    setup = []
    for index, fraction in enumerate(moments):
        path, duration, _ = footage[index % len(footage)]
        cut_length = min(length, duration - 1)
        fraction_in_source = (0.25, 0.61, 0.53, 0.32)[index]
        source_start = max(0, duration * fraction_in_source - cut_length / 2)
        setup.append((path, min(source_start, duration - cut_length - 0.1),
                      cut_length, seconds * fraction))
    return setup


def make_art(title, seed, thumbnail, background, eclipse_path, character_path, portrait):
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
    color = tuple(map(int, accent))
    eclipse = create_eclipse(color)
    eclipse.save(eclipse_path)
    character = character_layer(portrait, color)
    character.save(character_path)
    if portrait is not None:
        eclipse = eclipse.resize((530, 530), Image.Resampling.LANCZOS)
    art.paste(eclipse, (center_x - eclipse.width // 2, horizon - 95 - eclipse.height // 2), eclipse)
    art.paste(character, (750, 106), character)
    draw.text((64, 646), "CONTINUOUS SOUNDTRACK MIX", font=mono, fill=(192, 207, 241))
    art.save(thumbnail, quality=92)


def render_video(audio, backdrop, eclipse, character, scenes, output, seconds, draft):
    width, height, fps = (640, 360, 12) if draft else (WIDTH, HEIGHT, 24)
    wave_width, wave_height = round(width * 0.90), round(height * 0.083)
    ratio = width / WIDTH
    disc_size = round((530 if character else 410) * ratio)
    inputs = [
        "-framerate", str(fps), "-i", str(backdrop),
        "-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i", str(audio),
        "-framerate", str(fps), "-i", str(eclipse),
    ]
    graph = [
        f"[0:v]scale={width}:{height},format=yuv420p,loop=-1:1:0,setpts=N[sky]",
        f"[2:v]scale={disc_size}:{disc_size},format=yuva420p[disc]",
        "[sky][disc]overlay=x='W*0.74-w/2+W*0.026*sin(t/7)':"
        "y='H*0.45-h/2+H*0.04*sin(t/10)'[eclipse]",
        "[1:a]asplit[waveaudio][mix]",
        f"[waveaudio]aformat=channel_layouts=mono,showwaves=s={wave_width}x{wave_height}:r={fps}:"
        "mode=cline:scale=sqrt:draw=full:colors=0x74efb7,format=rgba[wave]",
    ]
    layer = "eclipse"
    next_input = 3
    if character:
        inputs += ["-framerate", str(fps), "-i", str(character)]
        graph += [
            f"[3:v]scale={round(420*ratio)}:{round(462*ratio)},format=yuva420p[portrait]",
            "[eclipse][portrait]overlay=x=W*0.586:y=H*0.147[characters]",
        ]
        layer = "characters"
        next_input += 1
    graph.append(f"[{layer}][wave]overlay=x=(W-w)/2:"
                 "y='H-h-H*0.043+H*0.01*sin(t*1.4)':eof_action=pass[visualizer]")
    layer = "visualizer"
    scene_w = 2 * round(width * 0.91 / 2)
    scene_h = 2 * round(height * 0.87 / 2)
    for index, (path, offset, length, when) in enumerate(scenes):
        inputs += ["-ss", str(offset), "-t", str(length), "-i", str(path)]
        graph += [
            f"[{next_input+index}:v]setpts=PTS-STARTPTS,fps={fps},"
            f"scale={scene_w}:{scene_h}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
            f"pad={scene_w}:{scene_h}:(ow-iw)/2:(oh-ih)/2:color=0x090e1e,format=yuva420p,"
            f"fade=t=in:st=0:d=0.55:alpha=1,fade=t=out:st={length-0.70}:d=0.65:alpha=1,"
            f"setpts=PTS+{when}/TB[cut{index}]",
            f"[{layer}][cut{index}]overlay=x=(W-w)/2:y=(H-h)/2:eof_action=pass:"
            f"enable='between(t,{when},{when+length})'[edit{index}]",
        ]
        layer = f"edit{index}"
    graph += [
        f"[{layer}]fade=t=in:st=0:d=1,fade=t=out:st={max(0, seconds-3)}:d=3,setsar=1[v]",
        f"[mix]afade=t=out:st={max(0, seconds-5)}:d=5[a]",
    ]
    subprocess.run(FFMPEG + ["-filter_complex_threads", "2", *inputs,
        "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]", "-t", str(seconds),
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


def write_metadata(out, title, seconds, credits, visual_credits):
    length = "1 Hour" if seconds == 3600 else f"{seconds}-Second Preview"
    suffix = f" Soundtrack Mix | {length} Cinematic Score"
    video_title = title[:100 - len(suffix)].rstrip() + suffix
    opening = (
        f"Listen to a continuous one-hour {title} soundtrack mix."
        if seconds == 3600 else f"Preview the continuous {title} soundtrack mix in {seconds} seconds."
    )
    visual_note = " Selected scene cutaways are paired with the score." if visual_credits else ""
    visual_links = "\n\nVisual sources\n" + "\n".join(
        f"{name} — {link}" if link else name for name, link in visual_credits
    ) if visual_credits else ""
    description = (
        f"{opening} "
        "The edit moves through quieter passages, fuller arrangements and a final resolution, "
        f"paired with an eclipse nightscape and audio-reactive visualizer.{visual_note}\n\n"
        "Independent mix.\n\n"
        "Audio sources\n" + "\n".join(f"{name} — {link}" if link else name for name, link in credits)
        + visual_links + "\n\n#Soundtrack #FilmScore #CinematicMusic\n"
    )
    (out / "youtube-title.txt").write_text(video_title + "\n", encoding="utf-8")
    (out / "youtube-description.txt").write_text(description, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--audio", nargs="+", type=Path, help="Local audio files for offline runs")
    parser.add_argument("--clips", nargs="+", type=Path, help="Local trailer or scene files for visuals")
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
            files, credits = sources_for(title, scratch / "audio-uploads")
        sources, passages, credits = prepare(files, credits, scratch)
        if len(passages) < 3 and args.seconds >= 600:
            raise SystemExit("Only one or two usable passages were found. Add more source audio for an hour with variation.")
        if args.clips:
            clip_files, clip_credits = args.clips, [(path.stem, "") for path in args.clips]
        elif args.audio:
            clip_files, clip_credits = [], []
        else:
            clip_files, clip_credits = footage_for(title, scratch / "visual-uploads")
        portrait, footage = inspect_footage(clip_files, clip_credits, scratch)
        scenes = scene_plan(footage, args.seconds)
        visual_credits = [credit for path, _, credit in footage if scenes or portrait is not None]
        mix = scratch / "mix.pcm"
        backdrop = scratch / "backdrop.png"
        write_mix(mix, sources, passages, args.seconds, random.Random(seed))
        eclipse = scratch / "eclipse.png"
        character = scratch / "character.png"
        make_art(title, seed, args.out / "thumbnail.jpg", backdrop, eclipse, character, portrait)
        render_video(mix, backdrop, eclipse, character if portrait is not None else None, scenes,
                     args.out / "soundtrack.mp4", args.seconds, args.draft)
        write_metadata(args.out, title, args.seconds, credits, visual_credits)


if __name__ == "__main__":
    main()
