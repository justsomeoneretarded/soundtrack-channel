"""Command-line entry point: find sources, arrange, render, and write upload copy."""
import argparse
import hashlib
from pathlib import Path
import random
import tempfile
import time

from .audio import loudest_window, loudness_filter, loudness_gain, measure_loudness, prepare, write_mix
from .common import clean_composer, length_label, terms
from .metadata import chapters, write_ffmetadata, write_outputs
from .footage import analyse, characters_for, gather, plan_scenes, portraits
from .sources import cue_labels, download_links, sources_for
from .video import render_short, render_video
from .visuals import (character_layer, make_art, make_vertical_art, now_playing_cards,
                      portrait_timeline, scene_shade)

MAX_SECONDS = 3 * 3600


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Produce a long-form soundtrack video, thumbnail, Shorts teaser, and YouTube copy.")
    parser.add_argument("--title", required=True, help="Movie or show title, optionally with its year")
    parser.add_argument("--composer", default="", help="Composer name for search, titles, and tags")
    parser.add_argument("--style", choices=("mix", "album"), default="mix",
                        help="mix: passages arranged along an intensity arc; album: cues play in order")
    parser.add_argument("--audio", nargs="+", type=Path, help="Local audio files for offline runs")
    parser.add_argument("--audio-url", nargs="+", default=[],
                        help="Audio links (direct files, Google Drive, Dropbox, or any yt-dlp site)")
    parser.add_argument("--clips", nargs="+", type=Path, help="Local trailer or scene files for visuals")
    parser.add_argument("--characters", default="",
                        help="Comma-separated character names to search scenes for (default: from Wikidata)")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--minutes", type=float, help="Duration in minutes; overrides --seconds")
    parser.add_argument("--cutaways", type=int, help="Number of scene cutaways (default: about one every 7 minutes, 0 disables)")
    parser.add_argument("--short-seconds", type=int, default=45,
                        help="Length of the vertical Shorts teaser; 0 skips it")
    parser.add_argument("--no-loudnorm", action="store_true", help="Skip loudness normalisation")
    parser.add_argument("--draft", action="store_true", help="Render a smaller, faster preview")
    parser.add_argument("--out", type=Path, default=Path("dist"))
    args = parser.parse_args(argv)
    if args.minutes is not None:
        args.seconds = round(args.minutes * 60)
    args.title = " ".join(args.title.split())
    args.composer = clean_composer(args.composer)
    if not terms(args.title) or len(args.title) > 160 or not 5 <= args.seconds <= MAX_SECONDS:
        parser.error(f"Provide a title of 1–160 characters and a duration of 5–{MAX_SECONDS} seconds.")
    args.audio_url = [url for value in args.audio_url for url in value.split() if url]
    if len(args.composer) > 80:
        parser.error("Keep the composer name under 80 characters.")
    if not 0 <= args.short_seconds <= 60:
        parser.error("The Shorts teaser must be 0–60 seconds.")
    return args


def main(argv=None):
    args = parse_args(argv)
    began = time.monotonic()
    title, seconds = args.title, args.seconds
    seed = int.from_bytes(hashlib.sha256(title.encode()).digest()[:8], "big")
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="soundtrack-") as temp:
        scratch = Path(temp)
        if args.audio:
            files, credits = args.audio, [(path.stem, "") for path in args.audio]
        elif args.audio_url:
            files, credits = download_links(args.audio_url, scratch / "audio-links")
        else:
            files, credits = sources_for(title, scratch / "audio-uploads", args.composer)
        sources, passages, credits = prepare(files, credits, scratch, args.style)
        if len(passages) < 3 and seconds >= 600:
            raise SystemExit("Only one or two usable passages were found. "
                             "Add more source audio for a long mix with variation.")
        labels = cue_labels([name for name, _ in credits], title, args.composer)

        characters = [name.strip() for name in args.characters.split(",") if name.strip()]
        if args.clips:
            clip_files, clip_credits = args.clips, [(path.stem, "") for path in args.clips]
        elif args.audio:
            clip_files, clip_credits = [], []
        else:
            characters = characters or characters_for(title)
            clip_files, clip_credits = gather(title, scratch / "visual-uploads", characters)
        analysis = analyse(clip_files, clip_credits, scratch / "analysis")
        scenes, scene_details = plan_scenes(analysis, seconds, args.cutaways)
        faces = portraits(analysis)
        lead = next(iter(faces.values()), None)
        visual_credits = [video.credit for video in analysis.videos] if scenes or faces else []

        mix = scratch / "mix.pcm"
        timeline = write_mix(mix, sources, passages, seconds, random.Random(seed), args.style)
        loudness = None if args.no_loudnorm else measure_loudness(mix)
        audio_filter = loudness_filter(loudness)
        if loudness:
            loudness["applied_gain_db"] = loudness_gain(loudness)

        backdrop, eclipse, character = (scratch / name for name in ("backdrop.png", "eclipse.png", "character.png"))
        preview = length_label(seconds).endswith("Preview")
        accent = make_art(title, seed, args.out / "thumbnail.jpg", backdrop, eclipse, character, lead,
                          badge=None if preview else length_label(seconds))
        layers = {}
        for person, image in faces.items():
            layers[person] = scratch / f"character-{len(layers)}.png"
            character_layer(image, accent).save(layers[person])
        portrait_list = portrait_timeline(layers, scenes, [d["character"] for d in scene_details],
                                          seconds, scratch / "characters.ffconcat")
        cards = now_playing_cards([(start, labels[source]) for start, source in timeline],
                                  accent, scratch / "cards", seconds)
        chapter_list = chapters(args.style, timeline, labels, seconds)
        metadata = write_ffmetadata(scratch / "chapters.txt", title, chapter_list, seconds) if chapter_list else None
        render_video(mix, backdrop, eclipse, portrait_list, scenes,
                     args.out / "soundtrack.mp4", seconds, args.draft, audio_filter, cards, metadata,
                     scene_shade(scratch / "shade.png") if scenes else None)
        print(f"Rendered soundtrack.mp4 after {time.monotonic() - began:.0f}s.", flush=True)

        short = None
        if args.short_seconds and seconds >= args.short_seconds + 10:
            start = loudest_window(mix, seconds, args.short_seconds)
            poster = scratch / "vertical.png"
            cta = f"Full {length_label(seconds).lower()} mix on the channel" if not preview else "Full mix on the channel"
            make_vertical_art(title, args.out / "thumbnail.jpg", poster, accent, cta)
            best = sorted(zip(scenes, scene_details), key=lambda pair: -pair[1]["score"])[:3]
            render_short(mix, poster, args.out / "short.mp4", start, args.short_seconds, audio_filter,
                         args.draft, [(path, offset, length) for (path, offset, length, _), _ in best])
            short = {"file": "short.mp4", "source_start": start, "seconds": args.short_seconds}

        write_outputs(args.out, title=title, seconds=seconds, style=args.style, composer=args.composer,
                      timeline=timeline, labels=labels, chapter_list=chapter_list, credits=credits,
                      visual_credits=visual_credits, loudness=loudness, scenes=scene_details, short=short,
                      characters=characters, people=len(analysis.people))
    print(f"Finished in {time.monotonic() - began:.0f}s. Outputs are in {args.out}/.", flush=True)
