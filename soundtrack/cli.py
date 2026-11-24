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
from .footage import analyse, gather, still, work_details
from .reel import best_shot, build_reel, frame_size, plan_reel
from .sources import cue_labels, download_links, sources_for
from .video import render_short, render_video
from .visuals import make_thumbnail, plain_backdrop, short_overlay, text_timeline

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

    def stage(message):
        elapsed = int(time.monotonic() - began)
        print(f"[{elapsed // 60:02d}:{elapsed % 60:02d}] {message}", flush=True)

    title, seconds = args.title, args.seconds
    seed = int.from_bytes(hashlib.sha256(title.encode()).digest()[:8], "big")
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="soundtrack-") as temp:
        scratch = Path(temp)
        characters = [name.strip() for name in args.characters.split(",") if name.strip()]
        others = []
        youtube_audio = not args.audio and not args.audio_url
        youtube_clips = not args.clips and not args.audio
        if youtube_audio or youtube_clips:
            found_characters, others = work_details(title)
            characters = characters or found_characters
        stage("Finding audio.")
        if args.audio:
            files, credits = args.audio, [(path.stem, "") for path in args.audio]
        elif args.audio_url:
            files, credits = download_links(args.audio_url, scratch / "audio-links")
        else:
            files, credits = sources_for(title, scratch / "audio-uploads", args.composer, seconds, others)
        sources, passages, credits = prepare(files, credits, scratch, args.style)
        if len(passages) < 3 and seconds >= 600:
            raise SystemExit("Only one or two usable passages were found. "
                             "Add more source audio for a long mix with variation.")
        labels = cue_labels([name for name, _ in credits], title, args.composer)

        stage(f"Prepared {len(passages)} passages from {len(credits)} sources. Finding scenes.")
        if args.clips:
            clip_files, clip_credits = args.clips, [(path.stem, "") for path in args.clips]
        elif args.audio:
            clip_files, clip_credits = [], []
        else:
            clip_files, clip_credits = gather(title, scratch / "visual-uploads", characters, others)
        stage(f"Analysing {len(clip_files)} videos.")
        analysis = analyse(clip_files, clip_credits, scratch / "analysis")
        shots = plan_reel(analysis, seed)
        visual_credits = [video.credit for video in analysis.videos] if shots else []

        stage(f"Chose {len(shots)} shots. Arranging the mix.")
        mix = scratch / "mix.pcm"
        timeline = write_mix(mix, sources, passages, seconds, random.Random(seed), args.style)
        loudness = None if args.no_loudnorm else measure_loudness(mix)
        audio_filter = loudness_filter(loudness)
        if loudness:
            loudness["applied_gain_db"] = loudness_gain(loudness)

        stage("Building the film reel.")
        reel, _ = build_reel(shots, scratch / "reel", seconds, seed, args.draft)
        backdrop = plain_backdrop(scratch / "backdrop.png")
        lead = best_shot(shots) if reel else None
        frame = still(lead.path, lead.start + lead.length / 2) if lead else None
        preview = length_label(seconds).endswith("Preview")
        duration = "Preview" if preview else f"{round(seconds / 60)} minutes"
        make_thumbnail(frame, title, f"Original score  ·  {duration}", args.out / "thumbnail.jpg")
        width, height, _ = frame_size(args.draft)
        text = text_timeline(title, args.composer, [(start, labels[source]) for start, source in timeline],
                             seconds, scratch / "text", (width, height))

        stage("Rendering the video.")
        chapter_list = chapters(args.style, timeline, labels, seconds)
        metadata = write_ffmetadata(scratch / "chapters.txt", title, chapter_list, seconds) if chapter_list else None
        render_video(mix, reel, backdrop, text, args.out / "soundtrack.mp4", seconds, args.draft,
                     audio_filter, metadata)
        print(f"Rendered soundtrack.mp4 after {time.monotonic() - began:.0f}s.", flush=True)

        short = None
        if args.short_seconds and seconds >= args.short_seconds + 10:
            start = loudest_window(mix, seconds, args.short_seconds)
            cta = (f"Full {round(seconds / 60)}-minute mix on the channel" if not preview
                   else "Full mix on the channel")
            overlay = short_overlay(title, cta, scratch / "short-overlay.png",
                                    (540, 960) if args.draft else (1080, 1920))
            render_short(mix, reel, backdrop, overlay, args.out / "short.mp4", start, args.short_seconds,
                         audio_filter, args.draft)
            short = {"file": "short.mp4", "source_start": start, "seconds": args.short_seconds}

        shot_details = [{"source": analysis.videos[shot.video].credit[0], "source_start": shot.start,
                         "length": shot.length, "character": shot.person, "score": shot.score}
                        for shot in shots]
        write_outputs(args.out, title=title, seconds=seconds, style=args.style, composer=args.composer,
                      timeline=timeline, labels=labels, chapter_list=chapter_list, credits=credits,
                      visual_credits=visual_credits, loudness=loudness, scenes=shot_details, short=short,
                      characters=characters, people=len(analysis.people))
    print(f"Finished in {time.monotonic() - began:.0f}s. Outputs are in {args.out}/.", flush=True)
