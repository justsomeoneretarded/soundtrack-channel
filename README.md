# Soundtrack Video

Enter a movie or show title in GitHub Actions to create a 60-minute cinematic soundtrack mix, a thumbnail, and YouTube upload copy.

## Run on GitHub

1. Copy this project's files into a GitHub repository. Include the hidden `.github/workflows/render.yml` file on the default branch.
2. Open **Actions → Render soundtrack video → Run workflow**. Enter a movie or show name. Include the release year for ambiguous titles.
3. Download the **soundtrack-video** artifact from the completed run.

The artifact contains:

| File | Purpose |
| --- | --- |
| `soundtrack.mp4` | 1280×720, 24 fps, H.264 video and stereo AAC audio, exactly 60 minutes |
| `thumbnail.jpg` | Original 1280×720 nightscape designed for the title |
| `youtube-title.txt` | Search-friendly title with the work name, soundtrack, and duration |
| `youtube-description.txt` | Ready-to-edit description with relevant terms, source credits, links, and hashtags |

The workflow uses the standard Ubuntu runner with no paid APIs and no repository secrets. GitHub offers standard runners free for public repositories. Private repositories consume their included Actions minutes and artifact storage. Artifacts expire after two days.

## How it works

The workflow searches YouTube for the title plus “official soundtrack” and “original score.” It scores the results for title and year matches and score-related terms, filters common false matches, and attempts to download audio from the strongest candidates. Search and upload labels are heuristics. Check the credited sources before publishing, and use audio you have permission to download, edit, and publish. Credits alone do not grant those rights. Content ID may claim or block an upload.

The arranger finds quieter transition points, varies the order of passages along an intensity curve, and overlaps them with variable-length equal-power crossfades. The original procedural nightscape uses a title-derived color palette and a waveform driven by the final mix. Very short sources can still sound repetitive and are rejected when there are too few usable passages for a long render.

YouTube sometimes rejects traffic from GitHub-hosted runners. A failed search or download fails the job with diagnostic logs. The local audio mode below uses the same production renderer if you already have the soundtrack files.

## Run locally

Install Python 3.10+, FFmpeg, and the DejaVu font package. Install Deno for YouTube extraction.

```bash
python -m pip install -r requirements.txt
python render.py --title 'Example Movie (2024)'
python render.py --title 'Example Movie (2024)' --audio cue-1.flac cue-2.mp3
python render.py --title 'Example Movie (2024)' --audio cue-1.flac --seconds 20 --draft
```

## Ten repositories reviewed

These are established public projects ranked here by the part of the workflow they can inform, with direct dependencies marked in the table.

| Repository | Application |
| --- | --- |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | Used for title search, candidate metadata, and downloading audio streams. |
| [FFmpeg](https://github.com/FFmpeg/FFmpeg) | Used for audio decoding, waveform generation, encoding, and duration inspection. |
| [Pillow](https://github.com/python-pillow/Pillow) | Used to generate the original backdrops and thumbnails. |
| [librosa](https://github.com/librosa/librosa) | Reference for using short frames and RMS to estimate musical intensity. |
| [Essentia](https://github.com/MTG/essentia) | Reference for structuring feature extraction before arrangement. |
| [aubio](https://github.com/aubio/aubio) | Reference for recognizing the difference between low-energy cut points and beat-aware alignment. |
| [LosslessCut](https://github.com/mifi/lossless-cut) | Reference for using explicit source time ranges and avoiding intermediate transcodes. |
| [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) | Reference for splitting and recombining streams in a filter graph. |
| [MoviePy](https://github.com/Zulko/moviepy) | Reference for layer composition and duration-managed exports. |
| [Overtone](https://github.com/overtone/overtone) | Reference for separating musical arrangement from audio generation. |
