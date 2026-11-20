# Soundtrack Video

Enter a movie or show title in GitHub Actions to produce a long-form cinematic soundtrack video, a matching thumbnail, a vertical Shorts teaser, and ready-to-paste YouTube copy with chapters and tags.

## Run on GitHub

1. Copy this project's files into a GitHub repository. Include the hidden `.github/workflows/` files on the default branch.
2. Open **Actions → Render soundtrack video → Run workflow**. Enter a movie or show name. Include the release year for ambiguous titles. The other inputs are optional:

   | Input | Default | Effect |
   | --- | --- | --- |
   | `title` | — | The work to search for. Required. |
   | `composer` | blank | Added to the search, the video title, the description, hashtags, and tags. |
   | `style` | `mix` | `mix` arranges passages along an intensity arc. `album` plays each cue in full, in order, with a timestamped tracklist. |
   | `minutes` | `60` | Video length, 1–180. An hour takes roughly 12 minutes to render on a standard runner. |

3. Download the **soundtrack-video** artifact from the completed run.

The artifact contains:

| File | Purpose |
| --- | --- |
| `soundtrack.mp4` | 1280×720, 24 fps, H.264 video and stereo AAC audio at −14 LUFS, exact requested length, with embedded chapters |
| `short.mp4` | 1080×1920 Shorts teaser (45 s) cut from the most intense stretch of the mix |
| `thumbnail.jpg` | 1280×720 eclipse art with a length badge and a character portrait when footage is available |
| `youtube-title.txt` | Search-friendly title with the work name, composer, soundtrack, and duration (≤100 characters) |
| `youtube-description.txt` | Description with YouTube chapters or a tracklist, featured cues, source credits, links, and hashtags |
| `youtube-tags.txt` | Comma-separated tags within YouTube's 500-character limit |
| `manifest.json` | Machine-readable record of the arrangement, chapters, sources, loudness, and cutaways |

The workflow uses the standard Ubuntu runner with no paid APIs and no repository secrets. GitHub offers standard runners free for public repositories. Private repositories consume their included Actions minutes and artifact storage. Artifacts expire after two days.

## How it works

The workflow searches YouTube for the title plus “official soundtrack” and “original score” (and the composer, when given). It scores the results for title and year matches and score-related terms, filters common false matches such as covers, piano tutorials, sped-up edits, and lo-fi remixes, and attempts to download audio from the strongest candidates. Search and upload labels are heuristics. Check the credited sources before publishing, and use audio you have permission to download, edit, and publish. Credits alone do not grant those rights. Content ID may claim or block an upload.

**Cue names.** Upload titles are reduced to readable cue names for the on-screen card, chapters, and tags. Parts that restate the work, describe the upload (“Official Soundtrack”, “HD”), name the composer, or repeat across several uploads (usually the artist or label) are dropped. *Hans Zimmer - Cornfield Chase (Interstellar Official Soundtrack)* becomes *Cornfield Chase*.

**Arrangement.** Each source is split at quiet points into passages, measured for intensity (RMS) and brightness (spectral centroid). In `mix` style, the arranger follows an intensity curve through six movements — Opening, Ascent, Reverie, Gathering, Summit, Resolution — prefers letting a passage run on into its natural continuation for a few minutes, avoids abrupt jumps in timbre, and overlaps changes of cue with variable-length equal-power crossfades. In `album` style, each cue plays in full at a consistent per-cue level and crossfades into the next; the list repeats if the cues are shorter than the requested length. Very short sources can still sound repetitive and are rejected when there are too few usable passages for a long render.

**Loudness.** The finished mix is measured (EBU R128) and brought to YouTube's −14 LUFS reference with a single constant gain. A transparent safety limiter is added only when peaks would exceed −2 dBFS, and it is never asked for more than 3 dB of reduction, so the score keeps its dynamics.

**Picture.** The eclipse drifts behind the foreground portrait while a mint-green waveform responds to the final mix. A “Now playing” card names the current cue, and a progress bar tracks the whole video. The workflow also searches for matching trailers and scene clips. It chooses a prominent, sharp, well-lit face for the foreground portrait and inserts four brief, muted scene cutaways during the video. The score continues underneath. If no clip can be downloaded, the animated eclipse and waveform still render. The description includes separate visual credits when footage is used. Check that you also have permission to reuse the footage before uploading.

**Chapters.** `mix` videos get one chapter per movement. `album` videos get a tracklist with one chapter per cue; cues shorter than ten seconds are merged so YouTube accepts the list. Chapters are written into the description and embedded in the MP4.

YouTube rejected audio downloads from the GitHub-hosted runner during an Interstellar test by requiring a sign-in check. A failed search or download fails the job with diagnostic logs. If you have a Linux x64 machine with working access to your permitted sources, you can register it as a [self-hosted runner](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners), give it the label `soundtrack-render`, and set the repository Actions variable `SOUNDTRACK_RUNNER_LABEL` to that label. The title remains the only input when running the workflow. Availability still depends on YouTube and the runner's network. Use a private repo for self-hosting.

The local audio mode below uses the same production renderer if you already have the soundtrack and clip files.

## Run locally

Install Python 3.10+, FFmpeg, and the DejaVu font package. Install Deno for YouTube extraction.

```bash
python -m pip install -r requirements.txt
python render.py --title 'Example Movie (2024)'
python render.py --title 'Example Movie (2024)' --audio cue-1.flac cue-2.mp3 --clips trailer.mp4
python render.py --title 'Example Movie (2024)' --audio cue-1.flac --seconds 20 --draft
python render.py --title 'Example Movie (2024)' --composer 'Jane Doe' --style album --minutes 90
python render.py --help
```

`--draft` renders at 640×360 and 12 fps for quick checks. `--cutaways N` changes the number of scene cutaways (0 disables them), `--short-seconds 0` skips the Shorts teaser, and `--no-loudnorm` keeps the arranged level.

## Project layout

| Path | Contents |
| --- | --- |
| `render.py` | Entry point used by the workflow |
| `soundtrack/sources.py` | YouTube search, downloads, and cue naming |
| `soundtrack/audio.py` | Passage analysis, arrangement, and loudness |
| `soundtrack/visuals.py` | Artwork, portraits, cutaway planning, and overlay cards |
| `soundtrack/video.py` | FFmpeg filter graphs for the main video and the Short |
| `soundtrack/metadata.py` | Chapters, YouTube title, description, tags, and manifest |
| `tests/` | Unit tests and an end-to-end render with synthetic audio (`python -m pip install pytest && python -m pytest`) |

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
