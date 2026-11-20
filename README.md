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
   | `minutes` | `60` | Video length, 1–180. An hour takes roughly 12 minutes to render on a standard runner, plus download and analysis time. |
   | `audio_urls` | blank | Audio links separated by spaces: direct files, Google Drive or Dropbox share links, a GitHub Release asset, or any site yt-dlp supports. When set, the YouTube audio search is skipped. |
   | `characters` | blank | Comma-separated character names to search scenes for. Blank looks them up on Wikidata. |

3. Download the **soundtrack-video** artifact from the completed run.

The artifact contains:

| File | Purpose |
| --- | --- |
| `soundtrack.mp4` | 1280×720, 24 fps, H.264 video and stereo AAC audio at −14 LUFS, exact requested length, with embedded chapters |
| `short.mp4` | 1080×1920 Shorts teaser (45 s) from the most intense stretch of the mix, showing the three best scenes |
| `thumbnail.jpg` | 1280×720 eclipse art with a length badge and the lead character's portrait when footage is available |
| `youtube-title.txt` | Search-friendly title with the work name, composer, soundtrack, and duration (≤100 characters) |
| `youtube-description.txt` | Description with YouTube chapters or a tracklist, featured cues, source credits, links, and hashtags |
| `youtube-tags.txt` | Comma-separated tags within YouTube's 500-character limit |
| `manifest.json` | Machine-readable record of the arrangement, chapters, sources, loudness, characters, and chosen scenes |

The workflow uses the standard Ubuntu runner with no paid APIs and no repository secrets. GitHub offers standard runners free for public repositories. Private repositories consume their included Actions minutes and artifact storage. Artifacts expire after two days.

## How it works

The workflow searches YouTube for the title plus “official soundtrack” and “original score” (and the composer, when given). It scores the results for title and year matches and score-related terms, filters common false matches such as covers, piano tutorials, sped-up edits, and lo-fi remixes, and attempts to download audio from the strongest candidates. Search and upload labels are heuristics. Check the credited sources before publishing, and use audio you have permission to download, edit, and publish. Credits alone do not grant those rights. Content ID may claim or block an upload.

**Cue names.** Upload titles are reduced to readable cue names for the on-screen card, chapters, and tags. Parts that restate the work, describe the upload (“Official Soundtrack”, “HD”), name the composer, or repeat across several uploads (usually the artist or label) are dropped. *Hans Zimmer - Cornfield Chase (Interstellar Official Soundtrack)* becomes *Cornfield Chase*.

**Arrangement.** Each source is split at quiet points into passages, measured for intensity (RMS) and brightness (spectral centroid). In `mix` style, the arranger follows an intensity curve through six movements — Opening, Ascent, Reverie, Gathering, Summit, Resolution — prefers letting a passage run on into its natural continuation for a few minutes, avoids abrupt jumps in timbre, and overlaps changes of cue with variable-length equal-power crossfades. In `album` style, each cue plays in full at a consistent per-cue level and crossfades into the next; the list repeats if the cues are shorter than the requested length. Very short sources can still sound repetitive and are rejected when there are too few usable passages for a long render.

**Loudness.** The finished mix is measured (EBU R128) and brought to YouTube's −14 LUFS reference with a single constant gain. A transparent safety limiter is added only when peaks would exceed −2 dBFS, and it is never asked for more than 3 dB of reduction, so the score keeps its dynamics.

**Best scenes and characters.** The workflow looks up the work's characters on Wikidata and searches YouTube for official trailers, official clips, scenes featuring each of the first four characters, and best-scene uploads. Interviews, featurettes, reactions, reviews, fan edits, and similar uploads are filtered out. Up to six videos are downloaded at 720p. Each one is decoded once: shot cuts are detected, and two frames a second are checked for exposure, sharpness, and faces. Faces are found with OpenCV's YuNet detector and grouped by identity with the SFace recogniser. Both models are downloaded once from the OpenCV model zoo, pinned to a commit, verified by SHA-256, and cached between runs. The people who appear most often, largest, and across the most videos rank as the main characters.

Every shot is scored by how clearly it shows a main character (face size, sharpness, lighting, and how directly they face the camera, weighted toward the lead), and 5–10-second scenes are built that start and end on shot cuts. Logos, title cards, and end credits are skipped, and rapid montages are marked down. About one scene every seven minutes is chosen, varied across characters and source videos, and the strongest one is placed at the mix's climax.

**Picture.** The eclipse drifts behind a portrait of the lead character while a mint-green waveform responds to the final mix. A “Now playing” card names the current cue, and a progress bar tracks the whole video. Scenes play full-frame and muted with soft dissolves, while the score, waveform, and progress bar continue on top. Halfway through each scene, hidden behind it, the portrait changes to the character that scene featured, so the video cycles through the cast. The Short shows the three strongest scenes in its picture panel. If no footage can be downloaded, the animated eclipse and waveform still render. The description includes separate visual credits when footage is used. Check that you also have permission to reuse the footage before uploading.

**Chapters.** `mix` videos get one chapter per movement. `album` videos get a tracklist with one chapter per cue; cues shorter than ten seconds are merged so YouTube accepts the list. Chapters are written into the description and embedded in the MP4.

## YouTube sign-in on GitHub Actions

YouTube asks for a sign-in before it serves video and audio to data-center addresses such as GitHub's runners. The workflow signs in with a `YOUTUBE_COOKIES` repository secret and starts a [bgutil proof-of-origin token provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider), which helps yt-dlp pass YouTube's bot checks.

1. Use a separate Google account, not your main one. YouTube can flag accounts that download from automation.
2. In a private browser window, sign in to YouTube with that account, then export the youtube.com cookies with a cookies.txt export extension. Close the private window without signing out, because signing out invalidates the cookies. The [yt-dlp guide to exporting YouTube cookies](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies) describes the steps.
3. Save the file's contents as the repository secret `YOUTUBE_COOKIES` (**Settings → Secrets and variables → Actions**). A Netscape cookies.txt, the same file base64-encoded, or a JSON cookie export all work.

The workflow checks the secret without printing it, warns when sign-in cookies are missing, writes it to a private temporary file, and deletes the file after the render. yt-dlp pauses between requests to look less automated. Cookies expire. If a run reports a sign-in or bot check, export fresh cookies and update the secret.

For the most reliable runs, or to use audio you have licensed, pass `audio_urls` instead. Those downloads skip YouTube. If you have a Linux x64 machine with working access to your permitted sources, you can also register it as a [self-hosted runner](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners), give it the label `soundtrack-render`, and set the repository Actions variable `SOUNDTRACK_RUNNER_LABEL` to that label. Use a private repo for self-hosting.

## Run locally

Install Python 3.10+, FFmpeg, and the DejaVu font package. Install Deno for YouTube extraction.

```bash
python -m pip install -r requirements.txt
python render.py --title 'Example Movie (2024)'
python render.py --title 'Example Movie (2024)' --audio cue-1.flac cue-2.mp3 --clips trailer.mp4
python render.py --title 'Example Movie (2024)' --audio cue-1.flac --seconds 20 --draft
python render.py --title 'Example Movie (2024)' --composer 'Jane Doe' --style album --minutes 90
python render.py --title 'Example Movie (2024)' --audio-url https://example.com/score.flac --characters 'Ava, Max'
SOUNDTRACK_COOKIES=cookies.txt python render.py --title 'Example Movie (2024)'
python render.py --help
```

`--draft` renders at 640×360 and 12 fps for quick checks. `--cutaways N` changes the number of scenes (0 disables them), `--short-seconds 0` skips the Shorts teaser, and `--no-loudnorm` keeps the arranged level.

## Project layout

| Path | Contents |
| --- | --- |
| `render.py` | Entry point used by the workflow |
| `soundtrack/sources.py` | YouTube search, audio downloads, links, and cue naming |
| `soundtrack/footage.py` | Character lookup, footage download, face analysis, and scene selection |
| `soundtrack/cookies.py` | Converts the `YOUTUBE_COOKIES` secret into a cookies file |
| `soundtrack/audio.py` | Passage analysis, arrangement, and loudness |
| `soundtrack/visuals.py` | Artwork, portrait layers, overlay cards, and timelines |
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
