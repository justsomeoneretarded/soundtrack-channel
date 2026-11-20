"""Audio analysis, arrangement, and loudness measurement."""
from dataclasses import dataclass
import json
import math
import re
import subprocess
import sys

import numpy as np

from .common import FFMPEG, RATE

# Intensity keyframes for the mix: opening, ascent, release, gathering, peak, resolution.
ARC = [0.18, 0.56, 0.78, 0.38, 0.58, 0.94, 0.25]
MOVEMENTS = ["Opening", "Ascent", "Reverie", "Gathering", "Summit", "Resolution"]
LOUDNESS = {"I": -14.0, "TP": -1.5, "LRA": 11.0}


@dataclass
class Passage:
    source: int
    start: int
    end: int
    energy: float
    brightness: float = 0.0
    level: float = 0.0
    ceiling: float = 0.0


def brightness(samples):
    """Log2 spectral centroid of a short excerpt, used to avoid jarring timbre changes."""
    block = 2048
    mono = np.asarray(samples, dtype=np.float32).mean(axis=1)
    usable = len(mono) // block * block
    if usable < block:
        return 0.0
    frames = mono[:usable].reshape(-1, block) * np.hanning(block).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames, axis=1)).mean(axis=0)
    total = float(spectrum.sum())
    if total <= 1e-9:
        return 0.0
    freqs = np.fft.rfftfreq(block, 1 / RATE)
    return math.log2(1 + float((freqs * spectrum).sum()) / total)


def prepare(files, credits, scratch, style="mix"):
    sources, passages, usable_credits = [], [], []
    for path, credit in zip(files, credits):
        raw = scratch / f"source-{len(sources)}.pcm"
        decoded = subprocess.run(FFMPEG + ["-i", str(path), "-map", "0:a:0", "-t", "1800",
                                          "-ac", "2", "-ar", str(RATE), "-f", "f32le", str(raw)])
        if decoded.returncode or not raw.exists() or raw.stat().st_size < RATE * 8 * 2:
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
            print(f"Skipping near-silent audio from {credit[0]!r}.", file=sys.stderr)
            continue
        first, last = int(active[0]), int(active[-1]) + 1
        source = len(sources)
        sources.append(samples)
        usable_credits.append(credit)
        source_level = float(np.percentile(envelope[first:last], 65))
        source_peak = float(np.max(np.abs(samples[first * frames:last * frames])))
        # Cut at quiet points close to the target length to avoid cutting through accents.
        # Envelope windows are quarter seconds, so lengths below are in seconds * 4.
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
            middle = (start + end) // 2 * frames
            excerpt = samples[max(start * frames, middle - 3 * RATE // 2):min(end * frames, middle + 3 * RATE // 2)]
            album = style == "album"
            passages.append(Passage(
                source, start * frames, end * frames, energy, brightness(excerpt),
                level=source_level if album else energy, ceiling=source_peak if album else 0.0,
            ))
            start = end
    if not passages:
        raise SystemExit("No usable audio passages were found.")
    return sources, passages, usable_credits


def _follows(passages, previous, index):
    if previous is None:
        return False
    before, after = passages[previous], passages[index]
    return before.source == after.source and before.end == after.start


def write_mix(path, sources, passages, secs, rng, style="mix"):
    """Write an arranged float32 stereo stream and return [(start_seconds, source)].

    "mix" picks passages along an intensity arc, favouring natural continuations and similar
    timbre. "album" plays each source in order with crossfades between them.
    """
    goal = secs * RATE
    position, previous = 0, None
    last_index, last_gain, run = None, None, 0
    recent, timeline = [], []
    energies = [passage.energy for passage in passages]
    low, high = np.percentile(energies, [15, 85])
    floor = max(float(high - low), 0.001)
    with path.open("wb") as output:
        while position < goal:
            if style == "album":
                index = 0 if last_index is None else (last_index + 1) % len(passages)
            else:
                target = float(np.interp(position / goal, np.linspace(0, 1, len(ARC)), ARC))
                pool = [i for i in range(len(passages)) if i not in recent] or list(range(len(passages)))

                def cost(i):
                    passage = passages[i]
                    energy = np.clip((passage.energy - low) / floor, 0, 1)
                    score = abs(energy - target) + rng.uniform(0, 0.45)
                    if last_index is not None:
                        last = passages[last_index]
                        if _follows(passages, last_index, i):
                            score += -0.12 if run < 3 else 0.1
                        elif passage.source == last.source:
                            score += 0.22
                        score += 0.25 * min(1.0, abs(passage.brightness - last.brightness) / 2)
                    return score

                index = min(pool, key=cost)
            passage = passages[index]
            follows = _follows(passages, last_index, index)
            run = run + 1 if follows else 0
            recent = (recent + [index])[-min(8, max(1, len(passages) - 1)):]
            clip = np.array(sources[passage.source][passage.start:passage.end], dtype=np.float32)
            peak = passage.ceiling or float(np.max(np.abs(clip)))
            gain = min((0.09 / max(passage.level, 0.005)) ** 0.65, 3.0, 0.85 / max(peak, 0.001))
            if follows and last_gain is not None and abs(gain - last_gain) > 1e-4:
                ramp = min(len(clip), RATE)
                clip[:ramp] *= np.linspace(last_gain, gain, ramp, dtype=np.float32)[:, None]
                clip[ramp:] *= gain
            else:
                clip *= gain
            last_index, last_gain = index, gain
            if previous is None:
                start_at = position
                fade = min(len(clip), RATE // 5)
                clip[:fade] *= np.linspace(0.1, 1, fade, dtype=np.float32)[:, None]
            elif follows:
                start_at = position + len(previous)
                clip = np.concatenate((previous, clip))
            else:
                overlap = min(len(clip) // 4, int(rng.uniform(1.4, 3.8) * RATE), len(previous))
                start_at = position + len(previous) - overlap
                theta = np.linspace(0, math.pi / 2, overlap, dtype=np.float32)[:, None]
                blend = previous[len(previous) - overlap:] * np.cos(theta) + clip[:overlap] * np.sin(theta)
                clip = np.concatenate((previous[:len(previous) - overlap], blend, clip[overlap:]))
            if start_at < goal and (not timeline or timeline[-1][1] != passage.source):
                timeline.append((start_at / RATE, passage.source))
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
    print(f"Arranged {position / RATE:.3f}s from {len(passages)} passages "
          f"({style}, {len(timeline)} cue changes).", flush=True)
    return timeline


def measure_loudness(pcm):
    """First loudnorm pass over the raw mix. Returns measured values or None."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info", "-f", "f32le", "-ar", str(RATE),
         "-ac", "2", "-i", str(pcm), "-af",
         "loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json".format(**LOUDNESS), "-f", "null", "-"],
        capture_output=True, text=True,
    )
    found = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr)
    if result.returncode or not found:
        print("Loudness measurement failed; the mix will keep its arranged level.", file=sys.stderr)
        return None
    measured = json.loads(found[-1])
    try:
        values = {key: float(measured[key]) for key in
                  ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}
    except (KeyError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values.values()):
        return None
    print(f"Measured {values['input_i']:.1f} LUFS, {values['input_tp']:.1f} dBTP; "
          f"normalising to {LOUDNESS['I']:.0f} LUFS.", flush=True)
    return values


def loudness_gain(measured, max_limiting=3.0):
    """Gain in dB toward the loudness target, allowing at most a few dB of peak limiting."""
    if not measured:
        return 0.0
    wanted = LOUDNESS["I"] - measured["input_i"]
    return round(min(wanted, LOUDNESS["TP"] - measured["input_tp"] + max_limiting), 2)


def loudness_filter(measured):
    """A constant gain plus a safety limiter. Unlike dynamic loudnorm, it keeps the score's dynamics."""
    if not measured:
        return "anull"
    gain = loudness_gain(measured)
    chain = f"volume={gain}dB"
    if measured["input_tp"] + gain > LOUDNESS["TP"] - 0.5:
        ceiling = 10 ** ((LOUDNESS["TP"] - 0.5) / 20)
        chain += f",alimiter=limit={ceiling:.4f}:attack=5:release=80:level=false"
    return chain


def loudest_window(pcm, total_seconds, length):
    """Start time of the most intense stretch, avoiding the opening and the final fade."""
    samples = np.memmap(pcm, dtype="<f4", mode="r").reshape(-1, 2)
    seconds = len(samples) // RATE
    if seconds <= length + 2:
        return 0.0
    probe = samples[:seconds * RATE:48].reshape(seconds, -1, 2)
    rms = np.sqrt(np.mean(np.square(probe), axis=(1, 2)))
    window = np.convolve(rms, np.ones(length) / length, mode="valid")
    lo = min(int(total_seconds * 0.05), len(window) - 1)
    hi = max(lo + 1, min(len(window), int(total_seconds - length - 6)))
    return float(lo + int(np.argmax(window[lo:hi])))
