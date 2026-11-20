"""Chapters, YouTube copy, MP4 chapter metadata, and the render manifest."""
import json
import re

from .audio import MOVEMENTS
from .common import YEAR, length_label, timestamp

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
MIN_CHAPTER = 10


def chapters(style, timeline, labels, seconds):
    """YouTube-valid chapters: first at 0:00, at least three, each at least ten seconds."""
    if style == "album":
        entries = []
        for start, source in timeline:
            label = labels[source]
            if entries and entries[-1][1] == label:
                continue
            if entries and start - entries[-1][0] < MIN_CHAPTER:
                entries[-1] = (entries[-1][0], label)
                continue
            entries.append((0.0 if not entries else start, label))
        while len(entries) > 1 and seconds - entries[-1][0] < MIN_CHAPTER:
            entries.pop()
        # Number repeats so a looping tracklist stays readable.
        counts, numbered = {}, []
        for start, label in entries:
            counts[label] = counts.get(label, 0) + 1
            numbered.append((start, label if counts[label] == 1 else f"{label} (reprise)"))
        entries = numbered
    else:
        step = seconds / len(MOVEMENTS)
        entries = [(round(index * step), f"{ROMAN[index]}. {name}") for index, name in enumerate(MOVEMENTS)]
    if len(entries) < 3 or any(b[0] - a[0] < MIN_CHAPTER for a, b in zip(entries, entries[1:])):
        return []
    return entries


def featured(timeline, labels):
    first = {}
    for start, source in timeline:
        first.setdefault(labels[source], start)
    return sorted(((start, label) for label, start in first.items()), key=lambda item: item[0])


def video_title(title, seconds, style, composer=""):
    length = length_label(seconds)
    kind = "Full Score" if style == "album" else "Cinematic Score"
    options = []
    if composer:
        options.append(f" Soundtrack | {composer} | {length} {kind}")
    options.append(f" Soundtrack Mix | {length} {kind}" if style == "mix" else f" Soundtrack | {length} {kind}")
    for suffix in options:
        if len(title) + len(suffix) <= 100:
            return title + suffix
    return title[:100 - len(options[-1])].rstrip() + options[-1]


def hashtag(text):
    words = [word for word in re.findall(r"[^\W_]+", text) if not YEAR.fullmatch(word)]
    tag = "".join(word[:1].upper() + word[1:] for word in words)
    return f"#{tag}" if tag and len(tag) <= 30 else ""


def tags(title, composer, labels, seconds):
    bare = " ".join(word for word in title.split() if not YEAR.fullmatch(word.strip("()[]")))
    bare = re.sub(r"[()\[\]]", "", bare).strip() or title
    length = length_label(seconds).lower().replace("-second preview", " second")
    values = [bare, f"{bare} soundtrack", f"{bare} ost", f"{bare} score", f"{bare} music",
              f"{bare} theme"]
    if composer:
        values += [composer, f"{composer} music", f"{composer} soundtrack"]
    values += [f"{length} soundtrack", "film score", "cinematic music", "soundtrack mix",
               "movie soundtrack", "epic music", "study music"]
    values += labels
    result, seen, total = [], set(), 0
    for value in values:
        value = re.sub(r"[<>,]", "", value).strip()
        key = value.casefold()
        if not value or key in seen or total + len(value) + 1 > 480:
            continue
        seen.add(key)
        result.append(value)
        total += len(value) + 1
    return result


def describe(title, seconds, style, composer, chapter_list, cues, credits, visual_credits):
    noun = "mix" if style == "mix" else "journey"
    if seconds % 60 == 0 and seconds >= 600:
        span = "one-hour" if seconds == 3600 else length_label(seconds).lower()
        opening = f"Listen to a continuous {span} {title} soundtrack {noun}."
    else:
        opening = f"Preview the continuous {title} soundtrack {noun} in {seconds} seconds."
    if composer:
        opening += f" Music by {composer}."
    if style == "album":
        shape = ("Each cue plays in full and flows into the next with gentle crossfades, "
                 "paired with an eclipse nightscape and audio-reactive visualizer.")
    else:
        shape = ("The edit moves through quieter passages, fuller arrangements and a final resolution, "
                 "paired with an eclipse nightscape and audio-reactive visualizer.")
    if visual_credits:
        shape += " Selected scene cutaways are paired with the score."
    blocks = [f"{opening} {shape}"]
    if chapter_list:
        heading = "Tracklist" if style == "album" else "Chapters"
        blocks.append(heading + "\n" + "\n".join(f"{timestamp(start)} {label}" for start, label in chapter_list))
    if style == "mix" and cues:
        blocks.append("Featured cues\n" + "\n".join(f"{timestamp(start)} {label}" for start, label in cues))
    blocks.append("Audio sources\n" + "\n".join(f"{name} — {link}" if link else name for name, link in credits))
    if visual_credits:
        blocks.append("Visual sources\n" + "\n".join(
            f"{name} — {link}" if link else name for name, link in visual_credits))
    blocks.append("Independent mix. All music and footage belong to their respective owners.")
    hashtags = [tag for tag in (hashtag(title), hashtag(composer) if composer else "") if tag]
    blocks.append(" ".join(hashtags + ["#Soundtrack", "#FilmScore", "#CinematicMusic"]))
    return "\n\n".join(blocks) + "\n"


def _escape(value):
    return re.sub(r"([=;#\\\n])", r"\\\1", value)


def write_ffmetadata(path, title, chapter_list, seconds):
    lines = [";FFMETADATA1", f"title={_escape(title)}"]
    for index, (start, label) in enumerate(chapter_list):
        end = chapter_list[index + 1][0] if index + 1 < len(chapter_list) else seconds
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(start * 1000)}",
                  f"END={int(end * 1000)}", f"title={_escape(label)}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_outputs(out, *, title, seconds, style, composer, timeline, labels, chapter_list,
                  credits, visual_credits, loudness, scenes, short, characters=(), people=0):
    cues = featured(timeline, labels)
    heading = video_title(title, seconds, style, composer)
    description = describe(title, seconds, style, composer, chapter_list, cues, credits, visual_credits)
    keywords = tags(title, composer, labels, seconds)
    (out / "youtube-title.txt").write_text(heading + "\n", encoding="utf-8")
    (out / "youtube-description.txt").write_text(description, encoding="utf-8")
    (out / "youtube-tags.txt").write_text(", ".join(keywords) + "\n", encoding="utf-8")
    manifest = {
        "title": title,
        "composer": composer or None,
        "style": style,
        "seconds": seconds,
        "youtube_title": heading,
        "chapters": [{"start": start, "timestamp": timestamp(start), "title": label}
                     for start, label in chapter_list],
        "timeline": [{"start": round(start, 3), "source": source, "cue": labels[source]}
                     for start, source in timeline],
        "audio_sources": [{"name": name, "url": link or None, "cue": label}
                          for (name, link), label in zip(credits, labels)],
        "visual_sources": [{"name": name, "url": link or None} for name, link in visual_credits],
        "character_names_searched": list(characters),
        "recurring_characters_found": people,
        "scene_cutaways": scenes,
        "loudness": loudness,
        "short": short,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")

