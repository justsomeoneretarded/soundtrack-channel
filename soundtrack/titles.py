"""YouTube title options in a second-person, late-night-ambition voice.

Lines are assembled from small parts (a moment, a pressure, a payoff) and filled with the
work's name and main character, so each film gets its own set rather than a fixed list.
"""
import random

from .thumbnails import short_name

MAX_LENGTH = 100
POPULAR_SITELINKS = 35  # Wikipedia language editions; blockbusters and hit series have 60+.

NIGHTS = ["Friday night", "Saturday night", "a Sunday night", "a holiday weekend"]
HOURS = ["2am", "3am", "midnight", "1am", "4am"]
LEFT = ["you have 1 hour left", "you have 2 hours left", "the deadline is tonight",
        "it's due at sunrise", "you have one shot at this"]
PROJECTS = ["your startup", "your side project", "your first company", "the app nobody believes in",
            "your empire", "the thing that changes everything"]
MOMENTS = [
    "you are working on {night} while everybody is out",
    "it's {hour} and you're still building",
    "{left}, and you're calm",
    "while they sleep, you're building {project}",
    "you are building {project} alone at {hour}",
    "you chose the work over the party",
    "you're not tired, you're locked in",
    "you are working under pressure",
    "nobody sees the late nights, they'll see the results",
    "{hour} focus: building what others can't imagine yet",
    "the quiet hours where you change your life",
    "you can either accept it, or you can get to work",
    "you are becoming who you said you'd be",
    "think different, work different, win different",
    "you are limitless tonight",
    "after hours ambition: one more build before sunrise",
    "you're trusting the process at {hour}",
    "deep work while the city sleeps",
]
WITH_CHARACTER = [
    "work like {character} is watching",
    "focus like {character}",
    "you are {character} and the clock is running",
    "{character} energy for deep work",
    "lock in like {character}",
]
WITH_WORK = [
    "{work} main theme, but you're building your empire",
    "{work} music for late night ambition",
    "{work} music for when you have to deliver",
]
SUFFIXES = ["| {work} Soundtrack", "| {work} Study Music", "| {work} Work Music",
            "| FOCUS MUSIC", "| STUDY MUSIC", "| WORK MUSIC"]


def _sentence(text):
    return text[:1].upper() + text[1:]


def title_options(title, character="", popular=False, seed=0, count=5):
    """`count` distinct titles. Half (rounded up) end with a '| … MUSIC' or '| … Soundtrack' tag.

    "<work> music to work with <character>" leads the list when the work is popular, and
    sits further down otherwise.
    """
    rng = random.Random(seed)
    work = short_name(title)
    fill = {"work": work, "character": character, "night": "", "hour": "", "left": "", "project": ""}

    def render(template):
        values = dict(fill, night=rng.choice(NIGHTS), hour=rng.choice(HOURS), left=rng.choice(LEFT),
                      project=rng.choice(PROJECTS))
        return _sentence(template.format(**values))

    signature = f"{work} music to work with {character}" if character else ""
    pool = list(MOMENTS)
    rng.shuffle(pool)
    extras = (rng.sample(WITH_CHARACTER, 2) if character else []) + rng.sample(WITH_WORK, 1)
    bases = [render(template) for template in extras + pool]
    bases = [base for base in dict.fromkeys(bases) if len(base) <= MAX_LENGTH]
    # Interleave so character and work lines are spread through the list.
    picked = [bases[3], bases[0], bases[4], bases[1], bases[5], bases[2], *bases[6:]][:count]
    if signature:
        picked = [signature] + picked[:count - 1] if popular else picked[:3] + [signature] + picked[3:count - 1]
    picked = picked[:count]

    # Tag half of the titles (rounded up), never the signature line (it already names the work).
    want = (count + 1) // 2
    order = [index for index in range(len(picked)) if picked[index] != signature]
    rng.shuffle(order)
    suffixes = list(SUFFIXES)
    rng.shuffle(suffixes)
    tagged = 0
    for index in order:
        if tagged == want:
            break
        for offset in range(len(suffixes)):
            suffix = suffixes[(tagged + offset) % len(suffixes)].format(work=work)
            candidate = f"{picked[index]} {suffix}"
            if len(candidate) <= MAX_LENGTH:
                picked[index] = candidate
                tagged += 1
                break
    return picked
