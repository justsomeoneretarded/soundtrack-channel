"""Turn the YOUTUBE_COOKIES secret into a Netscape cookies.txt that yt-dlp accepts.

Accepts a Netscape cookies.txt, the same file base64-encoded, or the JSON array that browser
cookie extensions export. Prints a diagnosis but never the cookie values.

Usage: YOUTUBE_COOKIES=... python -m soundtrack.cookies OUTPUT_PATH
"""
import base64
import binascii
import json
import os
from pathlib import Path
import sys

HEADER = "# Netscape HTTP Cookie File"
SIGN_IN = {"SAPISID", "__Secure-3PAPISID", "__Secure-1PSID", "__Secure-3PSID", "LOGIN_INFO", "SID"}


def _from_json(text):
    items = json.loads(text)
    if isinstance(items, dict):
        items = items.get("cookies", [])
    lines = []
    for item in items:
        domain = item.get("domain", "")
        if not domain or "name" not in item:
            continue
        expiry = item.get("expirationDate") or item.get("expires") or 0
        lines.append("\t".join([
            domain,
            "TRUE" if domain.startswith(".") else "FALSE",
            item.get("path", "/"),
            "TRUE" if item.get("secure") else "FALSE",
            str(int(float(expiry))) if expiry and float(expiry) > 0 else "0",
            item["name"],
            str(item.get("value", "")),
        ]))
    return "\n".join(lines)


def normalise(secret):
    """Return (cookies.txt text, youtube cookie names)."""
    text = secret.strip().replace("\r\n", "\n")
    if text and len(text.split()) == 1 and not text.startswith(("[", "{")):
        try:
            text = base64.b64decode(text, validate=False).decode("utf-8").strip().replace("\r\n", "\n")
        except (binascii.Error, UnicodeDecodeError):
            pass
    if text.startswith(("[", "{")):
        text = _from_json(text)
    # Pasting into a form can turn tabs into spaces; cookie fields never contain whitespace.
    rows = [line.split("\t") if "\t" in line else line.split(None, 6) for line in text.splitlines()
            if line.strip() and (not line.startswith("#") or line.startswith("#HttpOnly_"))]
    youtube = [row for row in rows if len(row) == 7 and row[0].removeprefix("#HttpOnly_").lstrip(".").endswith(
        ("youtube.com", "google.com"))]
    names = {row[5] for row in youtube}
    body = "\n".join("\t".join(row) for row in rows if len(row) == 7)
    return f"{HEADER}\n{body}\n", names


def main():
    secret = os.environ.get("YOUTUBE_COOKIES", "")
    if not secret.strip():
        print("::warning::No YOUTUBE_COOKIES secret is set; YouTube may block downloads from GitHub runners.")
        return 0
    try:
        text, names = normalise(secret)
    except (ValueError, TypeError, AttributeError) as error:
        print(f"::error::YOUTUBE_COOKIES could not be read ({type(error).__name__}). "
              "Export youtube.com cookies as cookies.txt (see README).")
        return 1
    if not names:
        print("::error::YOUTUBE_COOKIES has no youtube.com cookies. Export them from youtube.com "
              "while signed in (see README).")
        return 1
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch(mode=0o600)
    output.write_text(text, encoding="utf-8")
    signed_in = names & SIGN_IN
    print(f"Prepared {len(names)} YouTube cookies; sign-in cookies "
          f"{'present' if signed_in else 'missing'}.")
    if not signed_in:
        print("::warning::The cookies do not include sign-in cookies. Export them while signed in to YouTube.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
