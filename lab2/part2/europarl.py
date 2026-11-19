import os
import random
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

ALLOWED_SCRIPTS = ("LATIN", "CYRILLIC", "GREEK")

@lru_cache(maxsize=None)
def is_allowed(c: str) -> bool:
    if c == " ":
        return True
    try:
        return unicodedata.name(c).startswith(ALLOWED_SCRIPTS)
    except ValueError:                      # unnamed character
        return False


def validate_downloads():
    path = Path(__file__).parent / "data" / "europarl"
    return path.exists()

def preprocess_data(sentences: list[str]):
    processed = []
    for sentence in sentences:
        # Normalize superscripts etc
        s = unicodedata.normalize("NFKC", sentence).lower()

        # Replacement char or Latin-1-as-UTF-8 mojibake ("Ã©") -> drop the sentence
        if "\ufffd" in s or "ã" in s or "â" in s and "€" in s:
            continue

        # Trim whitespace
        s = sentence.strip()

        # To lowercase
        s = s.lower()

        # Remove punctuation. Unicode regex so Cyrillic and other non-latin languages keeps their letters.
        s = re.sub(r"[^\w\s]|\d|_", "", s)

        # Turn multi-spaces into single space
        s = re.sub(r"\s+", " ", s)

        # Trim whitespace again
        s = s.strip()

        # Skip empty strings
        if len(s) == 0:
            continue

        # Removed letters from non LATIN, GREEK or CYRILLIC (bulgarian)
        if not all(is_allowed(c) for c in s):
            continue

        processed.append(s)
    return processed

def load_file(path: Path):
    with open(path, encoding="utf-8") as f:
        lines = f.read()
    return lines.split("\n")

def list_europarl():
    path = Path(__file__).parent / "data" / "europarl"

    files = sorted(path.glob("*.txt"))

    return [(f.absolute(), f.stem) for f in files]

cache = None
def get_europarl():
    if not validate_downloads():
        raise RuntimeError("You haven't downloaded the files yet. Run download_wortschatz.py first.")

    global cache

    if cache is not None:
        return cache

    data: list[tuple[str, str]] = []
    langs = list_europarl()
    for (path, name) in langs:
        sentences = load_file(path)
        sentences = preprocess_data(sentences)
        for s in sentences:
            data.append((name, s))
    cache = data
    return data