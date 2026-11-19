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
    path = Path(__file__).parent / "data" / "wortschatz"
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

def list_wortschatz():
    path = Path(__file__).parent / "data" / "wortschatz"

    files = sorted(path.glob("*.txt"))

    return [(f.absolute(), f.stem) for f in files]

def get_wortschatz():
    if not validate_downloads():
        raise RuntimeError("You haven't downloaded the files yet. Run download_wortschatz.py first.")

    data: list[tuple[str, list[str]]] = []
    langs = list_wortschatz()
    for (path, name) in langs:
        sentences = load_file(path)
        sentences = preprocess_data(sentences)
        data.append((name, sentences))
    return data


cache = None

def get_train_test_data(splits: tuple[float, float, float] = (0.7, 0.15, 0.15), seed = 67):
    global cache
    if cache is not None:
        return cache

    languages = get_wortschatz()

    rand = random.Random(seed)

    train: list[tuple[str, str]] = []
    val: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    for (lang, sentences) in languages:
        for s in sentences:
            x = rand.random()
            if x < splits[0]:
                train.append((lang, s))
            elif x < splits[0] + splits[1]:
                val.append((lang, s))
            else:
                test.append((lang, s))

    cache = (train, val, test)
    return (train, val, test)

def calculate_alphabet(seed=67):
    sentences = get_train_test_data(seed=seed)[0]
    letters = {}
    for (_, s) in sentences:
        for c in s:
            letters[c] = 1

    s = ""
    for x in sorted(letters.keys()):
        s += x
    return s



if __name__ == "__main__":
    print(calculate_alphabet())
