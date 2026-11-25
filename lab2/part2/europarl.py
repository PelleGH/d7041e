import os
import random
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from dataset import preprocess_data


def validate_downloads():
    path = Path(__file__).parent / "data" / "europarl"
    return path.exists()

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