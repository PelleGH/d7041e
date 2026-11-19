# NOTE! THIS FILE WAS ENTIRELY GENERATED WITH CLAUDE CODE TO EASILY FACILIATE DOWNLOADING ALL 21 LANGUAGES

"""Download the Leipzig Wortschatz news corpora for the 21 languages of Part 2.

For every language it tries a list of (year, size) candidates, downloads the
first archive that exists, extracts only the "*-sentences.txt" file, removes
the leading sentence-id column and stores one sentence per line in
    data/wortschatz/<iso2>.txt

Usage:  python download_wortschatz.py            # all 21 languages
        python download_wortschatz.py de fr      # only some languages
"""
import io
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://downloads.wortschatz-leipzig.de/corpora/{code}_news_{year}_{size}.tar.gz"
OUT_DIR = Path(__file__).parent / "data" / "wortschatz"

# ISO-639-1 (used in the output file names, same as Europarl) -> ISO-639-3 (used by Wortschatz)
LANGS = {
    "bg": "bul", "cs": "ces", "da": "dan", "de": "deu", "el": "ell", "en": "eng",
    "et": "est", "fi": "fin", "fr": "fra", "hu": "hun", "it": "ita", "lv": "lav",
    "lt": "lit", "nl": "nld", "pl": "pol", "pt": "por", "ro": "ron", "sk": "slk",
    "sl": "slv", "es": "spa", "sv": "swe",
}

# Tried in this order. Paper used ~1M characters / language (about 10K sentences),
# 30K gives enough text for a train/validation/test split.
YEARS = [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016]
SIZES = ["30K", "100K", "10K"]

TIMEOUT = 60


def url_exists(url):
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return False


def find_url(code):
    for size in SIZES:
        for year in YEARS:
            url = BASE_URL.format(code=code, year=year, size=size)
            if url_exists(url):
                return url
    return None


def extract_sentences(archive_bytes):
    """Return the sentences from the *-sentences.txt member, without ids."""
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("-sentences.txt"))
        text = tar.extractfile(member).read().decode("utf-8")
    sentences = []
    for line in text.splitlines():
        parts = line.split("\t", 1)          # "<id>\t<sentence>"
        if len(parts) == 2 and parts[1].strip():
            sentences.append(parts[1].strip())
    return sentences


def download(iso2):
    out_file = OUT_DIR / f"{iso2}.txt"
    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"[{iso2}] already downloaded, skipping")
        return True

    url = find_url(LANGS[iso2])
    if url is None:
        print(f"[{iso2}] no news corpus found for {LANGS[iso2]}", file=sys.stderr)
        return False

    print(f"[{iso2}] downloading {url}")
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        data = r.read()
    sentences = extract_sentences(data)
    out_file.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    print(f"[{iso2}] {len(sentences)} sentences -> {out_file}")
    return True


def main():
    wanted = sys.argv[1:] or list(LANGS)
    unknown = [w for w in wanted if w not in LANGS]
    if unknown:
        sys.exit(f"unknown language code(s): {unknown}; choose from {list(LANGS)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = [l for l in wanted if not download(l)]
    if failed:
        sys.exit(f"failed: {failed}")
    print("done")


if __name__ == "__main__":
    main()
