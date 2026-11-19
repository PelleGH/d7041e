# NOTE! THIS FILE WAS ENTIRELY GENERATED WITH CLAUDE CODE TO EASILY FACILITATE DOWNLOADING THE EUROPARL TEST DATA

"""Download the Europarl test data for the 21 languages of Part 2.

Europarl v7 is distributed as one archive per language pair with English
(e.g. de-en.tgz). Each archive contains europarl-v7.<xx>-en.<xx> and
europarl-v7.<xx>-en.en. For every non-English language the <xx> side is used;
for English the .en side of the German pair is used.

The archives are big (100-200 MB), so the script streams them and stops reading
as soon as it has MAX_LINES lines of the wanted file. It then draws SAMPLES
random sentences (the paper used 1000 per language) and stores them, one per
line and not preprocessed, in
    data/europarl/<iso2>.txt

Usage:  python download_europarl.py            # all 21 languages
        python download_europarl.py de fr      # only some languages
"""
import random
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://www.statmt.org/europarl/v7/{pair}.tgz"
OUT_DIR = Path(__file__).parent / "data" / "europarl"

# Same ISO-639-1 codes as in download_wortschatz.py
LANGS = ["bg", "cs", "da", "de", "el", "en", "et", "fi", "fr", "hu", "it",
         "lv", "lt", "nl", "pl", "pt", "ro", "sk", "sl", "es", "sv"]

SAMPLES = 1000          # test sentences per language
MAX_LINES = 100_000     # read at most this many lines of the file, then stop downloading
MIN_CHARS = 20          # skip fragments such as "(Applause)" or "Thank you."
SEED = 67
TIMEOUT = 60


def archive_for(lang):
    # There is no en-en archive, so English comes from the German pair.
    other = "de" if lang == "en" else lang
    return f"{other}-en", f".{lang}"


def read_sentences(lang):
    pair, suffix = archive_for(lang)
    url = BASE_URL.format(pair=pair)
    print(f"[{lang}] streaming {url}")

    lines = []
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        # "r|gz" = streaming mode, members must be read in order
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                if not (member.isfile() and member.name.endswith(suffix)):
                    continue
                # TextIOWrapper needs a seekable stream, which a streaming tar is not,
                # so iterate the binary lines and decode them ourselves.
                raw_lines = tar.extractfile(member)
                for i, raw in enumerate(raw_lines):
                    if i >= MAX_LINES:
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if len(line) >= MIN_CHARS:
                        lines.append(line)
                break
    return lines


def download(lang):
    out_file = OUT_DIR / f"{lang}.txt"
    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"[{lang}] already downloaded, skipping")
        return True

    lines = read_sentences(lang)
    if len(lines) < SAMPLES:
        print(f"[{lang}] only {len(lines)} usable sentences, expected {SAMPLES}", file=sys.stderr)
        return False

    sample = random.Random(SEED).sample(lines, SAMPLES)
    out_file.write_text("\n".join(sample) + "\n", encoding="utf-8")
    print(f"[{lang}] {len(sample)} sentences -> {out_file}")
    return True


def main():
    wanted = sys.argv[1:] or LANGS
    unknown = [w for w in wanted if w not in LANGS]
    if unknown:
        sys.exit(f"unknown language code(s): {unknown}; choose from {LANGS}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for lang in wanted:
        try:
            if not download(lang):
                failed.append(lang)
        except Exception as e:           # network error, missing archive, bad tar, ...
            print(f"[{lang}] failed: {e}", file=sys.stderr)
            failed.append(lang)
    if failed:
        sys.exit(f"failed: {failed}")
    print("done")


if __name__ == "__main__":
    main()
