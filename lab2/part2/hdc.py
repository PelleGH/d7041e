import numpy as np
from dataset import calculate_alphabet
from tqdm import tqdm


def create_character_vectors(dimension, seed=67):
    rng = np.random.default_rng(seed)

    alphabet = calculate_alphabet(seed=seed)
    # alphabet = "abcdefghijklmnopqrstuvwxyz "
    vectors = {}

    for char in alphabet:
        vectors[char] = rng.choice([-1, 1], size=dimension)

    return vectors

# def get_vec(character_vectors, c):
#     if c not in alphabet:
#         # return zero-vector
#         dim = len(character_vectors[0])
#         pass
#     return character_vectors[c]
def get_vec(character_vectors, c, fill=0):
    v = character_vectors.get(c)
    if v is None:
        dim = len(next(iter(character_vectors.values())))
        return np.full(dim, fill, dtype=np.int64)    # fill=0 drops the tri-gram, fill=1 is the identity
    return v

def encode_trigram(trigram, character_vectors):
    a = np.roll(get_vec(character_vectors, trigram[0]), 1)
    b = np.roll(get_vec(character_vectors, trigram[1]), 2)
    c = np.roll(get_vec(character_vectors, trigram[2]), 3)

    return a * b * c

def encode_text(text, character_vectors, trigram_cache=None):
    if trigram_cache is None:
        trigram_cache = {}

    dimension = len(next(iter(character_vectors.values())))
    result = np.zeros(dimension, dtype=np.int64)

    counts = Counter(text[i:i+3] for i in range(len(text) - 2))

    for trigram, frequency in counts.items():
        if trigram not in trigram_cache:
            trigram_cache[trigram] = encode_trigram(
                trigram, character_vectors
            )

        result += frequency * trigram_cache[trigram]

    return result

from collections import Counter, defaultdict


def build_language_centroids(training_data, character_vectors):
    dimension = len(next(iter(character_vectors.values())))

    # Count occurrences of trigrams in each language
    trigram_counts = defaultdict(Counter)

    for language, text in tqdm(training_data, desc="Tri-grams", unit="sent", leave=False):
        trigrams = (text[i:i+3] for i in range(len(text) - 2))
        trigram_counts[language].update(trigrams)

    # Save calculated trigram vectors to avoid recalculation
    trigram_cache = {}
    centroids = {}

    for language, counts in trigram_counts.items():
        centroid = np.zeros(dimension, dtype=np.int64)

        for trigram, frequency in counts.items():
            if trigram not in trigram_cache:
                trigram_cache[trigram] = encode_trigram(
                    trigram, character_vectors
                )

            centroid += frequency * trigram_cache[trigram]

        centroids[language] = centroid
        # print(f"Finished centroid: {language}", flush=True)

    return centroids



# mening -> query vector