from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from hdc import encode_text
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from tqdm import tqdm


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def test_model(centroids, test_data: list[tuple[str, str]], character_vectors):
    actual = []
    predicted = []

    success = 0
    total = 0

    trigram_cache = {}

    labels = list(centroids.keys())
    centroid_matrix = np.array(
        [centroids[lang] for lang in labels],
        dtype=np.float64
    )

    norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
    centroid_matrix /= np.maximum(norms, 1e-12)

    for (lang, sentence) in tqdm(test_data, leave=False, desc="Testing", unit="sent"):
        query_vector = encode_text(
            sentence, character_vectors, trigram_cache
        ).astype(np.float64)

        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            best_language = "unknown"
        else:
            similarities = centroid_matrix @ (query_vector / query_norm)
            best_language = labels[np.argmax(similarities)]

        total += 1

        if best_language == lang:
            success += 1

        actual.append(lang)
        predicted.append(best_language)
    f1 = f1_score(actual, predicted, average="macro")

    labels = sorted(centroids.keys())
    cm = confusion_matrix(actual, predicted, labels=labels)

    accuracy = success / max(1, total)

    return accuracy, f1, cm,