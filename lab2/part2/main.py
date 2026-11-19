import matplotlib.pyplot as plt
from dataset import get_train_test_data
from europarl import get_europarl
from hdc import build_language_centroids, create_character_vectors
from infer import test_model
from sklearn.metrics import ConfusionMatrixDisplay


def main():

    seed = 67
    
    print("Getting datasets...")
    (train, _, test) = get_train_test_data(splits=(0.8, 0.0, 0.2), seed=seed)
    europarl = get_europarl()

    best_vectors = None
    best_centroids = None
    best_accuracy = 0
    best_d = 0

    # for d in [int(1e4)]:
    for d in [100, 200, 500, 1000, 1500, 2500, 5000, 10000]:
        print(f"\n === Running with dimensions d={d} ===")

        print("Building character vectors...")
        character_vectors = create_character_vectors(d, seed=seed)

        print("Training centroids...")
        centroids = build_language_centroids(train, character_vectors)

        print("Testing the model...")
        accuracy, f1, cm = test_model(centroids, test, character_vectors)

        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"F1-score: {f1:.4f}")

        labels = sorted(centroids.keys())
        
        fig, ax = plt.subplots(figsize=(14, 12))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels
        ).plot(
            ax=ax,
            xticks_rotation=90,
            values_format="d",              # plain integers, no 1e+03 notation
            text_kw={"fontsize": 7},        # smaller numbers in the cells
            cmap="Blues",
            colorbar=False,
        )
        ax.set_title(f"Confusion matrix with d={d}")
        fig.tight_layout()
        fig.savefig(f"img/cm_{d}.png")

        if accuracy > best_accuracy:
            best_centroids = centroids
            best_vectors = character_vectors
            best_accuracy = accuracy
            best_d = d

    print(f"Best d={best_d} with accuracy={best_accuracy}")

    print("== Testing on Europarl ==")
    accuracy, f1, cm = test_model(best_centroids, europarl, best_vectors)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"F1-score: {f1:.4f}")

    labels = sorted(centroids.keys())
    
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    ).plot(
        ax=ax,
        xticks_rotation=90,
        values_format="d",              # plain integers, no 1e+03 notation
        text_kw={"fontsize": 7},        # smaller numbers in the cells
        cmap="Blues",
        colorbar=False,
    )
    ax.set_title(f"Confusion matrix with d={d}")
    fig.tight_layout()
    fig.savefig(f"img/cm_europarl.png")


if __name__ == "__main__":
    main()