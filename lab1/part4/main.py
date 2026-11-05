import json

from crossvalidation import cross_validation
from datasets import get_datasets
from models import create_default_models
from sklearn import datasets
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split


def main():
    results = []

    for seed in range(10):
        ds = get_datasets(seed)
        models = create_default_models()

        for ((X_train, Y_train), (X_test, Y_test)), dataset_name in ds:

            print(f"Starting dataset {dataset_name}")

            for (model, param_grid), model_name in models:

                print(f"Model {model_name}")

                best_params, best_score, best_model = cross_validation(model, param_grid, X_train, Y_train, seed)

                Y_pred = best_model.predict(X_test)

                acc = accuracy_score(Y_test, Y_pred)
                f1 = f1_score(Y_test, Y_pred, average="macro")
                kappa = cohen_kappa_score(Y_test, Y_pred)

                results.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "best_params": best_params,
                    "cv_accuracy": best_score,
                    "test_accuracy": acc,
                    "f1_score": f1,
                    "kappa": kappa,
                    "seed": seed,
                })

    with open("results.json", "w") as f:
        json.dump(results, f)
                


if __name__ == "__main__":
    main()