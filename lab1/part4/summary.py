
import json
from collections import defaultdict

import numpy as np

# Load results from all experiments
with open("results.json", "r") as file:
    results = json.load(file)

# Group results by dataset and model
groups = defaultdict(list)

for result in results:
    key = (result["dataset"], result["model"])
    groups[key].append(result)

# Store the average results for each dataset/model
summaries = []

for (dataset, model), runs in sorted(groups.items()):

    accuracies = [r["test_accuracy"] for r in runs]
    cv_accuracies = [r["cv_accuracy"] for r in runs]
    f1_scores = [r["f1_score"] for r in runs]
    kappa_scores = [r["kappa"] for r in runs]

    summary = {
        "dataset": dataset,
        "model": model,
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "mean_cv": np.mean(cv_accuracies),
        "mean_f1": np.mean(f1_scores),
        "mean_kappa": np.mean(kappa_scores)
    }

    summaries.append(summary)

    print(f"\nDataset: {dataset}, Model: {model}")
    print(f"Number of runs:   {len(runs)}")
    print(f"Mean accuracy:    {summary['mean_accuracy']:.4f}")
    print(f"Std accuracy:     {summary['std_accuracy']:.4f}")
    print(f"Mean CV accuracy: {summary['mean_cv']:.4f}")
    print(f"Mean F1:          {summary['mean_f1']:.4f}")
    print(f"Mean Kappa:       {summary['mean_kappa']:.4f}")


# Overall averages across all datasets
print("\n--- Overall model performance ---")

models = sorted(set(s["model"] for s in summaries))

for model in models:
    model_summaries = [
        s for s in summaries if s["model"] == model
    ]

    overall_accuracy = np.mean([
        s["mean_accuracy"] for s in model_summaries
    ])

    overall_f1 = np.mean([
        s["mean_f1"] for s in model_summaries
    ])

    overall_kappa = np.mean([
        s["mean_kappa"] for s in model_summaries
    ])

    print(f"\nModel: {model}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"Overall F1:       {overall_f1:.4f}")
    print(f"Overall Kappa:    {overall_kappa:.4f}")
