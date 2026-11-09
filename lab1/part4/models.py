import numpy as np
from clf import ClusterClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def create_knn(k):

    # Param grid
    param_grid = {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
    }

    return KNeighborsClassifier(n_neighbors=k), param_grid


def create_svm(kernel):

    # Param grid
    param_grid = [
        {
            "kernel": ["linear"],
            "C": [0.1, 1, 10]
        },
        {
            "kernel": ["rbf"],
            "C": [0.1, 1, 10],
            "gamma": ["scale", 0.01, 0.1]
        },
        {
            "kernel": ["poly"],
            "C": [0.1, 1, 10],
            "degree": [2, 3]
        }
    ]

    return SVC(kernel=kernel), param_grid

def create_kmeans(seed):

    param_grid = {
        "n_clusters": range(2, 10),
        "algorithm": ["lloyd", "elkan"]
    }

    return ClusterClassifier(KMeans(random_state=seed)), param_grid

def create_gmm(seed=42):
    param_grid = {
        "clusterer__n_components": [2,3,4,5,6,7,8,9],
        "clusterer__covariance_type": [
            # "full" # "full", "tied", "diag", "spherical"
            "diag", "spherical"
        ]
    }

    model = ClusterClassifier(
        GaussianMixture(
            random_state=seed,
            n_init=3
        )
    )

    return model, param_grid

def create_default_models(seed):

    return [
        # (create_knn(1), "knn"),
        # (create_svm("linear"), "svm"),
        # (create_kmeans(seed), "kmeans"),
        (create_gmm(seed), "gmm")
    ]