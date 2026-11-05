import numpy as np
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


def create_default_models():

    return [
        (create_knn(1), "knn"),
        (create_svm("linear"), "svm")
    ]