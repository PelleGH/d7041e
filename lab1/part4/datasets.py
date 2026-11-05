import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris(return_X_y=True)
breast_cancer = datasets.load_breast_cancer(return_X_y=True)
wine = datasets.load_wine(return_X_y=True)
digits = datasets.load_digits(return_X_y=True)

glass = datasets.fetch_openml("glass", return_X_y=True, version=1)
ecoli = datasets.fetch_openml("ecoli", return_X_y=True, version=1)
vehicle = datasets.fetch_openml("vehicle", return_X_y=True, version=1)
yeast = datasets.fetch_openml("yeast", return_X_y=True, version=1)
ionosphere = datasets.fetch_openml("ionosphere", return_X_y=True, version=1)
diabetes = datasets.fetch_openml("diabetes", return_X_y=True, version=1)

def drop_rare_classes(dataset, min_count=5):
    X, Y = dataset
    X = np.asarray(X, dtype=float)
    _, Y = np.unique(np.asarray(Y), return_inverse=True)
    keep = np.bincount(Y)[Y] >= min_count
    X, Y = X[keep], Y[keep]
    _, Y = np.unique(Y, return_inverse=True)
    return X, Y

def split_dataset(dataset, seed):
    X, Y = dataset

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed, stratify=Y)
    return ((X_train, Y_train), (X_test, Y_test))

def normalize_dataset(dataset):
    ((X_train, Y_train), (X_test, Y_test)) = dataset

    scaler = StandardScaler().fit(X=X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return ((X_train, Y_train), (X_test, Y_test))

def get_datasets(seed = 67):
    sets = [
        (iris, "iris"),
        (breast_cancer, "breast_cancer"),
        (wine, "wine"),
        (digits, "digits"),
        (glass, "glass"),
        (ecoli, "ecoli"),
        (vehicle, "vehicle"),
        (yeast, "yeast"),
        (ionosphere, "ionosphere"),
        (diabetes, "diabetes")
    ]

    res = []

    for s, name in sets:
        s = drop_rare_classes(s)
        split = split_dataset(s, seed)
        normalized = normalize_dataset(split)
        # any more?

        res.append((normalized, name))

    return res