from typing import Any, cast

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class ClusterClassifier(ClassifierMixin, BaseEstimator):
    """Gör en klustrare (KMeans, GaussianMixture, ...) till en klassificerare.

    fit(X, y): klustra X (y används inte av klustringen), mappa sedan varje
    kluster till majoritetsklassen bland träningsexemplen i klustret.
    predict(X): kluster -> klass via mappningen.

    `clusterer` är en klustrare med fit(X) och predict(X). Typen är Any eftersom
    BaseEstimator inte deklarerar fit/predict (de kommer från mixins).
    """

    def __init__(self, clusterer: Any):
        self.clusterer = clusterer

    def fit(self, X, y):
        y = np.asarray(y)
        self.clusterer_: Any = cast(Any, clone(self.clusterer)).fit(X)
        clusters = self.clusterer_.predict(X)
        self.classes_ = np.unique(y)
        self.fallback_ = np.bincount(y).argmax()
        self.map_ = {c: np.bincount(y[clusters == c]).argmax()
                     for c in np.unique(clusters)}
        return self

    def predict(self, X):
        clusters = self.clusterer_.predict(X)
        return np.array([self.map_.get(c, self.fallback_) for c in clusters])   

    def set_params(self, **params):
        """Skickar vidare parametrar som inte tillhör ClusterClassifier till klustraren.
 
        Gör att samma param_grid-stil som för övervakade modeller fungerar:
        {"covariance_type": [...]} istället för {"clusterer__covariance_type": [...]}.
        Prefixet `clusterer__` fungerar fortfarande. Okända namn ger fortfarande
        ValueError från klustraren (stavfel upptäcks).
        """
        own = {k: v for k, v in params.items()
               if k == "clusterer" or k.startswith("clusterer__")}
        inner = {k: v for k, v in params.items() if k not in own}
        if own:
            super().set_params(**own)
        if inner:
            self.clusterer.set_params(**inner)
        return self