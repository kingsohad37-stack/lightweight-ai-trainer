"""
Classical ML algorithms. These call real training procedures (scikit-learn)
— gradient descent for logistic/linear regression, actual tree-building for
decision trees/random forests, actual distance computation for k-NN, etc.
No wrapper here fakes a fit() call.
"""
from __future__ import annotations
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans

CLASSICAL_BUILDERS = {
    "linear_regression": lambda cfg: LinearRegression(),
    "logistic_regression": lambda cfg: LogisticRegression(max_iter=1000, random_state=cfg.random_seed),
    "naive_bayes": lambda cfg: GaussianNB(),
    "knn": lambda cfg: KNeighborsClassifier(n_neighbors=5),
    "decision_tree": lambda cfg: (
        DecisionTreeRegressor(random_state=cfg.random_seed) if cfg.task == "regression"
        else DecisionTreeClassifier(random_state=cfg.random_seed)
    ),
    "random_forest": lambda cfg: (
        RandomForestRegressor(n_estimators=100, random_state=cfg.random_seed, n_jobs=1)
        if cfg.task == "regression"
        else RandomForestClassifier(n_estimators=100, random_state=cfg.random_seed, n_jobs=1)
    ),
    "svm": lambda cfg: (
        SVR() if cfg.task == "regression" else SVC(probability=True, random_state=cfg.random_seed)
    ),
    "kmeans": lambda cfg: KMeans(n_clusters=cfg.n_clusters, random_state=cfg.random_seed, n_init=10),
}


def build_model(algorithm: str, cfg):
    if algorithm not in CLASSICAL_BUILDERS:
        raise ValueError(f"Unknown classical algorithm: {algorithm}")
    return CLASSICAL_BUILDERS[algorithm](cfg)


def is_classical(algorithm: str) -> bool:
    return algorithm in CLASSICAL_BUILDERS
