"""Pré-processamento e algoritmos comparados."""
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.dados import CATEGORICAS, NUMERICAS, SEMENTE


def criar_pipeline(algoritmo) -> Pipeline:
    """Tudo que aprende com os dados fica no pipeline e é ajustado só no treino."""
    preparo = ColumnTransformer([
        ("numericas", StandardScaler(), NUMERICAS),
        ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ])
    return Pipeline([("preparo", preparo), ("modelo", algoritmo)])


def candidatos() -> dict:
    """Nome -> (algoritmo, hiperparâmetros testados).

    Não usamos class_weight: balancear as classes distorce a probabilidade prevista.
    """
    return {
        "Referência (taxa média)": (DummyClassifier(strategy="prior"), {}),
        "Regressão Logística": (LogisticRegression(max_iter=2000), {"modelo__C": [0.1, 1.0]}),
        "Árvore de Decisão": (
            DecisionTreeClassifier(min_samples_leaf=50, random_state=SEMENTE),
            {"modelo__max_depth": [4, 6]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=SEMENTE),
            {"modelo__max_depth": [6, 10]},
        ),
        "Gradient Boosting": (
            HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, l2_regularization=1.0,
                                           early_stopping=False, random_state=SEMENTE),
            {"modelo__max_leaf_nodes": [7, 15]},
        ),
    }
