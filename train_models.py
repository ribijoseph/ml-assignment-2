"""
ML Assignment 2 - training script
Dataset : Online Shoppers Purchasing Intention (UCI, id 468)
Task    : binary classification -> will the session end in a purchase (Revenue)?

Trains 5 classifiers on the same train split, evaluates them on a held-out
test split, saves every fitted pipeline to model/ and writes test_data.csv.

Run:  python train_models.py
"""
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 461  # last digits of roll no. 2026AJ05461
ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
TARGET = "Revenue"

# sklearn 1.9 flags SVC(probability=True) as deprecated; still works in 1.9/1.10
warnings.filterwarnings("ignore", category=FutureWarning)

# numeric codes in the raw file that are really category ids, not quantities
CATEGORICAL = ["Month", "VisitorType", "Weekend",
               "OperatingSystems", "Browser", "Region", "TrafficType"]
NUMERIC = ["Administrative", "Administrative_Duration", "Informational",
           "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
           "BounceRates", "ExitRates", "PageValues", "SpecialDay"]


def load_data():
    shop_df = pd.read_csv(DATA_FILE)
    shop_df = shop_df.drop_duplicates()
    shop_df["Weekend"] = shop_df["Weekend"].astype(str)
    shop_df[TARGET] = shop_df[TARGET].astype(int)
    return shop_df


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL),
    ])


def build_candidates():
    """Each entry: (file name, estimator, param grid searched with 5-fold CV)."""
    return {
        "Logistic Regression": ("logistic_regression",
                                LogisticRegression(max_iter=2000, class_weight="balanced"),
                                {"clf__C": [0.1, 1.0, 10.0]}),
        "Decision Tree": ("decision_tree",
                          DecisionTreeClassifier(random_state=SEED),
                          {"clf__max_depth": [4, 6, 8, 12],
                           "clf__min_samples_leaf": [5, 20, 50]}),
        "KNN": ("knn",
                KNeighborsClassifier(),
                {"clf__n_neighbors": [5, 11, 21, 31],
                 "clf__weights": ["uniform", "distance"]}),
        "Naive Bayes": ("naive_bayes", GaussianNB(),
                        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1, 0.5]}),
        "Support Vector Machine": ("svm",
                                   SVC(kernel="rbf", probability=True,
                                       class_weight="balanced", random_state=SEED),
                                   {"clf__C": [0.5, 2.0]}),
    }


def score_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main():
    shop_df = load_data()
    X = shop_df[NUMERIC + CATEGORICAL]
    y = shop_df[TARGET]
    print(f"Rows: {len(shop_df)} | features: {X.shape[1]} | "
          f"purchase rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    MODEL_DIR.mkdir(exist_ok=True)
    results = {}
    chosen_params = {}

    for display_name, (file_stub, estimator, grid) in build_candidates().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", estimator)])
        if grid:
            search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=5, n_jobs=-1)
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            chosen_params[display_name] = search.best_params_
        else:
            fitted = pipe.fit(X_train, y_train)
            chosen_params[display_name] = {}

        joblib.dump(fitted, MODEL_DIR / f"{file_stub}.joblib")
        results[display_name] = score_model(fitted, X_test, y_test)
        print(f"  trained {display_name:<24} {chosen_params[display_name]}")

    # held-out rows (raw form, target included) for upload in the Streamlit app
    test_out = X_test.copy()
    test_out[TARGET] = y_test
    test_out.to_csv(ROOT / "test_data.csv", index=False)

    metrics_table = pd.DataFrame(results).T.round(4)
    metrics_table.index.name = "ML Model Name"
    metrics_table.to_csv(MODEL_DIR / "test_metrics.csv")
    with open(MODEL_DIR / "best_params.json", "w") as fh:
        json.dump(chosen_params, fh, indent=2, default=str)

    print("\nTest-set metrics")
    print(metrics_table.to_string())


if __name__ == "__main__":
    main()
