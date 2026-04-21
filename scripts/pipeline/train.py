from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier


META_COLUMNS = ["hemisphere", "donor_id", "transcriptomics_batch", "cell_type"]


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray
    classes: np.ndarray


def prepare_dataset(
    expr_with_meta: pd.DataFrame,
    *,
    test_size: float = 0.3,
    random_state: int = 42,
) -> DatasetSplit:
    feature_cols = expr_with_meta.columns.difference(META_COLUMNS)
    X = expr_with_meta[feature_cols].to_numpy(dtype=float)
    y = expr_with_meta["cell_type"].to_numpy()
    groups = expr_with_meta["donor_id"].to_numpy()
    classes = np.unique(y)

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    return DatasetSplit(
        X_train=X[train_idx],
        X_test=X[test_idx],
        y_train=y[train_idx],
        y_test=y[test_idx],
        groups_train=groups[train_idx],
        classes=classes,
    )


def _metric_summary(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    y_bin = label_binarize(y_true, classes=classes)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_auc": roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"),
    }


def _per_class_table(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Precision": precision_score(y_true, y_pred, average=None, labels=classes),
            "Recall": recall_score(y_true, y_pred, average=None, labels=classes),
        },
        index=classes,
    )


def cross_validated_ensemble(
    estimator: Any,
    split: DatasetSplit,
    *,
    model_name: str,
    random_state: int = 42,
    n_splits: int = 3,
) -> dict[str, Any]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    val_metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "macro_auc": [],
    }
    test_probas = []

    for train_idx, val_idx in cv.split(split.X_train, split.y_train, split.groups_train):
        X_tr, X_val = split.X_train[train_idx], split.X_train[val_idx]
        y_tr, y_val = split.y_train[train_idx], split.y_train[val_idx]

        estimator.fit(X_tr, y_tr)
        y_val_pred = estimator.predict(X_val)
        y_val_proba = estimator.predict_proba(X_val)
        fold_metrics = _metric_summary(y_val, y_val_pred, y_val_proba, split.classes)
        for key, value in fold_metrics.items():
            val_metrics[key].append(value)

        test_probas.append(estimator.predict_proba(split.X_test))

    avg_test_proba = np.mean(test_probas, axis=0)
    y_test_pred = split.classes[np.argmax(avg_test_proba, axis=1)]
    test_metrics = _metric_summary(split.y_test, y_test_pred, avg_test_proba, split.classes)

    return {
        "model_name": model_name,
        "classes": split.classes,
        "cv_metrics": val_metrics,
        "test_metrics": test_metrics,
        "y_test": split.y_test,
        "y_test_pred": y_test_pred,
        "test_probabilities": avg_test_proba,
        "per_class": _per_class_table(split.y_test, y_test_pred, split.classes),
    }


def make_pca_lr_pipeline(*, n_components: int = 50, random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=random_state)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_pca_svm_pipeline(*, n_components: int = 50, random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=random_state)),
            ("clf", SVC(kernel="rbf", probability=True, random_state=random_state)),
        ]
    )


def make_pca_nb_pipeline(*, n_components: int = 50, random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=random_state)),
            ("clf", GaussianNB()),
        ]
    )


def run_xgboost(
    expr_with_meta: pd.DataFrame,
    *,
    test_size: float = 0.3,
    random_state: int = 42,
    n_splits: int = 3,
    device: str = "cuda",
) -> dict[str, Any]:
    feature_cols = expr_with_meta.columns.difference(META_COLUMNS)
    X = expr_with_meta[feature_cols].to_numpy(dtype=float)
    y_raw = expr_with_meta["cell_type"].to_numpy()
    groups = expr_with_meta["donor_id"].to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    classes = np.arange(len(encoder.classes_))

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    val_metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "macro_auc": [],
    }
    test_probas = []

    for train_fold_idx, val_fold_idx in cv.split(X_train, y_train, groups_train):
        X_tr, X_val = X_train[train_fold_idx], X_train[val_fold_idx]
        y_tr, y_val = y_train[train_fold_idx], y_train[val_fold_idx]

        model = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=random_state,
            tree_method="hist",
            device=device,
        )
        model.fit(X_tr, y_tr)

        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)
        y_val_bin = label_binarize(y_val, classes=classes)
        fold_metrics = {
            "accuracy": accuracy_score(y_val, y_val_pred),
            "balanced_accuracy": balanced_accuracy_score(y_val, y_val_pred),
            "macro_f1": f1_score(y_val, y_val_pred, average="macro"),
            "macro_auc": roc_auc_score(
                y_val_bin, y_val_proba, average="macro", multi_class="ovr"
            ),
        }
        for key, value in fold_metrics.items():
            val_metrics[key].append(value)
        test_probas.append(model.predict_proba(X_test))

    avg_test_proba = np.mean(test_probas, axis=0)
    y_test_pred_int = np.argmax(avg_test_proba, axis=1)
    y_test_pred = encoder.inverse_transform(y_test_pred_int)
    y_test_true = encoder.inverse_transform(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)

    test_metrics = {
        "accuracy": accuracy_score(y_test_true, y_test_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test_true, y_test_pred),
        "macro_f1": f1_score(y_test_true, y_test_pred, average="macro"),
        "macro_auc": roc_auc_score(
            y_test_bin, avg_test_proba, average="macro", multi_class="ovr"
        ),
    }

    return {
        "model_name": "XGBoost",
        "classes": encoder.classes_,
        "cv_metrics": val_metrics,
        "test_metrics": test_metrics,
        "y_test": y_test_true,
        "y_test_pred": y_test_pred,
        "test_probabilities": avg_test_proba,
        "per_class": pd.DataFrame(
            {
                "Precision": precision_score(
                    y_test_true,
                    y_test_pred,
                    average=None,
                    labels=encoder.classes_,
                ),
                "Recall": recall_score(
                    y_test_true,
                    y_test_pred,
                    average=None,
                    labels=encoder.classes_,
                ),
            },
            index=encoder.classes_,
        ),
    }


def summarize_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        cv_metrics = result["cv_metrics"]
        test_metrics = result["test_metrics"]
        rows.append(
            {
                "Model": result["model_name"],
                "CV Acc (mean)": np.mean(cv_metrics["accuracy"]),
                "CV Bal Acc (mean)": np.mean(cv_metrics["balanced_accuracy"]),
                "CV Macro F1 (mean)": np.mean(cv_metrics["macro_f1"]),
                "CV Macro AUC (mean)": np.mean(cv_metrics["macro_auc"]),
                "Test Acc": test_metrics["accuracy"],
                "Test Bal Acc": test_metrics["balanced_accuracy"],
                "Test Macro F1": test_metrics["macro_f1"],
                "Test Macro AUC": test_metrics["macro_auc"],
            }
        )
    return pd.DataFrame(rows).set_index("Model")
