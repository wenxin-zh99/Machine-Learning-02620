from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
MODEL_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def _unpack(results: list[dict[str, Any]]):
    model_names = [r["model_name"] for r in results]
    predictions = [r["y_test_pred"] for r in results]
    probabilities = [r["test_probabilities"] for r in results]
    cv_all = [r["cv_metrics"] for r in results]
    y_true = results[0]["y_test"]
    classes = results[0]["classes"]
    return model_names, predictions, probabilities, cv_all, y_true, classes


def plot_confusion_matrices(results: list[dict[str, Any]], save: bool = True) -> None:
    matplotlib.rcParams.update({"font.size": 11})
    model_names, predictions, _, _, y_true, classes = _unpack(results)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for ax, name, preds in zip(axes, model_names, predictions):
        cm = confusion_matrix(y_true, preds, labels=classes, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
        ax.set_title(name, fontsize=14, fontweight="bold")

    plt.suptitle("Confusion Matrices  (row-normalized)", fontsize=16, y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(results: list[dict[str, Any]], save: bool = True) -> None:
    matplotlib.rcParams.update({"font.size": 10})
    model_names, _, probabilities, _, y_true, classes = _unpack(results)
    y_bin = label_binarize(y_true, classes=classes)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, name, probas in zip(axes, model_names, probabilities):
        for i, (cls, col) in enumerate(zip(classes, PALETTE)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probas[:, i])
            ax.plot(fpr, tpr, color=col, lw=2, label=f"{cls}  AUC={auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name}  — ROC (OvR)", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8.5)

    plt.suptitle("ROC Curves — One-vs-Rest per class", fontsize=16, y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
    plt.show()


def model_comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    model_names, predictions, probabilities, cv_all, y_true, classes = _unpack(results)
    y_bin = label_binarize(y_true, classes=classes)
    rows = []

    for name, preds, probas, cv in zip(model_names, predictions, probabilities, cv_all):
        rows.append({
            "Model": name,
            "CV Acc (±std)": f"{np.mean(cv['accuracy']):.4f} ± {np.std(cv['accuracy']):.4f}",
            "CV F1  (±std)": f"{np.mean(cv['macro_f1']):.4f} ± {np.std(cv['macro_f1']):.4f}",
            "CV AUC (±std)": f"{np.mean(cv['macro_auc']):.4f} ± {np.std(cv['macro_auc']):.4f}",
            "Test Acc": round(accuracy_score(y_true, preds), 4),
            "Test Bal-Acc": round(balanced_accuracy_score(y_true, preds), 4),
            "Test F1": round(f1_score(y_true, preds, average="macro"), 4),
            "Test AUC": round(roc_auc_score(y_bin, probas, average="macro", multi_class="ovr"), 4),
        })

    return pd.DataFrame(rows).set_index("Model")


def plot_metric_bars(results: list[dict[str, Any]], save: bool = True) -> None:
    matplotlib.rcParams.update({"font.size": 11})
    model_names, predictions, probabilities, _, y_true, classes = _unpack(results)
    y_bin = label_binarize(y_true, classes=classes)

    metrics = {
        "Test Acc": [accuracy_score(y_true, p) for p in predictions],
        "Test Bal-Acc": [balanced_accuracy_score(y_true, p) for p in predictions],
        "Test F1": [f1_score(y_true, p, average="macro") for p in predictions],
        "Test AUC": [
            roc_auc_score(y_bin, pr, average="macro", multi_class="ovr")
            for pr in probabilities
        ],
    }

    metric_labels = list(metrics.keys())
    n_models = len(model_names)
    x = np.arange(len(metric_labels))
    width = 0.18
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, color) in enumerate(zip(model_names, MODEL_COLORS)):
        vals = [metrics[m][i] for m in metric_labels]
        bars = ax.bar(x + offsets[i], vals, width, label=name, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Per-Metric Bar Chart (held-out test set)")
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    if save:
        plt.savefig("model_comparison_bars.png", dpi=150, bbox_inches="tight")
    plt.show()


def mcnemar_test(results: list[dict[str, Any]]) -> pd.DataFrame:
    model_names, predictions, _, _, y_true, _ = _unpack(results)
    correct = {
        name: (np.array(preds) == np.array(y_true))
        for name, preds in zip(model_names, predictions)
    }

    rows = []
    for m1, m2 in combinations(model_names, 2):
        c1, c2 = correct[m1], correct[m2]
        b = int(np.sum(c1 & ~c2))
        c = int(np.sum(~c1 & c2))
        if b + c == 0:
            stat, p = 0.0, 1.0
        else:
            stat = (abs(b - c) - 1) ** 2 / (b + c)
            p = float(1 - chi2.cdf(stat, df=1))
        rows.append({"Model A": m1, "Model B": m2, "b (A✓ B✗)": b, "c (A✗ B✓)": c,
                     "chi2 (cc)": round(stat, 4), "p-value": round(p, 4)})

    df = pd.DataFrame(rows)
    pvals = df["p-value"].values
    _, p_bonf, _, _ = multipletests(pvals, alpha=0.05, method="bonferroni")
    _, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    df["p_bonferroni"] = np.round(p_bonf, 4)
    df["p_fdr_bh"] = np.round(p_fdr, 4)
    df["sig (raw)"] = df["p-value"] < 0.05
    df["sig (Bonf)"] = df["p_bonferroni"] < 0.05
    df["sig (FDR)"] = df["p_fdr_bh"] < 0.05
    return df


def run_evaluation(results: list[dict[str, Any]]) -> dict:
    print("=== Confusion Matrices ===")
    plot_confusion_matrices(results)

    print("\n=== ROC Curves ===")
    plot_roc_curves(results)

    print("\n=== Metric Bar Chart ===")
    plot_metric_bars(results)

    print("\n=== Model Comparison Table ===")
    table = model_comparison_table(results)
    print(table.to_string())

    print("\n=== McNemar Test ===")
    mc = mcnemar_test(results)
    print(mc.to_string(index=False))

    return {"comparison_table": table, "mcnemar": mc}
