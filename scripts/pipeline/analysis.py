from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

META_COLS = ["hemisphere", "donor_id", "transcriptomics_batch", "cell_type"]


def plot_xgb_feature_importance(
    expr_with_meta: pd.DataFrame,
    *,
    top_n: int = 25,
    test_size: float = 0.3,
    random_state: int = 42,
    device: str = "cuda",
    save: bool = True,
) -> list[str]:
    feature_cols = np.array(list(expr_with_meta.columns.difference(META_COLS)))
    X = expr_with_meta[feature_cols].values.astype(float)
    y_raw = expr_with_meta["cell_type"].values
    groups = expr_with_meta["donor_id"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, _ = next(gss.split(X, y, groups=groups))
    X_train, y_train = X[train_idx], y[train_idx]

    clf = XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=random_state,
        tree_method="hist", device=device,
    )
    clf.fit(X_train, y_train)

    imp = clf.feature_importances_
    top_idx = np.argsort(imp)[::-1][:top_n]
    top_genes = feature_cols[top_idx]
    top_imp = imp[top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(top_n), top_imp[::-1], color="#4C72B0", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_genes[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"Top {top_n} Genes — XGBoost Feature Importance (full training set)",
                 fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("xgb_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Top {top_n} most important genes:")
    for gene, v in zip(top_genes, top_imp):
        print(f"  {gene:<20s}: {v:.6f}")

    return list(top_genes)


def plot_pca_marker_heatmap(
    expr_with_meta: pd.DataFrame,
    *,
    n_components: int = 50,
    top_per_pc: int = 12,
    test_size: float = 0.3,
    random_state: int = 42,
    save: bool = True,
) -> None:
    feature_cols = np.array(list(expr_with_meta.columns.difference(META_COLS)))
    X = expr_with_meta[feature_cols].values.astype(float)
    y = expr_with_meta["cell_type"].values
    groups = expr_with_meta["donor_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, _ = next(gss.split(X, y, groups=groups))
    X_train, y_train = X[train_idx], y[train_idx]

    sc = StandardScaler()
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit_transform(sc.fit_transform(X_train))
    loadings = pca.components_

    top_set = set()
    for pc in range(3):
        for idx in np.argsort(np.abs(loadings[pc]))[::-1][:top_per_pc]:
            top_set.add(int(idx))
    top_idx = sorted(top_set)
    top_genes = feature_cols[top_idx]

    df_train = pd.DataFrame(X_train, columns=feature_cols)
    df_train["cell_type"] = y_train
    mean_expr = df_train.groupby("cell_type")[top_genes].mean()
    mean_z = (mean_expr - mean_expr.mean()) / (mean_expr.std() + 1e-8)

    cell_types = list(mean_z.index)
    fig, ax = plt.subplots(figsize=(max(10, len(top_genes) * 0.55), 4))
    im = ax.imshow(mean_z.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(top_genes)))
    ax.set_xticklabels(top_genes, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(cell_types)))
    ax.set_yticklabels(cell_types, fontsize=10)
    plt.colorbar(im, ax=ax, label="Z-score of mean log-CPM")
    ax.set_title("Mean Expression of Top PC-loading Genes per Cell Type (Z-scored)",
                 fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("pca_marker_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()

    for pc_idx in range(3):
        top5 = feature_cols[np.argsort(loadings[pc_idx])[::-1][:5]]
        print(f"PC{pc_idx + 1} top positive genes: {list(top5)}")


def run_analysis(expr_with_meta: pd.DataFrame, device: str = "cuda") -> None:
    print("=== XGBoost Feature Importance ===")
    plot_xgb_feature_importance(expr_with_meta, device=device)

    print("\n=== PCA Marker Gene Heatmap ===")
    plot_pca_marker_heatmap(expr_with_meta)
