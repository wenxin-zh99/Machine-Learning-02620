from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

META_COLS = ["hemisphere", "donor_id", "transcriptomics_batch", "cell_type"]
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def _feat_array(expr_with_meta: pd.DataFrame):
    feat_cols = expr_with_meta.columns.difference(META_COLS)
    X = expr_with_meta[feat_cols].values.astype(float)
    ct = expr_with_meta["cell_type"].values
    cell_types = list(expr_with_meta["cell_type"].value_counts().index)
    color_map = dict(zip(cell_types, PALETTE))
    return X, ct, cell_types, color_map, feat_cols


def plot_cell_type_distribution(expr_with_meta: pd.DataFrame, save: bool = True) -> None:
    matplotlib.rcParams.update({"font.size": 11})
    ct = expr_with_meta["cell_type"].value_counts().sort_values(ascending=False)
    colors = PALETTE[: len(ct)]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.bar(ct.index, ct.values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Cell Type Distribution", fontweight="bold")
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Number of Cells")
    for i, v in enumerate(ct.values):
        ax.text(i, v + max(ct.values) * 0.01, str(v), ha="center", va="bottom", fontsize=10)
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    if save:
        plt.savefig("eda_overview.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_sparsity(expr_with_meta: pd.DataFrame, save: bool = True) -> None:
    X, ct, cell_types, color_map, _ = _feat_array(expr_with_meta)
    lib_size = X.sum(axis=1)
    zero_frac = (X == 0).mean(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(lib_size, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.5)
    axes[0].set_title("Library Size per Cell", fontweight="bold")
    axes[0].set_xlabel("Total CPM")
    axes[0].set_ylabel("Number of Cells")
    axes[0].axvline(lib_size.mean(), color="red", linestyle="--", linewidth=1.2,
                    label=f"Mean = {lib_size.mean():.0f}")
    axes[0].legend()

    vdata = [lib_size[ct == c] for c in cell_types]
    vp = axes[1].violinplot(vdata, showmedians=True)
    for patch, c in zip(vp["bodies"], cell_types):
        patch.set_facecolor(color_map[c])
        patch.set_alpha(0.8)
    axes[1].set_xticks(range(1, len(cell_types) + 1))
    axes[1].set_xticklabels(cell_types)
    axes[1].set_title("Library Size by Cell Type", fontweight="bold")
    axes[1].set_ylabel("Total CPM")

    zdata = [zero_frac[ct == c] for c in cell_types]
    vp2 = axes[2].violinplot(zdata, showmedians=True)
    for patch, c in zip(vp2["bodies"], cell_types):
        patch.set_facecolor(color_map[c])
        patch.set_alpha(0.8)
    axes[2].set_xticks(range(1, len(cell_types) + 1))
    axes[2].set_xticklabels(cell_types)
    axes[2].set_title("Zero-Expression Fraction by Cell Type", fontweight="bold")
    axes[2].set_ylabel("Fraction of Genes = 0")

    plt.tight_layout()
    if save:
        plt.savefig("eda_sparsity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Overall zero fraction: {zero_frac.mean():.3f}")


def plot_hvg(expr_with_meta: pd.DataFrame, n_hvg: int = 2000, save: bool = True) -> list:
    X, _, _, _, feat_cols = _feat_array(expr_with_meta)
    gene_mean = X.mean(axis=0)
    gene_std = X.std(axis=0)
    gene_cv = np.where(gene_mean > 0, gene_std / gene_mean, 0)
    hvg_idx = np.argsort(gene_cv)[::-1][:n_hvg]
    hvg_names = np.array(list(feat_cols))[hvg_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(gene_mean, gene_cv, s=1, alpha=0.3, color="#AAAAAA", label="All genes")
    ax.scatter(gene_mean[hvg_idx], gene_cv[hvg_idx], s=4, alpha=0.6,
               color="#C44E52", label=f"Top {n_hvg} HVG")
    ax.set_xscale("log")
    ax.set_xlabel("Mean Expression (CPM, log scale)")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Mean–CV Plot: Highly Variable Genes", fontweight="bold")
    ax.legend(markerscale=4)
    plt.tight_layout()
    if save:
        plt.savefig("eda_hvg.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Top 10 HVG: {list(hvg_names[:10])}")
    return list(hvg_names)


def plot_umap(expr_with_meta: pd.DataFrame, save: bool = True) -> None:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import umap

    X, ct, cell_types, color_map, _ = _feat_array(expr_with_meta)
    X_scaled = StandardScaler().fit_transform(X)
    X_pca50 = PCA(n_components=50, random_state=42).fit_transform(X_scaled)
    embedding = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42, n_jobs=1).fit_transform(X_pca50)

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in cell_types:
        mask = ct == c
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=5, alpha=0.5,
                   label=c, color=color_map[c])
    ax.set_title("UMAP (50 PCs) — colored by Cell Type", fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=4, framealpha=0.8)
    plt.tight_layout()
    if save:
        plt.savefig("eda_umap.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_marker_heatmap(expr_with_meta: pd.DataFrame, save: bool = True) -> None:
    X, ct, cell_types, _, feat_cols = _feat_array(expr_with_meta)
    feat_arr = np.array(list(feat_cols))

    top_per_class = []
    for c in cell_types:
        mask = ct == c
        ct_mean = X[mask].mean(axis=0)
        top5 = np.argsort(ct_mean)[::-1][:5]
        top_per_class.extend(top5.tolist())

    hm_idx = list(dict.fromkeys(top_per_class))
    hm_genes = feat_arr[hm_idx]
    hm_matrix = np.array([X[ct == c][:, hm_idx].mean(axis=0) for c in cell_types])
    hm_z = (hm_matrix - hm_matrix.mean(axis=0)) / (hm_matrix.std(axis=0) + 1e-8)

    fig, ax = plt.subplots(figsize=(max(10, len(hm_genes) * 0.5), 4))
    im = ax.imshow(hm_z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(hm_genes)))
    ax.set_xticklabels(hm_genes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cell_types)))
    ax.set_yticklabels(cell_types)
    ax.set_title("Mean Expression Z-score: Top 5 HVG per Cell Type", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Z-score")
    plt.tight_layout()
    if save:
        plt.savefig("eda_marker_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_pca_scree(expr_with_meta: pd.DataFrame, n_components: int = 200,
                   n_chosen: int = 50, save: bool = True) -> None:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X, _, _, _, _ = _feat_array(expr_with_meta)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)

    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr) * 100
    evr_pct = evr * 100
    var_at_chosen = cum_evr[n_chosen - 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(range(1, n_components + 1), evr_pct, color="#4C72B0", alpha=0.55, label="Individual")
    ax2 = ax.twinx()
    ax2.plot(range(1, n_components + 1), cum_evr, color="#C44E52", lw=2, label="Cumulative")
    ax2.axvline(n_chosen, color="green", linestyle="--", lw=1.8,
                label=f"PC={n_chosen} ({var_at_chosen:.1f}%)")
    ax2.axhline(var_at_chosen, color="green", linestyle=":", lw=1, alpha=0.6)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)", color="#4C72B0")
    ax2.set_ylabel("Cumulative Explained Variance (%)", color="#C44E52")
    ax.set_title(f"Scree Plot ({n_components} PCs)", fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    ax = axes[1]
    ax.plot(range(1, 81), cum_evr[:80], color="#C44E52", lw=2)
    ax.axvline(n_chosen, color="green", linestyle="--", lw=1.8,
               label=f"PC={n_chosen}: {var_at_chosen:.1f}% variance")
    ax.axhline(var_at_chosen, color="green", linestyle=":", lw=1, alpha=0.6)
    for thresh in [80, 90, 95]:
        pc_thresh = int(np.searchsorted(cum_evr, thresh)) + 1
        ax.axhline(thresh, color="gray", linestyle=":", lw=0.8, alpha=0.5)
        ax.text(81, thresh, f"{thresh}% (PC{pc_thresh})", va="center", fontsize=8, color="gray")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("Cumulative Variance — First 80 PCs", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(1, 80)

    plt.tight_layout()
    if save:
        plt.savefig("pca_scree_plot.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Variance explained by {n_chosen} PCs : {var_at_chosen:.2f}%")
    for thresh in [80, 90, 95]:
        print(f"PCs needed for {thresh}% variance  : {int(np.searchsorted(cum_evr, thresh)) + 1}")


def run_eda(expr_with_meta: pd.DataFrame) -> None:
    print("=== EDA: Cell Type Distribution ===")
    plot_cell_type_distribution(expr_with_meta)

    print("\n=== EDA: Sparsity & Library Size ===")
    plot_sparsity(expr_with_meta)

    print("\n=== EDA: Highly Variable Genes ===")
    plot_hvg(expr_with_meta)

    print("\n=== EDA: UMAP ===")
    plot_umap(expr_with_meta)

    print("\n=== EDA: Marker Gene Heatmap ===")
    plot_marker_heatmap(expr_with_meta)

    print("\n=== EDA: PCA Scree Plot ===")
    plot_pca_scree(expr_with_meta)
