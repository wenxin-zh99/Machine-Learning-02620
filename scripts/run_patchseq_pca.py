import argparse
import csv
import heapq
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_PATH = Path(
    "/.../.../20200513_Mouse_PatchSeq_Release_cpm.v2/"
    "20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
)
DEFAULT_PCA_PATH = Path("/.../.../pca.py")
DEFAULT_OUTPUT_DIR = Path("/.../.../")
DEFAULT_METADATA_PATH = Path("/.../.../20200625_patchseq_metadata_mouse.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run custom PCA on Patch-seq expression data and save features for downstream models."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the Patch-seq CSV file.",
    )
    parser.add_argument(
        "--pca-path",
        type=Path,
        default=DEFAULT_PCA_PATH,
        help="Path to the custom PCA implementation file.",
    )
    parser.add_argument(
        "--top-k-genes",
        type=int,
        default=2000,
        help="Number of high-variance genes to keep before PCA.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="Number of principal components to compute.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Optional metadata CSV for coloring PCA points by labels.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="corresponding_AIT2.3.1_alias",
        help="Metadata column used as class labels.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="full",
        choices=["full", "prefix"],
        help="Use the full label string or only its first word.",
    )
    parser.add_argument(
        "--metadata-id-col",
        type=str,
        default="transcriptomics_sample_id",
        help="Metadata column used to match PCA cell IDs.",
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=4,
        help="Keep the most frequent N labels and group the rest into 'Other'.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a PC1 vs PC2 scatter plot.",
    )
    return parser.parse_args()


def load_custom_pca_class(pca_path):
    spec = importlib.util.spec_from_file_location("custom_pca_module", pca_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PCA


def iter_gene_rows(csv_path):
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cell_ids = header[1:]
        for row in reader:
            gene = row[0]
            values = np.asarray(row[1:], dtype=np.float32)
            yield gene, values, cell_ids


def select_top_variable_genes(csv_path, top_k=2000):
    heap = []
    cell_ids = None

    for gene, values, current_cell_ids in iter_gene_rows(csv_path):
        if cell_ids is None:
            cell_ids = current_cell_ids

        log_values = np.log1p(values)
        variance = float(np.var(log_values, ddof=1))

        if len(heap) < top_k:
            heapq.heappush(heap, (variance, gene))
        elif variance > heap[0][0]:
            heapq.heapreplace(heap, (variance, gene))

    selected = sorted(heap, reverse=True)
    selected_genes = [gene for _, gene in selected]
    return selected_genes, cell_ids


def load_selected_gene_matrix(csv_path, selected_genes):
    selected_set = set(selected_genes)
    gene_to_values = {}
    cell_ids = None

    for gene, values, current_cell_ids in iter_gene_rows(csv_path):
        if cell_ids is None:
            cell_ids = current_cell_ids
        if gene in selected_set:
            gene_to_values[gene] = np.log1p(values)

    ordered_rows = [gene_to_values[gene] for gene in selected_genes if gene in gene_to_values]
    genes_found = [gene for gene in selected_genes if gene in gene_to_values]

    if not ordered_rows:
        raise ValueError("No selected genes were loaded from the CSV file.")

    # CSV is gene x cell, but PCA expects sample x feature, so transpose.
    X = np.stack(ordered_rows, axis=0).T.astype(np.float64, copy=False)
    return X, cell_ids, genes_found


def save_pca_scores(cell_ids, X_pca, out_path):
    headers = ["cell_id"] + [f"PC{i + 1}" for i in range(X_pca.shape[1])]
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for cell_id, coords in zip(cell_ids, X_pca):
            writer.writerow([cell_id, *map(float, coords)])


def save_selected_genes(genes, out_path):
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gene"])
        for gene in genes:
            writer.writerow([gene])


def load_metadata(metadata_path, id_col):
    with metadata_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    mapping = {}
    for row in rows:
        cell_id = row.get(id_col, "")
        if cell_id:
            mapping[cell_id] = row
    return mapping


def normalize_label(value, label_mode):
    value = value.strip()
    if not value:
        return "Unknown"
    if label_mode == "prefix":
        return value.split()[0]
    return value


def build_plot_labels(cell_ids, metadata_map, label_col, top_labels, label_mode):
    if metadata_map is None:
        return None

    raw_labels = []
    for cell_id in cell_ids:
        row = metadata_map.get(cell_id)
        if row is None:
            raw_labels.append("Unknown")
        else:
            value = row.get(label_col, "")
            raw_labels.append(normalize_label(value, label_mode))

    counts = {}
    for label in raw_labels:
        counts[label] = counts.get(label, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    keep = {label for label, _ in ranked[:top_labels]}

    grouped_labels = []
    for label in raw_labels:
        if label in keep:
            grouped_labels.append(label)
        else:
            grouped_labels.append("Other")

    return grouped_labels


def save_plot(X_pca, explained_variance_ratio, out_path, grouped_labels=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    if grouped_labels is None:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], s=8, alpha=0.6, linewidths=0)
    else:
        unique_labels = list(dict.fromkeys(grouped_labels))
        cmap = plt.get_cmap("tab10")
        for idx, label in enumerate(unique_labels):
            mask = np.array([value == label for value in grouped_labels])
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                s=10,
                alpha=0.7,
                linewidths=0,
                color=cmap(idx % 10),
                label=label,
            )

            center_x = float(np.mean(X_pca[mask, 0]))
            center_y = float(np.mean(X_pca[mask, 1]))
            ax.text(center_x, center_y, label, fontsize=9, weight="bold")

        ax.legend(frameon=False, fontsize=8)

    ax.set_title("Patch-seq PCA")
    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.2f}% var)")
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1] * 100:.2f}% var)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    if args.top_k_genes < 1:
        raise ValueError("--top-k-genes must be at least 1.")
    if args.n_components < 1:
        raise ValueError("--n-components must be at least 1.")
    if args.top_labels < 1:
        raise ValueError("--top-labels must be at least 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    PCA = load_custom_pca_class(args.pca_path)

    top_genes, cell_ids = select_top_variable_genes(args.data_path, top_k=args.top_k_genes)
    print(f"Selected {len(top_genes)} high-variance genes.")
    print("Top 10 genes:", top_genes[:10])

    X, cell_ids, genes_used = load_selected_gene_matrix(args.data_path, top_genes)
    print(f"Matrix shape for PCA: {X.shape} (cells x genes)")

    pca = PCA(n_components=args.n_components)
    X_pca = pca.fit_transform(X)

    print("\nExplained variance ratio:", pca.explained_variance_ratio_)
    print("Total explained variance:", float(pca.explained_variance_ratio_.sum()))

    print("\nFirst 10 PCA coordinates:")
    preview_dims = min(2, X_pca.shape[1])
    for idx in range(min(10, len(cell_ids))):
        preview = " ".join(str(float(X_pca[idx, j])) for j in range(preview_dims))
        print(cell_ids[idx], preview)

    scores_path = args.output_dir / "patchseq_pca_scores.csv"
    genes_path = args.output_dir / "patchseq_selected_genes.csv"
    save_pca_scores(cell_ids, X_pca, scores_path)
    save_selected_genes(genes_used, genes_path)

    print(f"\nSaved PCA scores to: {scores_path}")
    print(f"Saved selected genes to: {genes_path}")
    print(f"Genes used for PCA: {len(genes_used)}")

    if args.plot:
        if X_pca.shape[1] < 2:
            raise ValueError("Need at least 2 principal components to make a scatter plot.")
        plot_path = args.output_dir / "patchseq_pca_scatter.png"
        metadata_map = None
        grouped_labels = None
        if args.metadata_path is not None:
            metadata_map = load_metadata(args.metadata_path, args.metadata_id_col)
            grouped_labels = build_plot_labels(
                cell_ids,
                metadata_map,
                args.label_col,
                args.top_labels,
                args.label_mode,
            )
            grouped_unique = sorted(set(grouped_labels))
            print(f"Using metadata labels from: {args.metadata_path}")
            print(f"Coloring by column: {args.label_col}")
            print(f"Label mode: {args.label_mode}")
            print(f"Plot groups: {grouped_unique}")
        save_plot(X_pca, pca.explained_variance_ratio_, plot_path, grouped_labels=grouped_labels)
        print(f"Saved PCA scatter plot to: {plot_path}")


if __name__ == "__main__":
    main()
