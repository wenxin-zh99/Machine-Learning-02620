from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_META_FILENAME = "20200625_patchseq_metadata_mouse.csv"
DEFAULT_EXPR_FILENAME = (
    "20200513_Mouse_PatchSeq_Release_cpm.v2/"
    "20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
)
META_COLUMNS = [
    "hemisphere",
    "donor_id",
    "transcriptomics_batch",
    "corresponding_AIT2.3.1_alias",
]
LEAKY_GENE_LABELS = ["Lamp5", "Pvalb", "Sncg", "Sst", "Vip"]


@dataclass
class PatchSeqDataLoader:
    data_dir: Path | str = Path("../data")
    min_samples: int = 50
    log1p: bool = True
    HVG: bool = True
    n_hvg: int = 8000

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.meta_path = self.data_dir / DEFAULT_META_FILENAME
        self.expr_path = self.data_dir / DEFAULT_EXPR_FILENAME

    def load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        meta = pd.read_csv(self.meta_path)
        expr = pd.read_csv(self.expr_path)

        expr_t = expr.T
        expr_t.columns = expr_t.iloc[0]
        expr_clean = expr_t.drop(expr_t.index[0])

        return meta, expr_clean

    def build_dataset(self) -> pd.DataFrame:
        meta, expr = self.load_raw()

        meta_matched = meta.set_index("transcriptomics_sample_id").loc[expr.index]
        meta_cols = meta_matched[META_COLUMNS].copy()
        expr_with_meta = pd.concat([expr, meta_cols], axis=1)

        expr_with_meta = expr_with_meta.drop(
            columns=[c for c in LEAKY_GENE_LABELS if c in expr_with_meta.columns]
        )
        expr_with_meta["corresponding_AIT2.3.1_alias"] = (
            expr_with_meta["corresponding_AIT2.3.1_alias"].str.split().str[0]
        )
        expr_with_meta = expr_with_meta.rename(
            columns={"corresponding_AIT2.3.1_alias": "cell_type"}
        )

        cell_type_counts = expr_with_meta["cell_type"].value_counts()
        valid_cell_types = cell_type_counts[cell_type_counts >= self.min_samples].index
        expr_with_meta = expr_with_meta[
            expr_with_meta["cell_type"].isin(valid_cell_types)
        ].copy()

        feature_cols = expr_with_meta.columns.difference(
            ["hemisphere", "donor_id", "transcriptomics_batch", "cell_type"]
        )
        feature_matrix = expr_with_meta.loc[:, feature_cols].to_numpy(dtype=float)

        if self.log1p:
            feature_matrix = np.log1p(feature_matrix)

        expr_with_meta.loc[:, feature_cols] = feature_matrix

        if self.HVG:
            expr_with_meta = self.select_hvg(expr_with_meta, n_hvg=self.n_hvg)

        return expr_with_meta

    def select_hvg(self, expr_with_meta: pd.DataFrame, *, n_hvg: int) -> pd.DataFrame:
        feature_cols = expr_with_meta.columns.difference(
            ["hemisphere", "donor_id", "transcriptomics_batch", "cell_type"]
        )
        X = expr_with_meta.loc[:, feature_cols].to_numpy(dtype=float)
        gene_mean = X.mean(axis=0)
        gene_std = X.std(axis=0)
        gene_cv = (gene_std / gene_mean)
        gene_cv[gene_mean == 0] = 0.0

        n_keep = min(n_hvg, len(feature_cols))
        hvg_idx = gene_cv.argsort()[::-1][:n_keep]
        selected_cols = list(feature_cols[hvg_idx])
        keep_cols = selected_cols + ["hemisphere", "donor_id", "transcriptomics_batch", "cell_type"]
        return expr_with_meta.loc[:, keep_cols].copy()
