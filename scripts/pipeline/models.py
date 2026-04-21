from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .data import PatchSeqDataLoader
from .train import (
    cross_validated_ensemble,
    make_pca_lr_pipeline,
    make_pca_nb_pipeline,
    make_pca_svm_pipeline,
    prepare_dataset,
    run_xgboost,
    summarize_results,
)


@dataclass
class DATE:
    data_dir: Path | str = Path("../data")
    min_samples: int = 50
    log1p: bool = True
    HVG: bool = True
    n_hvg: int = 8000
    test_size: float = 0.3
    random_state: int = 42
    n_splits: int = 3
    expr_with_meta: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        loader = PatchSeqDataLoader(
            data_dir=self.data_dir,
            min_samples=self.min_samples,
            log1p=self.log1p,
            HVG=self.HVG,
            n_hvg=self.n_hvg,
        )
        self.expr_with_meta = loader.build_dataset()

    def dataset(self) -> pd.DataFrame:
        return self.expr_with_meta.copy()

    def perform_PCA_LR(self, *, n_components: int = 50) -> dict[str, Any]:
        split = prepare_dataset(
            self.expr_with_meta,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        pipeline = make_pca_lr_pipeline(
            n_components=n_components,
            random_state=self.random_state,
        )
        return cross_validated_ensemble(
            pipeline,
            split,
            model_name="PCA+LR",
            random_state=self.random_state,
            n_splits=self.n_splits,
        )

    def perform_PCA_SVM(self, *, n_components: int = 50) -> dict[str, Any]:
        split = prepare_dataset(
            self.expr_with_meta,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        pipeline = make_pca_svm_pipeline(
            n_components=n_components,
            random_state=self.random_state,
        )
        return cross_validated_ensemble(
            pipeline,
            split,
            model_name="PCA+SVM",
            random_state=self.random_state,
            n_splits=self.n_splits,
        )

    def perform_PCA_NB(self, *, n_components: int = 50) -> dict[str, Any]:
        split = prepare_dataset(
            self.expr_with_meta,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        pipeline = make_pca_nb_pipeline(
            n_components=n_components,
            random_state=self.random_state,
        )
        return cross_validated_ensemble(
            pipeline,
            split,
            model_name="PCA+NaiveBayes",
            random_state=self.random_state,
            n_splits=self.n_splits,
        )

    def perform_XGBoost(self, *, device: str = "cuda") -> dict[str, Any]:
        return run_xgboost(
            self.expr_with_meta,
            test_size=self.test_size,
            random_state=self.random_state,
            n_splits=self.n_splits,
            device=device,
        )

    def run_all(self, *, n_components: int = 50, xgb_device: str = "cuda") -> dict[str, Any]:
        results = [
            self.perform_PCA_LR(n_components=n_components),
            self.perform_PCA_SVM(n_components=n_components),
            self.perform_PCA_NB(n_components=n_components),
            self.perform_XGBoost(device=xgb_device),
        ]
        return {
            "results": results,
            "summary": summarize_results(results),
        }
