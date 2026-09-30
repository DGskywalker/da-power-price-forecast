"""SHAP (SHapley Additive exPlanations) Interpretability Analysis for Electricity Price Forecasts."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from loguru import logger
from dappf.models.gbdt import LightGBMForecaster


class ShapInterpreter:
    """Computes Shapley values, generates global and local explanations for LightGBM models."""

    def __init__(self, figures_dir: str | Path = "reports/figures"):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        forecaster: LightGBMForecaster,
        X_sample: pd.DataFrame,
        zone: str,
        hour: int = 12,
        n_samples: int = 400
    ) -> Dict[str, Any]:
        """
        Run TreeExplainer on the hour h model (typically noon peak or evening peak).
        
        Saves:
        - Summary plot
        - Top dependence plots
        - Local waterfall / force plot for the highest price spike
        """
        logger.info(f"Computing SHAP values for zone {zone} (Hour {hour})...")
        if hour not in forecaster.models:
            hour = list(forecaster.models.keys())[0]

        model = forecaster.models[hour]
        
        # Take representative sample for fast, robust computation
        if len(X_sample) > n_samples:
            sample_df = X_sample.sample(n=n_samples, random_state=42)
        else:
            sample_df = X_sample

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(sample_df)

        # 1. Global Summary Plot (Beeswarm)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, sample_df, show=False, max_display=15)
        plt.title(f"SHAP Feature Importance (Beeswarm) - {zone} Hour {hour:02d}:00", fontsize=12, fontweight="bold")
        plt.tight_layout()
        summary_path = self.figures_dir / f"shap_summary_{zone}.png"
        plt.savefig(summary_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP summary plot: {summary_path}")

        # 2. Dependence Plots for Top 2 Features
        # Compute mean absolute SHAP values across features
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:2]
        feature_names = sample_df.columns

        dep_paths = []
        for rank, idx in enumerate(top_indices):
            feat_name = feature_names[idx]
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(
                feat_name,
                shap_values.values,
                sample_df,
                show=False,
                interaction_index="auto"
            )
            plt.title(f"SHAP Dependence: {feat_name} ({zone})", fontsize=11, fontweight="bold")
            plt.tight_layout()
            dep_path = self.figures_dir / f"shap_dep_{zone}_{rank+1}_{feat_name}.png"
            plt.savefig(dep_path, dpi=200, bbox_inches="tight")
            plt.close()
            dep_paths.append(dep_path)

        # 3. Local Explanation for Peak Price Observation
        preds = model.predict(sample_df)
        max_idx = int(np.argmax(preds))
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[max_idx], max_display=10, show=False)
        plt.title(f"SHAP Local Explanation: Highest Price Day ({zone} Pred: {preds[max_idx]:.1f})", fontsize=11, fontweight="bold")
        plt.tight_layout()
        waterfall_path = self.figures_dir / f"shap_waterfall_spike_{zone}.png"
        plt.savefig(waterfall_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP local spike explanation: {waterfall_path}")

        return {
            "summary_path": str(summary_path),
            "dependence_paths": [str(p) for p in dep_paths],
            "waterfall_path": str(waterfall_path),
            "top_features": [feature_names[i] for i in top_indices],
        }
