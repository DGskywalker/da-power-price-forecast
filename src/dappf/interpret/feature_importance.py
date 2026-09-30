"""Feature Importance and Cross-Zone Comparative Interpretability Analysis."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger
from dappf.models.gbdt import LightGBMForecaster


class FeatureImportanceAnalyzer:
    """Computes tree gain and permutation importance, providing comparative fundamental commentary."""

    def __init__(self, figures_dir: str | Path = "reports/figures"):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def compute_permutation_importance(
        self,
        forecaster: LightGBMForecaster,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        zone: str,
        n_repeats: int = 5,
        hour: int = 12,
    ) -> pd.DataFrame:
        """Calculate permutation feature importance on out-of-fold validation data."""
        logger.info(f"Computing permutation feature importance for {zone}...")
        if hour not in forecaster.models:
            hour = list(forecaster.models.keys())[0]

        model = forecaster.models[hour]
        mask = (X_val["hour"] == hour) if "hour" in X_val.columns else slice(None)
        X_h = X_val.loc[mask]
        y_h = y_val.loc[mask]

        if len(X_h) > 500:
            sample_idx = np.random.choice(len(X_h), 500, replace=False)
            X_h = X_h.iloc[sample_idx]
            y_h = y_h.iloc[sample_idx]

        r = permutation_importance(
            model, X_h, y_h, n_repeats=n_repeats, random_state=42, scoring="neg_mean_absolute_error"
        )

        df_imp = pd.DataFrame({
            "feature": X_h.columns,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

        # Plot top 15 features
        plt.figure(figsize=(10, 6))
        top_15 = df_imp.head(15)
        plt.barh(top_15["feature"][::-1], top_15["importance_mean"][::-1], xerr=top_15["importance_std"][::-1], color="#2b5c8f")
        plt.xlabel("Permutation Importance (Increase in MAE when shuffled)")
        plt.title(f"Permutation Feature Importance - {zone} (Hour {hour:02d}:00)", fontweight="bold")
        plt.tight_layout()
        plot_path = self.figures_dir / f"perm_importance_{zone}.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved permutation importance plot: {plot_path}")

        return df_imp

    @staticmethod
    def get_cross_zone_insights() -> str:
        """Domain analysis contrasting structural market dynamics between DE-LU and GB."""
        return (
            "Cross-Zone Structural Drivers:\n"
            "- Germany (DE-LU): Price formation is heavily dominated by the Merit-Order Effect of massive "
            "non-dispatchable wind (onshore/offshore) and solar penetration. Residual load is the paramount driver; "
            "periods of negative pricing occur during simultaneous high wind/solar and low weekend load. Lignite and coal "
            "provide baseload inertia, while gas CCGT acts as the marginal price-setting technology at peak hours.\n"
            "- Great Britain (GB): Post-Brexit separate bidding zone with high reliance on CCGT gas generation as the "
            "frequent marginal setter. Higher sensitivity to UK NBP / TTF gas spreads and UK ETS carbon prices. Interconnector "
            "flows (IFA, IFA2, BritNed, Nemo Link, North Sea Link) play an outsized balancing role, importing power from France/Norway "
            "when continental prices are low and exporting when GB wind is surplus."
        )
