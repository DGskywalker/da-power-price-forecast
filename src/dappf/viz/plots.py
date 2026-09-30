"""Visualizations: Forecast vs Actual, Error Heatmaps, Intervals, and Walk-Forward Traces."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


class Visualizer:
    """Generates publication-quality charts for power price forecasting reports."""

    def __init__(self, figures_dir: str | Path = "reports/figures"):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        # Apply standard quant styling
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams["font.sans-serif"] = "Helvetica, Arial, sans-serif"
        plt.rcParams["axes.edgecolor"] = "#cccccc"
        plt.rcParams["axes.linewidth"] = 0.8

    def plot_forecast_vs_actual_overlay(
        self,
        df_eval: pd.DataFrame,
        zone: str,
        model_name: str,
        target_col: str,
        days: int = 14,
    ) -> Path:
        """Plot actuals vs model forecast with 80% confidence interval over recent days."""
        df_sub = df_eval.tail(days * 24).copy().reset_index(drop=True)
        ts = pd.to_datetime(df_sub["timestamp"])
        y_true = df_sub[target_col].values
        y_pred = df_sub[f"pred_{model_name}"].values

        plt.figure(figsize=(14, 6))
        plt.plot(ts, y_true, label="Actual DA Price", color="#1f2937", linewidth=1.8, alpha=0.9)
        plt.plot(ts, y_pred, label=f"Forecast ({model_name})", color="#2563eb", linewidth=1.8, linestyle="--")

        # 80% prediction interval if available
        if f"q10_{model_name}" in df_sub.columns and f"q90_{model_name}" in df_sub.columns:
            q10 = df_sub[f"q10_{model_name}"].values
            q90 = df_sub[f"q90_{model_name}"].values
            plt.fill_between(ts, q10, q90, color="#93c5fd", alpha=0.35, label="80% Prediction Interval [q10, q90]")

        currency = "EUR/MWh" if zone == "DE_LU" else "GBP/MWh"
        plt.title(f"Day-Ahead Electricity Price Forecast vs Actual - {zone} (Last {days} Days)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Delivery Timestamp (UTC)", fontsize=11)
        plt.ylabel(f"Price ({currency})", fontsize=11)
        plt.legend(frameon=True, facecolor="white", edgecolor="#e5e7eb", loc="upper right")
        plt.tight_layout()

        out_path = self.figures_dir / f"forecast_vs_actual_{zone}_{model_name}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_hourly_mae_heatmap(
        self,
        df_eval: pd.DataFrame,
        zone: str,
        model_name: str,
        target_col: str,
    ) -> Path:
        """Plot MAE heatmap across delivery Hour of Day vs Day of Week."""
        df = df_eval.copy()
        ts = pd.to_datetime(df["timestamp"])
        df["hour"] = ts.dt.hour
        df["dayofweek"] = ts.dt.day_name()
        df["abs_error"] = np.abs(df[target_col] - df[f"pred_{model_name}"])

        pivot = df.pivot_table(index="dayofweek", columns="hour", values="abs_error", aggfunc="mean")
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = pivot.reindex([d for d in dow_order if d in pivot.index])

        plt.figure(figsize=(12, 5))
        sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={"label": "Mean Absolute Error"})
        plt.title(f"MAE by Hour of Day & Day of Week - {zone} ({model_name})", fontsize=12, fontweight="bold", pad=10)
        plt.xlabel("Delivery Hour (0-23 UTC)", fontsize=10)
        plt.ylabel("Day of Week", fontsize=10)
        plt.tight_layout()

        out_path = self.figures_dir / f"hourly_mae_heatmap_{zone}_{model_name}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_walk_forward_mae_evolution(
        self,
        fold_results: pd.DataFrame,
        zone: str,
    ) -> Path:
        """Plot out-of-sample MAE evolution across expanding-window monthly folds."""
        plt.figure(figsize=(12, 5))
        for model_name, grp in fold_results.groupby("model"):
            grp_sorted = grp.sort_values("fold_idx")
            plt.plot(grp_sorted["test_start"].astype(str), grp_sorted["mae"], marker="o", linewidth=2, label=model_name)

        currency = "EUR/MWh" if zone == "DE_LU" else "GBP/MWh"
        plt.title(f"Walk-Forward Out-of-Sample MAE per Month - {zone}", fontsize=13, fontweight="bold", pad=10)
        plt.xlabel("Walk-Forward Test Month", fontsize=11)
        plt.ylabel(f"MAE ({currency})", fontsize=11)
        plt.xticks(rotation=45)
        plt.legend(frameon=True, facecolor="white", edgecolor="#e5e7eb")
        plt.tight_layout()

        out_path = self.figures_dir / f"walk_forward_mae_{zone}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_price_regime_errors(
        self,
        df_eval: pd.DataFrame,
        zone: str,
        target_col: str,
        model_names: List[str],
    ) -> Path:
        """Analyze MAE partitioned into Low (<30), Mid (30-120), and High (>120) price regimes."""
        df = df_eval.copy()
        prices = df[target_col]
        regimes = pd.cut(
            prices,
            bins=[-np.inf, 30.0, 120.0, np.inf],
            labels=["Low (<30)", "Mid (30-120)", "High Scarcity (>120)"],
        )
        df["regime"] = regimes

        regime_stats = []
        for m in model_names:
            col = f"pred_{m}"
            if col in df.columns:
                df[f"err_{m}"] = np.abs(df[target_col] - df[col])
                means = df.groupby("regime", observed=False)[f"err_{m}"].mean()
                for reg, val in means.items():
                    regime_stats.append({"model": m, "regime": reg, "mae": val})

        df_reg = pd.DataFrame(regime_stats)
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_reg, x="regime", y="mae", hue="model", palette="Blues_d")
        currency = "EUR/MWh" if zone == "DE_LU" else "GBP/MWh"
        plt.title(f"Forecasting Error by Market Price Regime - {zone}", fontsize=12, fontweight="bold")
        plt.xlabel("Market Regime", fontsize=11)
        plt.ylabel(f"MAE ({currency})", fontsize=11)
        plt.legend(frameon=True, facecolor="white", edgecolor="#e5e7eb")
        plt.tight_layout()

        out_path = self.figures_dir / f"regime_errors_{zone}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return out_path
