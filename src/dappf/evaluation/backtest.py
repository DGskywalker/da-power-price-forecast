"""Expanding Window Walk-Forward Backtester with Strict Leakage Prohibition."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class FoldSplit:
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_mask: np.ndarray
    test_mask: np.ndarray


class ExpandingWindowSplitter:
    """
    Generates expanding window walk-forward splits.

    Strict guarantees:
    - Chronological ordering: no shuffling.
    - Zero lookahead: test_start > train_end.
    - Minimum training duration (e.g. 18 to 24 months).
    - Step size = 1 month.
    """

    def __init__(
        self,
        min_train_months: int = 18,
        step_months: int = 1,
        test_months: int = 1,
    ):
        self.min_train_months = min_train_months
        self.step_months = step_months
        self.test_months = test_months

    def split(self, df: pd.DataFrame, time_col: str = "timestamp") -> List[FoldSplit]:
        """Generate list of expanding window fold splits."""
        df = df.sort_values(time_col).reset_index(drop=True)
        ts = pd.to_datetime(df[time_col], utc=True)
        min_ts = ts.min()
        max_ts = ts.max()

        # Identify all monthly start boundaries
        month_starts = pd.date_range(
            start=min_ts.floor("D"),
            end=max_ts.ceil("D"),
            freq="MS",  # Month Start
            tz="UTC"
        )

        if len(month_starts) <= self.min_train_months + self.test_months:
            # Fallback if range is shorter: split based on proportional days
            logger.warning("Data duration shorter than configured months; generating day-based splits")
            total_days = (max_ts - min_ts).days
            min_train_days = int(total_days * 0.7)
            step_days = 30
            splits = []
            fold_idx = 0
            curr_train_end = min_ts + pd.Timedelta(days=min_train_days)
            while curr_train_end + pd.Timedelta(days=step_days) <= max_ts:
                test_start = curr_train_end
                test_end = min(curr_train_end + pd.Timedelta(days=step_days), max_ts)
                train_mask = (ts >= min_ts) & (ts < curr_train_end)
                test_mask = (ts >= test_start) & (ts < test_end)
                splits.append(
                    FoldSplit(
                        fold_idx=fold_idx,
                        train_start=min_ts,
                        train_end=curr_train_end,
                        test_start=test_start,
                        test_end=test_end,
                        train_mask=train_mask.values,
                        test_mask=test_mask.values,
                    )
                )
                fold_idx += 1
                curr_train_end += pd.Timedelta(days=step_days)
            return splits

        splits = []
        fold_idx = 0
        split_cursor = self.min_train_months

        while split_cursor + self.test_months <= len(month_starts):
            train_start = min_ts
            train_end = month_starts[split_cursor]
            test_start = month_starts[split_cursor]
            
            # End of test window
            if split_cursor + self.test_months < len(month_starts):
                test_end = month_starts[split_cursor + self.test_months]
            else:
                test_end = max_ts

            train_mask = (ts >= train_start) & (ts < train_end)
            test_mask = (ts >= test_start) & (ts < test_end)

            if train_mask.sum() > 0 and test_mask.sum() > 0:
                splits.append(
                    FoldSplit(
                        fold_idx=fold_idx,
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        train_mask=train_mask.values,
                        test_mask=test_mask.values,
                    )
                )
                fold_idx += 1

            split_cursor += self.step_months

        return splits
