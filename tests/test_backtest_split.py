"""Unit tests for expanding window walk-forward backtest splitting."""

import numpy as np
import pandas as pd
import pytest
from dappf.evaluation.backtest import ExpandingWindowSplitter


def test_expanding_window_split_properties():
    # 3 years of synthetic hourly timestamps (approx 36 months)
    dates = pd.date_range("2021-01-01 00:00:00", "2023-12-31 23:00:00", freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp": dates, "val": np.random.randn(len(dates))})

    splitter = ExpandingWindowSplitter(min_train_months=24, step_months=1, test_months=1)
    splits = splitter.split(df, time_col="timestamp")

    assert len(splits) > 0, "Splitter should produce at least one fold"

    prev_train_len = 0
    prev_test_end = None

    for s in splits:
        # 1. No overlap between train and test
        train_times = df.loc[s.train_mask, "timestamp"]
        test_times = df.loc[s.test_mask, "timestamp"]

        assert train_times.max() < test_times.min(), (
            f"Fold {s.fold_idx}: Leakage detected! Train max {train_times.max()} >= Test min {test_times.min()}"
        )

        # 2. Expanding window: train set must grow strictly across folds
        curr_train_len = len(train_times)
        assert curr_train_len > prev_train_len, (
            f"Fold {s.fold_idx}: Train set length did not expand ({curr_train_len} <= {prev_train_len})"
        )
        prev_train_len = curr_train_len

        # 3. Contiguity: test start corresponds to train end
        assert s.train_end == s.test_start

        # 4. Successive test folds do not overlap
        if prev_test_end is not None:
            assert s.test_start >= prev_test_end
        prev_test_end = s.test_end
