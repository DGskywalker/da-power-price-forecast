"""Calendar, Holiday, DST, and Fourier Harmonic Features."""

from datetime import datetime
from typing import List, Optional
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


def get_german_public_holidays(years: List[int]) -> pd.DatetimeIndex:
    """Compute German national and major public holidays for given years."""
    holidays = []
    for y in years:
        # Fixed holidays
        holidays.extend([
            f"{y}-01-01",  # New Year
            f"{y}-05-01",  # Labour Day
            f"{y}-10-03",  # German Unity Day
            f"{y}-12-25",  # Christmas Day
            f"{y}-12-26",  # Boxing Day
        ])
        # Easter-based holidays (Gauss algorithm / approx)
        a = y % 19
        b = y // 100
        c = y % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        easter_sunday = pd.Timestamp(year=y, month=month, day=day)
        
        good_friday = easter_sunday - pd.Timedelta(days=2)
        easter_monday = easter_sunday + pd.Timedelta(days=1)
        ascension_day = easter_sunday + pd.Timedelta(days=39)
        whit_monday = easter_sunday + pd.Timedelta(days=50)

        for h_date in [good_friday, easter_monday, ascension_day, whit_monday]:
            holidays.append(h_date.strftime("%Y-%m-%d"))

    return pd.DatetimeIndex(pd.to_datetime(holidays, utc=True).normalize().unique())


def get_uk_bank_holidays(years: List[int]) -> pd.DatetimeIndex:
    """Compute UK bank holidays for given years."""
    holidays = []
    for y in years:
        # Fixed holidays
        holidays.extend([
            f"{y}-01-01",
            f"{y}-12-25",
            f"{y}-12-26",
        ])
        # Easter approximation
        a = y % 19
        b = y // 100
        c = y % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        easter_sunday = pd.Timestamp(year=y, month=month, day=day)
        good_friday = easter_sunday - pd.Timedelta(days=2)
        easter_monday = easter_sunday + pd.Timedelta(days=1)

        holidays.extend([good_friday.strftime("%Y-%m-%d"), easter_monday.strftime("%Y-%m-%d")])
        # Early May bank holiday: first Monday in May
        may_first = pd.Timestamp(year=y, month=5, day=1)
        days_to_monday = (7 - may_first.dayofweek) % 7
        early_may = may_first + pd.Timedelta(days=days_to_monday)
        holidays.append(early_may.strftime("%Y-%m-%d"))

        # Spring bank holiday: last Monday in May
        may_last = pd.Timestamp(year=y, month=5, day=31)
        spring_bank = may_last - pd.Timedelta(days=may_last.dayofweek)
        holidays.append(spring_bank.strftime("%Y-%m-%d"))

        # Summer bank holiday: last Monday in August
        aug_last = pd.Timestamp(year=y, month=8, day=31)
        summer_bank = aug_last - pd.Timedelta(days=aug_last.dayofweek)
        holidays.append(summer_bank.strftime("%Y-%m-%d"))

    return pd.DatetimeIndex(pd.to_datetime(holidays, utc=True).normalize().unique())


def add_calendar_features(df: pd.DataFrame, zone: str = "DE_LU") -> pd.DataFrame:
    """Add hour, weekday, month, DST, country holidays, and Fourier harmonic terms."""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)

    # Local timezone conversion for local time features
    tz_str = "Europe/Berlin" if zone == "DE_LU" else "Europe/London"
    local_ts = ts.dt.tz_convert(tz_str)

    # Core calendar features
    df["hour"] = local_ts.dt.hour
    df["dayofweek"] = local_ts.dt.dayofweek
    df["dayofyear"] = local_ts.dt.dayofyear
    df["month"] = local_ts.dt.month
    df["quarter"] = local_ts.dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # DST Flag: check if UTC offset differs from standard winter time
    # Germany: standard is UTC+1 (3600s), DST is UTC+2 (7200s)
    # UK: standard is UTC+0 (0s), DST is UTC+1 (3600s)
    offsets = [t.utcoffset().total_seconds() for t in local_ts]
    std_offset = 3600 if zone == "DE_LU" else 0
    df["is_dst"] = [1 if off > std_offset else 0 for off in offsets]

    # Holiday features
    years = sorted(ts.dt.year.unique().tolist())
    if zone == "DE_LU":
        h_dates = get_german_public_holidays(years)
    else:
        h_dates = get_uk_bank_holidays(years)

    norm_ts = ts.dt.normalize()
    is_h = norm_ts.isin(h_dates).astype(int)
    day_before_h = (norm_ts + pd.Timedelta(days=1)).isin(h_dates).astype(int)
    day_after_h = (norm_ts - pd.Timedelta(days=1)).isin(h_dates).astype(int)

    df["is_holiday"] = is_h.values
    df["day_before_holiday"] = day_before_h.values
    df["day_after_holiday"] = day_after_h.values

    # Fourier terms: Daily (24h period) and Weekly (168h period) with k=1, 2, 3 harmonics
    hour_val = df["hour"].values
    hour_of_week = (df["dayofweek"].values * 24 + hour_val)

    for k in [1, 2, 3]:
        df[f"fourier_daily_sin_{k}"] = np.sin(2 * np.pi * k * hour_val / 24.0)
        df[f"fourier_daily_cos_{k}"] = np.cos(2 * np.pi * k * hour_val / 24.0)
        df[f"fourier_weekly_sin_{k}"] = np.sin(2 * np.pi * k * hour_of_week / 168.0)
        df[f"fourier_weekly_cos_{k}"] = np.cos(2 * np.pi * k * hour_of_week / 168.0)

    return df
