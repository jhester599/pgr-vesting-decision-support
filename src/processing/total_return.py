"""
True DRIP total return reconstruction using unadjusted prices.

This module implements the mathematically correct method for simulating a
buy-and-hold position that:
  1. Receives dividends and immediately reinvests them as fractional shares
     at the unadjusted closing price on the ex-dividend date.
  2. Has its share count forward-multiplied on each stock split date.

This differs from adjusted-close total return in that it accurately
represents the exact capital required at each historical date and the exact
fractional share accumulation, which matters for modeling employer RSU
accumulation over a 10–15 year horizon.

Key formula:
  On ex-div date t_d:
    new_shares = (shares_held[t_d] * div_per_share[t_d]) / close_price[t_d]
    shares_held[t > t_d] += new_shares

  Total return over [t_0, t_T]:
    TR = (shares_held[t_T] * price[t_T]) / (shares_held[t_0] * price[t_0]) - 1
"""

import pandas as pd
import numpy as np


def build_position_series(
    price_history: pd.DataFrame,
    dividend_history: pd.DataFrame,
    split_history: pd.DataFrame,
    initial_shares: float = 1.0,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Simulate a DRIP position from ``start_date`` to ``end_date``.

    Applies splits and dividend reinvestments chronologically to produce a
    daily series of (shares_held, portfolio_value).

    Args:
        price_history:   DataFrame from price_loader.load() (DatetimeIndex, close column).
        dividend_history: DataFrame from dividend_loader.load() (DatetimeIndex, dividend col).
        split_history:   DataFrame from split_loader.load() (DatetimeIndex, split_ratio col).
        initial_shares:  Starting share count at ``start_date``.
        start_date:      First date of the simulation (inclusive). Defaults to the
                         earliest date in price_history.
        end_date:        Last date of the simulation (inclusive). Defaults to the
                         latest date in price_history.

    Returns:
        DataFrame with DatetimeIndex (business days in [start_date, end_date]) and columns:
          - shares_held (float64): Cumulative shares after splits + DRIP
          - close_price (float64): Unadjusted closing price
          - portfolio_value (float64): shares_held * close_price
    """
    if start_date is None:
        start_date = price_history.index.min()
    if end_date is None:
        end_date = price_history.index.max()

    prices = price_history.loc[start_date:end_date, "close"].copy()
    if prices.empty:
        raise ValueError(
            f"No price data between {start_date.date()} and {end_date.date()}."
        )

    # Build output DataFrame; shares_held starts at initial_shares.
    df = pd.DataFrame({"close_price": prices})
    df["shares_held"] = np.nan
    df.iloc[0, df.columns.get_loc("shares_held")] = float(initial_shares)

    # Collect all corporate actions in the window, sorted chronologically.
    dividends_window = dividend_history.loc[
        (dividend_history.index >= start_date) & (dividend_history.index <= end_date)
    ]
    splits_window = split_history.loc[
        (split_history.index >= start_date) & (split_history.index <= end_date)
    ]

    # Merge all events into a single timeline.
    events: list[tuple[pd.Timestamp, str, float]] = []
    for dt, row in dividends_window.iterrows():
        events.append((dt, "div", row["dividend"]))
    for dt, row in splits_window.iterrows():
        events.append((dt, "split", row["split_ratio"]))
    events.sort(key=lambda x: x[0])

    # Forward-fill shares day by day, applying events when encountered.
    dates = df.index.tolist()
    current_shares = float(initial_shares)
    event_idx = 0
    n_events = len(events)

    for i, date in enumerate(dates):
        # Apply any events on or before this date that haven't been applied yet.
        while event_idx < n_events and events[event_idx][0] <= date:
            evt_date, evt_type, evt_value = events[event_idx]
            if evt_type == "split":
                current_shares *= evt_value
            elif evt_type == "div":
                # Reinvest dividend: buy fractional shares at closing price on ex-div date.
                # We use the close price on the event date (or nearest prior date).
                if evt_date in df.index:
                    div_price = df.at[evt_date, "close_price"]
                else:
                    # Use the most recent available price before the event.
                    prior = df.loc[:evt_date, "close_price"].dropna()
                    div_price = prior.iloc[-1] if not prior.empty else np.nan
                if div_price and div_price > 0:
                    new_shares = (current_shares * evt_value) / div_price
                    current_shares += new_shares
            event_idx += 1

        df.iloc[i, df.columns.get_loc("shares_held")] = current_shares

    df["portfolio_value"] = df["shares_held"] * df["close_price"]
    return df


def compute_total_return(
    position: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """
    Compute the DRIP total return over a sub-period of a position series.

    Args:
        position:   DataFrame from build_position_series().
        start_date: Start of the return window (inclusive).
        end_date:   End of the return window (inclusive).

    Returns:
        Total return as a decimal (e.g. 0.25 means +25%).

    Raises:
        KeyError: If start_date or end_date are not in the position index.
    """
    # Use .asof() to handle weekends/holidays gracefully.
    start_val = position["portfolio_value"].asof(start_date)
    end_val = position["portfolio_value"].asof(end_date)

    if pd.isna(start_val) or start_val == 0:
        raise ValueError(f"Portfolio value at {start_date.date()} is zero or NaN.")
    if pd.isna(end_val):
        raise ValueError(f"Portfolio value at {end_date.date()} is NaN.")

    return float(end_val / start_val) - 1.0


def build_monthly_returns(
    price_history: pd.DataFrame,
    dividend_history: pd.DataFrame,
    split_history: pd.DataFrame,
    forward_months: int = 6,
) -> pd.Series:
    """
    Build a monthly series of forward total returns for use as the ML target.

    For each month-end date t, computes the DRIP total return from t to
    t + ``forward_months`` calendar months. Observations where the forward
    window extends beyond available data are NaN (no leakage).

    Args:
        price_history:    DataFrame from price_loader.load().
        dividend_history: DataFrame from dividend_loader.load().
        split_history:    DataFrame from split_loader.load().
        forward_months:   Number of months in the forward return window.

    Returns:
        Series indexed by month-end date with forward total return values.
        Name: ``target_{forward_months}m_return``.
    """
    # Build the full position series once (1 share initial, whole history).
    full_position = build_position_series(
        price_history, dividend_history, split_history, initial_shares=1.0
    )

    # Resample to month-end business days.
    monthly_dates = full_position.resample("BME").last().index

    returns: dict[pd.Timestamp, float] = {}
    data_end = full_position.index.max()

    for t in monthly_dates:
        t_end = t + pd.DateOffset(months=forward_months)
        if t_end > data_end:
            returns[t] = np.nan
            continue
        try:
            tr = compute_total_return(full_position, t, t_end)
            returns[t] = tr
        except (ValueError, KeyError):
            returns[t] = np.nan

    result = pd.Series(returns, name=f"target_{forward_months}m_return")
    result.index.name = "date"
    return result
