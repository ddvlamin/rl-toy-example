import pandas as pd
from typing import Optional
import numpy as np

def subsample_data(df: pd.DataFrame, freq: str = 'H') -> pd.DataFrame:
    """
    Subsamples the price data to a lower frequency.

    Args:
        df (pd.DataFrame): DataFrame with 'timestamp' and 'close_eur' columns.
        freq (str): The target frequency (e.g., 'H' for hourly, 'D' for daily).

    Returns:
        pd.DataFrame: The subsampled DataFrame.
    """
    # Ensure 'timestamp' is the index for resampling
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    # Resample and take the last price in each new interval, then drop any NaNs
    df_resampled = df.resample(freq, label='right').last().dropna().reset_index()
    return df_resampled