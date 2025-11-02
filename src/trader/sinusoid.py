#required imports
import numpy as np
import pandas as pd
from typing import Optional

def generate_trending_sinusoid(
    num_days: int = 365,
    initial_price: float = 100.0,
    trend_per_day: float = 0.1,
    monthly_amplitude: float = 10.0,
    noise_std: float = 0.5
) -> pd.DataFrame:
    """
    Generates a sinusoidal wave with a 1-minute sample rate, a monthly frequency,
    and a linear upward trend.

    Args:
        num_days (int): The total number of days for the data.
        initial_price (float): The starting price of the series.
        trend_per_day (float): The average daily increase in price for the trend.
        monthly_amplitude (float): The amplitude of the monthly sine wave.
        noise_std (float): The standard deviation of the random noise to add.

    Returns:
        pd.DataFrame: A DataFrame with 'timestamp' and 'price' columns.
    """
    # 1. Define the time range at a 1-minute sample rate
    total_minutes = num_days * 24 * 60
    t = np.arange(total_minutes)

    # 2. Create the upward trend line
    trend_per_minute = trend_per_day / (24 * 60)
    trend = initial_price + t * trend_per_minute

    # 3. Create the sinusoidal component with a ~1-month frequency
    # Period T = number of minutes in a month
    minutes_in_month = 30 * 24 * 60
    sinusoid = monthly_amplitude * np.sin(2 * np.pi * t / minutes_in_month)

    # 4. Create some random noise
    noise = np.random.normal(0, noise_std, total_minutes)

    # 5. Combine the components
    prices = trend + sinusoid #+ noise

    # 6. Create a pandas DataFrame with a datetime index
    timestamps = pd.to_datetime(pd.date_range(start='2023-01-01', periods=total_minutes, freq='T'))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close_eur': prices
    })
    
    return df