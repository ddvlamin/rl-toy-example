import pandas as pd
from pathlib import Path

def convert_usd_to_eur(crypto_df: pd.DataFrame, eur_usd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts crypto prices from USD to EUR using daily exchange rates.

    The function assumes the EUR/USD exchange rate means "how many USD for 1 EUR".
    For example, a rate of 1.08 means 1 EUR = 1.08 USD.

    Args:
        crypto_df (pd.DataFrame): DataFrame with crypto prices.
                                  Must contain 'open_time' (datetime) and 'close' (USD price).
        eur_usd_df (pd.DataFrame): DataFrame with daily EUR/USD exchange rates.
                                   Must contain 'date' (datetime, daily) and 'close' (EUR/USD rate).

    Returns:
        pd.DataFrame: The crypto DataFrame with an added 'close_eur' column.
    """
    # --- 1. Prepare DataFrames ---
    # Make a copy to avoid modifying the original DataFrame outside the function
    crypto_df = crypto_df.copy()

    # Ensure 'open_time' is in datetime format
    crypto_df['open_time'] = pd.to_datetime(crypto_df['open_time'])
    eur_usd_df['open_time'] = pd.to_datetime(eur_usd_df['open_time'])

    # --- 2. Align Exchange Rates with Crypto Data ---
    # Create a 'date' column in the crypto DataFrame to match the daily exchange rate data.
    crypto_df['date'] = crypto_df['open_time'].dt.date
    eur_usd_df['date'] = eur_usd_df['open_time'].dt.date

    # Create a mapping from date to exchange rate for efficient lookup
    exchange_rate_map = eur_usd_df.set_index('date')['close']

    # Add the corresponding daily exchange rate to each row of the crypto data
    crypto_df['eur_usd_rate'] = crypto_df['date'].map(exchange_rate_map)

    # --- 3. Handle Missing Exchange Rates (e.g., weekends/holidays) ---
    # Fill missing values (e.g., for weekends/holidays).
    # ffill() propagates the last valid observation forward, bfill() fills the rest backward.
    crypto_df['eur_usd_rate'] = crypto_df['eur_usd_rate'].ffill().bfill()

    if crypto_df['eur_usd_rate'].isnull().any():
        raise ValueError("Could not find exchange rates for all dates. Please check your EUR/USD data.")

    # --- 4. Perform the Conversion ---
    # Price_in_EUR = Price_in_USD / Exchange_Rate (where rate is USD per 1 EUR)
    crypto_df['close_eur'] = crypto_df['close'] / crypto_df['eur_usd_rate']

    # --- 5. Clean Up ---
    # Drop the temporary columns
    #crypto_df.drop(columns=['date', 'eur_usd_rate'], inplace=True)

    return crypto_df

if __name__ == '__main__':
    # Define file paths
    data_path = Path('data/')
    crypto_file = data_path / 'FORMUSDT_1m_1095days.tsv'
    eur_usd_file = data_path / 'EURUSDT_1day_1095days.tsv'
    output_file = data_path / 'FORMEUR_1m_1095days.tsv'

    # Load the datasets
    print(f"Loading crypto data from: {crypto_file}")
    crypto_prices_df = pd.read_csv(crypto_file, sep='\t')

    print(f"Loading EUR/USD data from: {eur_usd_file}")
    eur_usd_rates_df = pd.read_csv(eur_usd_file, sep='\t')

    # Perform the conversion
    print("\nConverting prices from USD to EUR...")
    crypto_prices_in_eur_df = convert_usd_to_eur(crypto_prices_df, eur_usd_rates_df)

    # Display the result
    print("\nConversion complete.")
    crypto_prices_in_eur_df.to_csv(output_file, sep='\t')