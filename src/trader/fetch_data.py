import pandas as pd
from binance.client import Client
import os

def fetch_and_save_binance_data(
    symbol='BTCUSDT', 
    interval=Client.KLINE_INTERVAL_1MINUTE, 
    start_date="1 day ago UTC", 
    output_folder="data",
    filename=None,
    save_file=True
):
    """
    Fetches historical klines from Binance, saves them to a tab-separated file,
    and returns them as a pandas DataFrame.

    :param symbol: The symbol to fetch data for (e.g., 'BTCUSDT').
    :param interval: The interval for the klines (e.g., Client.KLINE_INTERVAL_1MINUTE).
    :param start_date: The start date to fetch data from (e.g., "1 day ago UTC").
                       Be careful with long periods for minute-data as it can be very large.
    :param output_folder: The folder to save the data file in.
    :param filename: The name of the output file. If None, a default is generated.
    :param save_file: If True, saves the fetched data to a file.
    :return: A pandas DataFrame with the historical data, or None if fetching fails.
    """
    # Initialize the client (no API key needed for public data)
    client = Client()

    print(f"Fetching {interval} data for {symbol} from {start_date}...")

    # Use the klines generator to fetch all data since the start_date
    klines_generator = client.get_historical_klines_generator(symbol, interval, start_date)

    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    
    df = pd.DataFrame(klines_generator, columns=columns)

    if df.empty:
        print(f"No data found for {symbol}.")
        return None

    # --- Data Cleaning and Formatting ---
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.set_index('open_time', inplace=True)

    print(f"Successfully fetched {len(df)} data points for {symbol}.")

    if save_file:
        # --- Saving the data ---
        if filename is None:
            # Generate a descriptive filename
            filename = f"{symbol}_{interval}_{start_date.replace(' ', '').replace(',', '')}.tsv"

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        filepath = os.path.join(output_folder, filename)

        # Save to a tab-separated file
        df.to_csv(filepath, sep='\t')
        print(f"Data successfully saved to {filepath}")
    
    return df


if __name__ == '__main__':
    period_in_days = 3 * 365
    symbol = 'FORMUSDT'
    interval = Client.KLINE_INTERVAL_1MINUTE
    start_date = f"{period_in_days} days ago UTC"
    output_folder = "data"

    # --- Example 1: Fetch 1-minute data for the last 3 days for BTC ---
    print("--- Fetching 1-minute BTC data ---")
    fetch_and_save_binance_data(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        filename=f"{symbol}_{interval}_{period_in_days}days.tsv"
    )
    print("-" * 30)
