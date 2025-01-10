import pandas as pd
import time
import os
from utils.api_client import APIClient
from utils.load_keys import load_secrets


def fetch_historical_data_backward(client: APIClient, pair: str, interval: int, years: int=2) -> pd.DataFrame:
    """
    Fetch historical OHLC data for a given pair, iterating backward.

    Parameters:
    - client: Kraken APIClient instance.
    - pair: Trading pair (e.g., "XBT/USD").
    - interval: Time interval in minutes (e.g., 1, 5).
    - years: Number of years of historical data to fetch.

    Returns:
    - pd.DataFrame: Combined historical OHLC data.
    """
    all_data = []
    end_time = int(time.time()) - years * 365 * 24 * 60 * 60 
    current_since = None  # Start with the most recent data
    print(f"Fetching data for {pair} from {pd.to_datetime(end_time, unit='s')} backward.")

    while True:
        response = client.get_ohlc_data(pair=pair, interval=interval, since=current_since)

        # Handle Kraken's pair format differences - they chance btc to xbt for example
        if pair == "XBTUSD":
            ohlc_data = response["result"]["XXBTZUSD"]
        else:
            ohlc_data = response["result"][pair]

        if not ohlc_data:
            print("No more data returned; exiting loop.")
            break

        all_data.extend(ohlc_data)
        print(f"Fetched {len(all_data)} records so far...")
        print(f"Batch start: {pd.to_datetime(ohlc_data[0][0], unit='s')}, "
              f"Batch end: {pd.to_datetime(ohlc_data[-1][0], unit='s')}")

        if int(ohlc_data[0][0]) <= end_time:
            print("Reached the specified historical range; exiting loop.")
            break

        # Move backward by setting `since` to the timestamp of the first record in the batch
        current_since = int(ohlc_data[0][0]) - 1
        time.sleep(3)  # to respect API rate limits
        # TODO: this didnt work, the cap remains in place and cant be batch called like this

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("timestamp", inplace=True)
    return df


def compute_metrics(data: pd.DataFrame, window: int=20) -> pd.DataFrame:
    """
    Compute rolling metrics for the given data.

    Parameters:
    - data: Historical OHLC data.
    - window: Rolling window size.

    Returns:
    - pd.DataFrame: Data with added rolling metrics.
    """
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data.dropna(subset=["close"], inplace=True)

    # Calculate rolling metrics
    data["SMA"] = data["close"].rolling(window=window).mean()
    data["EMA"] = data["close"].ewm(span=window).mean()
    data["RollingHigh"] = data["high"].rolling(window=window).max()
    data["RollingLow"] = data["low"].rolling(window=window).min()
    data["StdDev"] = data["close"].rolling(window=window).std()
    data["UpperBand"] = data["SMA"] + 2 * data["StdDev"]
    data["LowerBand"] = data["SMA"] - 2 * data["StdDev"]

    # Compute RSI
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    data["RSI"] = 100 - (100 / (1 + gain / loss))

    return data


def process_pairs(client: APIClient, pairs: list, interval: int, years: int) -> dict:
    """
    Generate historical data with metrics for multiple trading pairs.

    Parameters:
    - client (APIClient): Kraken APIClient instance.
    - pairs (list): List of trading pairs (e.g., ["XBT/USD", "ETH/USD"]).
    - interval (int): Time interval in minutes (e.g., 1, 5).
    - years (int): Number of years of historical data.

    Returns:
    - dict: Dictionary of DataFrames, one per trading pair.
    """
    data_dict = {}
    # since = int(time.time()) - years * 365 * 24 * 60 * 60  # Start timestamp

    for pair in pairs:
        try:
            raw_data = fetch_historical_data_backward(client, pair, interval, years)
            data_with_metrics = compute_metrics(raw_data)
            data_dict[pair] = data_with_metrics
        except Exception as e:
            print(f"Error processing {pair}: {e}")

    return data_dict


def save_data(data_dict: dict, folder: str="./data/processed") -> None:
    """
    Save processed data to CSV files.

    Parameters:
    - data_dict: Dictionary of DataFrames, one per trading pair.
    - folder: Folder to save the files.
    """
    os.makedirs(folder, exist_ok=True)

    for pair, data in data_dict.items():
        file_name = pair.replace("/", "_") + ".csv"
        data.to_csv(os.path.join(folder, file_name), index=True, encoding="utf-8")
        print(f"Saved {pair} data to {file_name}")


if __name__ == "__main__":
    secrets = load_secrets()
    client = APIClient("https://api.kraken.com/", secrets["rest_api_key"], secrets["rest_api_secret"])

    pairs = ["XBTUSD"]
    interval = 1  # interval in mins
    years = 2  # years of historical data (doesnt allow. restricted by max data limit 720 records so wont go beyond the last 720 records of a given time interval)

    historical_data = process_pairs(client, pairs, interval, years)

    save_data(historical_data) # no longer needed. kraken provides zip of all coins and several time frames
