import zipfile
import pandas as pd

def load_specific_csv_from_zip(zip_path, file_names):
    """
    Load specific CSV files from a ZIP archive into a dictionary of DataFrames.

    Parameters:
    - zip_path (str): Path to the ZIP file.
    - file_names (list): List of filenames to load (e.g., ["XBTUSD_1.csv"]).

    Returns:
    - dict: A dictionary where keys are filenames and values are DataFrames.
    """
    data_dict = {}
    column_names = ["timestamp", "open", "high", "low", "close", "volume", "trades"]

    with zipfile.ZipFile(zip_path, "r") as z:
        for file_name in file_names:
            if file_name in z.namelist():
                with z.open(file_name) as f:
                    # Read the CSV and specify column names
                    df = pd.read_csv(
                        f,
                        names=column_names,  # Add headers explicitly
                        header=None,         # Indicate no header row in CSV
                    )
                    # Ensure timestamps are numeric to avoid FutureWarning
                    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                    df.set_index("timestamp", inplace=True)
                    data_dict[file_name] = df
                    print(f"Loaded {file_name} with {len(df)} records.")
            else:
                print(f"File {file_name} not found in the ZIP archive.")

    return data_dict
