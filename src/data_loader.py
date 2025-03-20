import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV (or TXT) file into a DataFrame."""
    return pd.read_csv(file_path, sep='\t')

