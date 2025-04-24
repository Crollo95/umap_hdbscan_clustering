import pandas as pd
import csv

def load_data(file_path: str, sep: str = None, index_col: int = None) -> pd.DataFrame:
    """
    Load data from a CSV (or TXT) file into a DataFrame.
    
    If `sep` is not provided, the function attempts to auto-detect the delimiter (comma or tab).
    If `index_col` is not provided, the function checks the header: if the first column header 
    starts with 'Unnamed:', it assumes that column is an index and sets index_col=0.
    
    Parameters:
        file_path (str): Path to the CSV/TXT file.
        sep (str, optional): Delimiter used in the file. Auto-detected if not provided.
        index_col (int, optional): Column to use as the row labels of the DataFrame.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    
    # Auto-detect delimiter if not provided
    if sep is None:
        with open(file_path, 'r') as f:
            sample = f.read(1024)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[',', '\t'])
            sep = dialect.delimiter
        except csv.Error:
            sep = ','
    
    # Auto-detect if the first column should be used as an index
    if index_col is None:
        try:
            df_header = pd.read_csv(file_path, sep=sep, nrows=0)
            first_col = df_header.columns[0]
            if (isinstance(first_col, str) and first_col.startswith("Unnamed:")) or 'id' in first_col.lower():
                index_col = 0
        except Exception:
            # If header reading fails, do nothing and leave index_col as None
            pass
    
    return pd.read_csv(file_path, sep=sep, index_col=index_col)
