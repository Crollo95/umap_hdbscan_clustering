from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_data(df, scaling_method=None):
    """
    Apply selected scaling method to the dataframe.

    Parameters:
    - df: DataFrame to be scaled.
    - scaling_method: str or None, default None.
        Options: 'standard', 'minmax', or None

    Returns:
    - Scaled dataframe or original dataframe if scaling_method is None.
    """
    if scaling_method == 'standard':
        scaler = StandardScaler()
        return scaler.fit_transform(df)
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
        return scaler.fit_transform(df)
    elif scaling_method is None:
        return df
    else:
        raise ValueError("scaling_method should be 'standard', 'minmax', or None.")
