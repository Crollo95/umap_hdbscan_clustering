from sklearn.preprocessing import StandardScaler

def scale_data(df):
    """Apply standard scaling to the dataframe."""
    scaler = StandardScaler()
    return scaler.fit_transform(df)

