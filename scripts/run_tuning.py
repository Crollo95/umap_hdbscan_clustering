#!/usr/bin/env python
import argparse
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import scale_data
from src.tuning import tune_hyperparameters

def main():
    parser = argparse.ArgumentParser(description="Run UMAP + HDBSCAN hyperparameter tuning")
    parser.add_argument("--data", type=str, required=True, help="Path to data file (CSV/TXT)")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file for tuning results")
    args = parser.parse_args()

    df = load_data(args.data)
    data_scaled = scale_data(df)

    results = tune_hyperparameters(data_scaled)
    results_df = pd.DataFrame(results)

    # Example metric to sort by: silhouette_score adjusted by noise proportion
    results_df['silhouette_x_noise'] = results_df['silhouette_score'] * (1 - results_df['noise_proportion'])
    results_df_sorted = results_df.sort_values(by='silhouette_x_noise', ascending=False)
    results_df_sorted.to_csv(args.output, index=False)
    print("Tuning completed. Results saved to:", args.output)

if __name__ == "__main__":
    main()

