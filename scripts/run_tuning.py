import argparse
import os
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import scale_data
from src.tuning import tune_hyperparameters

def main():
    parser = argparse.ArgumentParser(description="Run UMAP + HDBSCAN hyperparameter tuning")
    parser.add_argument("--data", type=str, required=True, help="Path to data file (CSV/TXT)")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file for tuning results")
    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = load_data(args.data)
    data_scaled = scale_data(df)

    results = tune_hyperparameters(data_scaled)
    results_df = pd.DataFrame(results)

    # Compute a combined metric to sort the tuning results
    results_df['silhouette_x_noise'] = results_df['silhouette_score'] * (1 - results_df['noise_proportion'])
    results_df_sorted = results_df.sort_values(by='silhouette_x_noise', ascending=False)
    results_df_sorted.to_csv(args.output, index=False)
    print("Tuning completed. Results saved to:", args.output)

if __name__ == "__main__":
    main()
