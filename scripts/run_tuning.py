import argparse
import os
import pandas as pd
import yaml
from src.data_loader import load_data
from src.preprocessing import scale_data
from src.tuning import tune_hyperparameters

def main():
    parser = argparse.ArgumentParser(description="Run UMAP + HDBSCAN hyperparameter tuning with config file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Ensure the output directory exists
    output_dir = os.path.dirname(config['tuning_output'])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = load_data(config['data'])
    data_scaled = scale_data(df, config.get('scaling_method', None))

    results = tune_hyperparameters(data_scaled)
    results_df = pd.DataFrame(results)

    # Compute a combined metric to sort the tuning results
    results_df['silhouette_x_noise'] = results_df['silhouette_score'] * (1 - results_df['noise_proportion'])
    results_df_sorted = results_df.sort_values(by='silhouette_x_noise', ascending=False)
    results_df_sorted.to_csv(config['tuning_output'], index=False)
    print("Tuning completed. Results saved to:", config['tuning_output'])

if __name__ == "__main__":
    main()

