import argparse
import os
import pandas as pd
import yaml
from src.data_loader import load_data
from src.preprocessing import scale_data
from src.clustering import perform_clustering, save_cluster_assignments
from src.plotting import plot_embedding

def main():
    parser = argparse.ArgumentParser(description="Run clustering and plotting using YAML config")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Ensure the assignments output directory exists
    assignments_dir = os.path.dirname(config['assignments_output'])
    if assignments_dir and not os.path.exists(assignments_dir):
        os.makedirs(assignments_dir)

    df = load_data(config['data'])
    data_scaled = scale_data(df, config.get('scaling_method', None))

    # Load tuning results and extract the best parameters (first row)
    tuning_df = pd.read_csv(config['tuning_output'])
    best_params = tuning_df.iloc[0].to_dict()

    embedding, labels, silhouette, noise_proportion = perform_clustering(data_scaled, best_params)
    save_cluster_assignments(df, labels, config['assignments_output'])
    print("Clustering completed. Cluster assignments saved to:", config['assignments_output'])

    # Save the embedding plot in the same directory as the assignments file
    if assignments_dir:
        embedding_plot_path = os.path.join(assignments_dir, "embedding_plot.png")
    else:
        embedding_plot_path = "embedding_plot.png"
    plot_embedding(embedding, labels, silhouette, noise_proportion, save_path=embedding_plot_path)
    print("Embedding plot saved to:", embedding_plot_path)

if __name__ == "__main__":
    main()