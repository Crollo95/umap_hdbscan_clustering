#!/usr/bin/env python
import argparse
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import scale_data
from src.clustering import perform_clustering, save_cluster_assignments
from src.plotting import plot_embedding

def main():
    parser = argparse.ArgumentParser(description="Run clustering and plotting using best tuned parameters")
    parser.add_argument("--data", type=str, required=True, help="Path to data file (CSV/TXT)")
    parser.add_argument("--tuning", type=str, required=True, help="CSV file with tuning results")
    parser.add_argument("--assignments", type=str, default="cluster_assignments.csv",
                        help="Output CSV file for cluster assignments")
    args = parser.parse_args()

    df = load_data(args.data)
    data_scaled = scale_data(df)
    
    # Load tuning results and extract the best parameters
    tuning_df = pd.read_csv(args.tuning)
    best_params = tuning_df.iloc[0].to_dict()

    embedding, labels, silhouette, noise_proportion = perform_clustering(data_scaled, best_params)
    save_cluster_assignments(df, labels, args.assignments)
    print("Clustering completed. Cluster assignments saved to:", args.assignments)

    plot_embedding(embedding, labels, silhouette, noise_proportion, save_path="results/embedding_plot.png")

if __name__ == "__main__":
    main()

