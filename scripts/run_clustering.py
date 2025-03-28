import argparse
import os
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import scale_data
from src.clustering import perform_clustering, save_cluster_assignments
from src.plotting import plot_embedding

def main():
    parser = argparse.ArgumentParser(description="Run clustering and plotting using best tuned parameters")
    parser.add_argument("--data", type=str, required=True, help="Path to data file (CSV/TXT)")
    parser.add_argument("--tuning", type=str, required=True, help="CSV file with tuning results")
    parser.add_argument("--assignments", type=str, default="results/cluster_assignments.csv",
                        help="Output CSV file for cluster assignments")
    args = parser.parse_args()

    # Ensure the assignments output directory exists
    assignments_dir = os.path.dirname(args.assignments)
    if assignments_dir and not os.path.exists(assignments_dir):
        os.makedirs(assignments_dir)

    df = load_data(args.data)
    data_scaled = scale_data(df)
    
    # Load tuning results and extract the best parameters (first row)
    tuning_df = pd.read_csv(args.tuning)
    best_params = tuning_df.iloc[0].to_dict()

    embedding, labels, silhouette, noise_proportion = perform_clustering(data_scaled, best_params)
    save_cluster_assignments(df, labels, args.assignments)
    print("Clustering completed. Cluster assignments saved to:", args.assignments)

    # Save the embedding plot in the same directory as the assignments file
    if assignments_dir:
        embedding_plot_path = os.path.join(assignments_dir, "embedding_plot.png")
    else:
        embedding_plot_path = "embedding_plot.png"
    plot_embedding(embedding, labels, silhouette, noise_proportion, save_path=embedding_plot_path)
    print("Embedding plot saved to:", embedding_plot_path)

if __name__ == "__main__":
    main()
