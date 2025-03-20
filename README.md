# UMAP + HDBSCAN Clustering Pipeline

## Project Overview
This project provides a pipeline for clustering high-dimensional data using **UMAP** (Uniform Manifold Approximation and Projection) for dimensionality reduction and **HDBSCAN** (Hierarchical DBSCAN) for clustering. It automates hyperparameter tuning to find the optimal UMAP and HDBSCAN parameters for a given dataset, then performs clustering using the best parameters. The result is a set of cluster assignments for each data point and a 2D visualization of the clusters, helping users identify meaningful groups and outliers in their data.


## Installation
1. **Clone or Download**: Obtain the project code by cloning the repository or downloading the ZIP.
2. **Install Dependencies**: Ensure you have **Python 3.x** installed, then install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```
3. **Project Setup**: Navigate to the project directory. If not already present, create a `data/` folder for your dataset and a `results/` folder for outputs.


## Usage
Before running the pipeline, prepare your dataset as a CSV or TXT file (tab or comma-separated values) and place it in the `data/` folder (e.g., `data/data.txt`). Then, from the project’s root directory, use the following commands:

### Hyperparameter Tuning
Search for the best UMAP and HDBSCAN parameters by running:
```bash
python -m scripts.run_tuning --data data/data.txt --output results/tuning_results.csv
```

- What it does:
    - Loads and scales your dataset.
    - Evaluates multiple combinations of UMAP (`n_neighbors`, `min_dist`) and HDBSCAN (`min_cluster_size`, `cluster_selection_epsilon`) parameters.
    - Saves the tuning results (including metrics like silhouette score, noise proportion, and number of clusters) to `results/tuning_results.csv`.
    - 

### Clustering and Visualization
Perform clustering using the best found parameters and generate a visualization:
```bash
python -m scripts.run_clustering --data data/data.txt --tuning results/tuning_results.csv --assignments results/cluster_assignments.csv
```

- What it does:

    - Loads your dataset and the tuning results.
    - Uses the top-performing parameters (first row in the tuning CSV) to perform UMAP dimensionality reduction and HDBSCAN clustering.
    - Saves the cluster assignments to `results/cluster_assignments.csv`.
    - Displays a 2D scatter plot of the UMAP embedding. (Optionally, modify the script to also save the plot as an image file.)


### Results

After running the scripts, check the `results/` directory for:

- `tuning_results.csv` – Contains hyperparameter combinations and their clustering performance metrics.
- `cluster_assignments.csv` – Contains the cluster label for each sample (with -1 indicating noise).
- Plot – A scatter plot of the 2D UMAP embedding with clusters distinguished by color. (If modified to save, it will be saved as specified, e.g., `embedding_plot.png`.)

    