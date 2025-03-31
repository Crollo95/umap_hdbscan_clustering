# UMAP + HDBSCAN Clustering and Classification Pipeline

## Project Overview
This project provides a pipeline for clustering high-dimensional data using **UMAP** (Uniform Manifold Approximation and Projection) for dimensionality reduction and **HDBSCAN** (Hierarchical DBSCAN) for clustering. It automates hyperparameter tuning to find optimal parameters for clustering, then performs clustering using the best parameters. Users can select from different scaling methods: `standard`, `minmax`, or no scaling (`null`). Additionally, it trains a **Random Forest classifier** to predict cluster assignments, providing classification metrics and explainability with **SHAP** values. The outputs include cluster assignments, classification reports, confusion matrices, and SHAP visualizations of feature importance.

## Installation
1. **Clone or Download**: Obtain the project code by cloning the repository or downloading the ZIP.
2. **Install Dependencies**: Ensure you have **Python 3.x** installed, then install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```
3. **Project Setup**: Navigate to the project directory. If not already present, create a `data/` folder for your dataset and a `results/` folder for outputs.

## Configuration
The pipeline uses a YAML configuration file (`config.yaml`) to simplify parameter management. Prepare your configuration file as follows:

```yaml
data: data/data.csv
scaling_method: null # options: "standard", "minmax", or null for no scaling
tuning_output: results/tuning_results.csv
assignments_output: results/cluster_assignments.csv
classification_report_output: results/classification_report.csv
```

## Usage
Before running the pipeline, prepare your dataset as a CSV or TXT file (tab or comma-separated values) and place it in the `data/` folder (e.g., `data/data.csv`). Then, from the project’s root directory, use the following commands:

### Hyperparameter Tuning
Search for the best UMAP and HDBSCAN parameters by running:
```bash
python -m scripts.run_tuning --config config.yaml
```

- **What it does**:
    - Loads and optionally scales your dataset according to the configuration.
    - Evaluates multiple combinations of UMAP (`n_neighbors`, `min_dist`) and HDBSCAN (`min_cluster_size`, `cluster_selection_epsilon`) parameters.
    - Saves tuning results (including metrics like silhouette score, noise proportion, and number of clusters) to the location specified in `tuning_output`.

### Clustering and Visualization
Perform clustering using the best-found parameters and generate a visualization:
```bash
python -m scripts.run_clustering --config config.yaml
```

- **What it does**:
    - Loads your dataset and scaling options from the YAML configuration.
    - Uses the top-performing parameters (first row in the tuning CSV) to perform UMAP dimensionality reduction and HDBSCAN clustering.
    - Saves the cluster assignments to the location specified in `assignments_output`.
    - Saves a 2D scatter plot of the UMAP embedding as `embedding_plot.png` in the same directory as your cluster assignments.

### Classification and Explainability
Train a Random Forest classifier on the cluster assignments and compute classification metrics with SHAP explanations:
```bash
python -m scripts.run_classification --config config.yaml
```

- **What it does**:
    - Loads your dataset, scaling options, and cluster assignments from the YAML configuration.
    - Performs a stratified train-test split, excluding noise points.
    - Trains a Random Forest classifier to predict cluster labels.
    - Computes and saves classification metrics (precision, recall, F1-score, balanced accuracy, MCC) and a confusion matrix.
    - Generates SHAP beeswarm plots for feature importance, saved in a dedicated `shap_plots` directory within the results folder.

### Results
After running all scripts, check the specified output directory (`results/`) for:

- `tuning_results.csv` – Hyperparameter combinations and clustering performance metrics.
- `cluster_assignments.csv` – Cluster labels for each sample (with -1 indicating noise).
- `embedding_plot.png` – Scatter plot of the 2D UMAP embedding with clusters distinguished by color.
- `classification_report.csv` – Classification metrics from the Random Forest classifier.
- `confusion_matrix.csv` – Confusion matrix showing classifier performance.
- `additional_metrics.csv` – Includes balanced accuracy and MCC metrics.
- `shap_plots/` – Contains SHAP beeswarm plots for feature importance, aiding interpretability and explainability.


    