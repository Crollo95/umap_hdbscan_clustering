from itertools import product
from joblib import Parallel, delayed
import umap.umap_ as umap
import hdbscan
from sklearn.metrics import silhouette_score
import numpy as np
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.utils.deprecation"
)
warnings.filterwarnings(
    "ignore",
    message=r"n_jobs value .* overridden to .*",
    category=UserWarning
)


def evaluate_params(n_neighbors, min_dist, min_cluster_size, epsilon, data_scaled, umap_metric):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, 
        min_dist=min_dist,
        metric=umap_metric,
        n_components=2, 
        random_state=0
    )
    embedding = reducer.fit_transform(data_scaled)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=epsilon
    )
    labels = clusterer.fit_predict(embedding)

    mask = labels != -1
    if len(np.unique(labels[mask])) > 1:
        silhouette = silhouette_score(embedding[mask], labels[mask])
    else:
        silhouette = -1  # invalid silhouette

    noise_proportion = np.mean(labels == -1)

    return {
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'min_cluster_size': min_cluster_size,
        'cluster_selection_epsilon': epsilon,
        'silhouette_score': silhouette,
        'noise_proportion': noise_proportion,
        'num_clusters': len(set(labels)) - (1 if -1 in labels else 0)
    }

def tune_hyperparameters(data_scaled, n_jobs=-1, verbose=10):
    # Determine metric based on data type (binary or numeric)

    unique_vals = np.unique(data_scaled)

    # True when *all* unique values belong to the allowed alphabet
    allowed = np.array([0.0, 0.5, 1.0])
    use_cosine = np.all(np.isin(unique_vals, allowed))

    umap_metric = "cosine" if use_cosine else "euclidean"

    print(f"Using {umap_metric} as UMAP metric ")
    if len(unique_vals) <=3:
        print(f"(unique values = {unique_vals.tolist()})")

    # Hyperparameter grid
    umap_n_neighbors = [5, 10, 15, 20, 30, 50]
    umap_min_dist = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    hdbscan_min_cluster_size = [10, 20, 30, 40, 50]
    hdbscan_cluster_selection_epsilon = [0, 0.01, 0.02, 0.05, 0.1]

    # Create all combinations of parameters
    param_grid = list(product(
        umap_n_neighbors, 
        umap_min_dist, 
        hdbscan_min_cluster_size, 
        hdbscan_cluster_selection_epsilon
    ))

    # Parallel execution of tuning
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(evaluate_params)(
            n_neighbors, min_dist, min_cluster_size, epsilon, data_scaled, umap_metric
        ) 
        for (n_neighbors, min_dist, min_cluster_size, epsilon) in param_grid
    )
    return results
