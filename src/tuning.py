from itertools import product
from joblib import Parallel, delayed
import umap.umap_ as umap
import hdbscan
from sklearn.metrics import silhouette_score
import numpy as np

def evaluate_params(n_neighbors, min_dist, min_cluster_size, epsilon, data_scaled):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
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
    # Hyperparameter grid
    umap_n_neighbors = [5, 10, 15, 20, 30, 50]
    umap_min_dist = [0.01, 0.05, 0.1, 0.15, 0.2]
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
            n_neighbors, min_dist, min_cluster_size, epsilon, data_scaled
        ) 
        for (n_neighbors, min_dist, min_cluster_size, epsilon) in param_grid
    )
    return results


