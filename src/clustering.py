import umap.umap_ as umap
import hdbscan
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd

def perform_clustering(data_scaled, params):
    """Perform UMAP embedding and HDBSCAN clustering with given parameters.
    
    Args:
        data_scaled: Preprocessed data.
        params: Dictionary containing parameters (as returned from tuning).
    
    Returns:
        A tuple: (embedding, labels, silhouette, noise_proportion)
    """
    # Determine metric based on data type (binary or numeric)
    unique_values = np.unique(data_scaled)
    if np.array_equal(unique_values, [0, 1]) or np.array_equal(unique_values, [0]) or np.array_equal(unique_values, [1]):
        umap_metric = "cosine"
    else:
        umap_metric = "euclidean"

    print(f"Using {umap_metric} metric")

    reducer = umap.UMAP(
        n_neighbors=int(params['n_neighbors']),
        min_dist=params['min_dist'],
        metric=umap_metric,
        n_components=2,
        random_state=0
    )
    embedding = reducer.fit_transform(data_scaled)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(params['min_cluster_size']),
        cluster_selection_epsilon=float(params['cluster_selection_epsilon'])
    )
    labels = clusterer.fit_predict(embedding)

    # Compute silhouette score (ignoring noise)
    mask = labels != -1
    if len(np.unique(labels[mask])) > 1:
        silhouette = silhouette_score(embedding[mask], labels[mask])
    else:
        silhouette = -1

    noise_proportion = np.mean(labels == -1)

    return embedding, labels, silhouette, noise_proportion

def save_cluster_assignments(df, labels, output_file):
    """Save cluster assignments to CSV."""
    cluster_assignments = pd.DataFrame({
        'sample_id': df.index,
        'cluster_label': labels
    })
    cluster_assignments.to_csv(output_file, index=False)
    return cluster_assignments
