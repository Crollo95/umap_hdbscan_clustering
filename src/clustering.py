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

    unique_vals = np.unique(data_scaled)

    # True when *all* unique values belong to the allowed alphabet
    allowed = np.array([0.0, 0.5, 1.0])
    use_cosine = np.all(np.isin(unique_vals, allowed))

    umap_metric = "cosine" if use_cosine else "euclidean"

    print(f"Using {umap_metric} as UMAP metric ")
    if len(unique_vals) <=3:
        print(f"(unique values = {unique_vals.tolist()})")
        
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

def save_embedding(embedding, df_index, output_file):
    """Save 2D embedding to CSV."""
    embedding_df = pd.DataFrame(
        embedding,
        index=df_index,
        columns=['UMAP1', 'UMAP2']
    )
    # Include index as sample_id for clarity
    embedding_df.index.name = 'sample_id'
    embedding_df.to_csv(output_file)
    return embedding_df