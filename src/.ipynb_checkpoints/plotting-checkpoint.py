import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

def plot_embedding(embedding, labels, silhouette, noise_proportion, save_path=None):
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Filter out noise for color assignment
    cluster_labels = [label for label in unique_labels if label != -1]

    # Using tab10 for clusters and grey for noise
    colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_labels)))
    color_mapping = {label: colors[i] for i, label in enumerate(cluster_labels)}
    # Convert hex to RGBA for consistency
    color_mapping[-1] = mcolors.to_rgba('#bdbdbd')

    point_colors = np.array([color_mapping[label] for label in labels])

    plt.figure(figsize=(12, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=point_colors, s=20, alpha=0.8)
    plt.title(
        f'UMAP + HDBSCAN Clustering\nSilhouette Score: {silhouette:.3f} | Noise: {noise_proportion:.2%}',
        fontsize=16
    )
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    handles = [
        Patch(color=color_mapping[label],
              label=f'Noise (-1): {counts[unique_labels.tolist().index(label)]} pts' if label == -1 
              else f'Cluster {label}: {counts[unique_labels.tolist().index(label)]} pts')
        for label in unique_labels
    ]
    plt.legend(handles=handles, title='Clusters (size)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

