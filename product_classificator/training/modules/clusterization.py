import umap
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score


def get_reduced_embeds(embeddings: np.ndarray):
    reducer = umap.UMAP()
    return reducer.fit_transform(embeddings)


def get_clusterizer(reduced_embed: np.ndarray, n_clusters: int, random_state: int = 42):
    clusterizer = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusterizer.fit(reduced_embed)
    return clusterizer


def get_clusterization_metrics(df: pd.DataFrame, params: list[str], clusterizer, reduced_embed) -> pd.DataFrame:
    clusterisation_results = pd.DataFrame(columns=['AMI', 'V-Measure'])
    labels = clusterizer.predict(reduced_embed)

    for param in params:
        idxs = df[df[param].notna()].index

        clusterisation_results.loc[param, :] = [
            adjusted_mutual_info_score(df.loc[idxs, param], labels[idxs]),
            v_measure_score(df.loc[idxs, param], labels[idxs])
        ]

    return clusterisation_results


__all__ = [
    'get_reduced_embeds',
    'get_clusterizer',
    'get_clusterization_metrics',
]
