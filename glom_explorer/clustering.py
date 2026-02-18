"""NMF clustering, reconstruction analysis, and consensus stability."""

from itertools import permutations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralBiclustering
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

from .config import ANALYSIS_DEFAULTS

__all__ = [
    "assign_cluster_indices",
    "apply_cluster_indices",
    "permute_by_cluster_indices",
    "reorder_by_clusters",
    "query_data_cluster",
    "run_reconstruction_analysis",
    "calc_nmf_basis_scores",
    "consensus_clustering_stability",
]


# ---------------------------------------------------------------------------
# Cluster assignment & manipulation
# ---------------------------------------------------------------------------

def assign_cluster_indices(df, method, params=None):
    """Compute row and column cluster labels for *df*.

    Parameters
    ----------
    df : DataFrame
        Odorant-by-glomerulus response matrix.
    method : {'nmf', 'agglomerative', 'kmeans', 'spectral'}
    params : dict, optional
        Override default model parameters.

    Returns
    -------
    row_labels, col_labels : ndarray
    """
    n_clusters = ANALYSIS_DEFAULTS["n_clusters"]
    seed = ANALYSIS_DEFAULTS["nmf_seed"]
    max_iter = ANALYSIS_DEFAULTS["nmf_max_iter"]

    default_params = {
        "agglomerative": {"n_clusters": n_clusters, "linkage": "average", "metric": "euclidean"},
        "kmeans": {"n_clusters": n_clusters},
        "nmf": {"n_components": n_clusters, "random_state": seed, "max_iter": max_iter},
        "spectral": {"n_clusters": (n_clusters, n_clusters)},
    }

    epsilon = np.finfo(np.float32).eps
    df_stable = df.copy() + epsilon

    if params is None:
        params = default_params[method]

    methods = {
        "agglomerative": lambda d, p: AgglomerativeClustering(**p).fit(normalize(d, axis=1)).labels_,
        "kmeans": lambda d, p: KMeans(**p).fit(normalize(d, axis=1)).labels_,
        "nmf": lambda d, p: NMF(**p).fit_transform(d).argmax(axis=1),
        "spectral": lambda d, p: SpectralBiclustering(**p).fit(normalize(d, axis=1)).row_labels_,
    }

    row_labels = methods[method](df_stable, params)

    if method == "nmf":
        model = NMF(**params)
        model.fit(df_stable)
        col_labels = model.components_.argmax(axis=0)
    elif method == "spectral":
        col_labels = SpectralBiclustering(**params).fit(normalize(df_stable, axis=1)).column_labels_
    else:
        col_labels = methods[method](df_stable.T, params)

    return row_labels, col_labels


def apply_cluster_indices(df, row_labels, col_labels):
    """Attach cluster labels as new index levels."""
    df = df.copy()
    df.set_index(
        pd.MultiIndex.from_arrays([df.index, row_labels], names=["odorant", "clust"]),
        inplace=True,
    )
    df.columns = pd.MultiIndex.from_arrays(
        [
            df.columns.get_level_values("subject"),
            df.columns.get_level_values("gloms"),
            col_labels,
        ],
        names=["subject", "gloms", "clust"],
    )
    return df


def permute_by_cluster_indices(df, axis="both"):
    """Sort rows and/or columns by their ``clust`` level."""
    if not ("clust" in df.index.names and "clust" in df.columns.names):
        return df
    if axis in ("both", "odorants"):
        df = df.sort_index(level="clust", axis=0)
    if axis in ("both", "gloms"):
        df = df.sort_index(level="clust", axis=1)
    return df


def reorder_by_clusters(df, cluster_order, row_cluster_level=1, col_cluster_level=2):
    """Reorder rows and columns by a custom cluster ordering."""
    col_clusters = df.columns.get_level_values(col_cluster_level)
    col_indices = []
    for c in cluster_order:
        col_indices.extend(np.where(col_clusters == c)[0])
    df_reordered = df.iloc[:, col_indices]

    row_clusters = df_reordered.index.get_level_values(row_cluster_level)
    row_indices = []
    for c in cluster_order:
        row_indices.extend(np.where(row_clusters == c)[0])
    return df_reordered.iloc[row_indices, :]


def query_data_cluster(df, cluster_num, axis="gloms"):
    """Extract data for a single cluster along *axis*."""
    if axis == "gloms":
        return df.xs(cluster_num, level="clust", axis=1)
    elif axis == "odorants":
        return df.xs(cluster_num, level="clust", axis=0)


# ---------------------------------------------------------------------------
# Reconstruction & basis analysis
# ---------------------------------------------------------------------------

def run_reconstruction_analysis(data, max_components=15,
                                n_iterations=None):
    """PCA and NMF reconstruction curves (variance explained vs. n_components).

    Parameters
    ----------
    data : array-like
        Odorant-by-glomerulus matrix.
    max_components : int
    n_iterations : int, optional
        Defaults to ``ANALYSIS_DEFAULTS['n_shuffle_iterations']``.

    Returns
    -------
    dict
        ``{'PCA': DataFrame, 'PCA (Shuffled)': DataFrame, 'NMF': DataFrame}``
        each shaped (max_components, n_iterations).
    """
    if n_iterations is None:
        n_iterations = ANALYSIS_DEFAULTS["n_shuffle_iterations"]

    results = {}
    configs = {
        "PCA": {"model": PCA, "shuffle": False, "preprocess": lambda x: x},
        "PCA (Shuffled)": {"model": PCA, "shuffle": True, "preprocess": lambda x: x},
        "NMF": {"model": NMF, "shuffle": False,
                "preprocess": lambda x: x - x.min() if x.min() < 0 else x},
    }

    for method_name, cfg in configs.items():
        print(f"Running {method_name}...")
        X_base = cfg["preprocess"](np.array(data, dtype=float).copy())
        tss = np.sum((X_base - X_base.mean()) ** 2)

        method_results = []
        for i in range(n_iterations):
            if cfg["shuffle"]:
                rng = np.random.default_rng(seed=i)
                X = rng.permutation(X_base.flatten()).reshape(X_base.shape)
                X = cfg["preprocess"](X)
            else:
                X = X_base

            row = []
            for n_comp in range(1, max_components + 1):
                if cfg["model"] == NMF:
                    model = NMF(n_components=n_comp, random_state=i,
                                max_iter=ANALYSIS_DEFAULTS["nmf_max_iter"])
                    W = model.fit_transform(X)
                    X_rec = W @ model.components_
                else:
                    model = PCA(n_components=n_comp, random_state=i)
                    X_rec = model.inverse_transform(model.fit_transform(X))

                rss = np.sum((X - X_rec) ** 2)
                row.append(1 - rss / tss)
            method_results.append(row)

        results[method_name] = pd.DataFrame(
            np.array(method_results).T,
            index=range(1, max_components + 1),
            columns=range(n_iterations),
        )

    return results


def calc_nmf_basis_scores(X, n_components=None, n_runs=100):
    """KL-divergence scores across many NMF seeds.

    Returns
    -------
    dict
        ``{'kl-div': ndarray, 'seed values': ndarray}``
    """
    if n_components is None:
        n_components = ANALYSIS_DEFAULTS["n_clusters"]
    epsilon = 1e-10
    all_entropies = []

    np.random.seed(2011)
    random_seeds = np.random.randint(0, 2**31, n_runs)

    for i in range(n_runs):
        model = NMF(n_components=n_components, random_state=random_seeds[i],
                     max_iter=ANALYSIS_DEFAULTS["nmf_max_iter"])
        W = model.fit_transform(X) + epsilon
        W_norm = W / W.sum(axis=0, keepdims=True)

        basis_entropy = []
        for j, k in permutations(range(W.shape[1]), 2):
            basis_entropy.append(entropy(W_norm[:, j], W_norm[:, k]))
        all_entropies.append(np.mean(basis_entropy))

    return {"kl-div": np.array(all_entropies), "seed values": random_seeds}


def consensus_clustering_stability(X, n_components, n_runs=None):
    """Consensus clustering stability (AUC and ARI) for a given k.

    Returns
    -------
    dict
        Keys: ``consensus_auc``, ``mean_ari_to_consensus``,
        ``std_ari_to_consensus``, ``consensus_matrix``, ``consensus_clusters``.
    """
    if n_runs is None:
        n_runs = ANALYSIS_DEFAULTS["n_shuffle_iterations"]

    connectivity_matrices = []
    cluster_assignments = []

    for i in range(n_runs):
        model = NMF(n_components=n_components, random_state=i,
                     max_iter=ANALYSIS_DEFAULTS["nmf_max_iter"])
        W = model.fit_transform(X)
        clusters = np.argmax(W, axis=1)
        cluster_assignments.append(clusters)

        n_samples = len(clusters)
        conn = np.zeros((n_samples, n_samples))
        for j in range(n_samples):
            for k in range(n_samples):
                if clusters[j] == clusters[k]:
                    conn[j, k] = 1
        connectivity_matrices.append(conn)

    consensus = np.mean(connectivity_matrices, axis=0)
    distance = 1 - consensus
    link = linkage(squareform(distance), method="average")
    consensus_clusters = fcluster(link, n_components, criterion="maxclust")

    vals = consensus[np.triu_indices_from(consensus, k=1)]
    auc = np.trapz(np.sort(vals), dx=1 / len(vals))

    ari_scores = [adjusted_rand_score(ca, consensus_clusters)
                  for ca in cluster_assignments]

    return {
        "consensus_auc": auc,
        "mean_ari_to_consensus": np.mean(ari_scores),
        "std_ari_to_consensus": np.std(ari_scores),
        "consensus_matrix": consensus,
        "consensus_clusters": consensus_clusters,
    }
