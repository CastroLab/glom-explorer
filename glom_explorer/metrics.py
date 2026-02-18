"""Topography metric computation (pure numpy, no visualization dependencies)."""

import numpy as np

__all__ = [
    "compute_metric",
    "response_amplitude",
    "binary_cluster_membership",
    "nmf_class_selectivity",
    "chemical_class_selectivity",
    "lifetime_sparseness",
]


def _lifetime_sparseness_raw(data):
    """S = (1 - mean(r)^2 / mean(r^2)) / (1 - 1/n), clipped to [0, 1]."""
    n = data.shape[0]
    if n <= 1:
        return np.zeros(data.shape[1])
    mean_r = data.mean(axis=0)
    mean_r2 = (data ** 2).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (1 - (mean_r ** 2) / mean_r2) / (1 - 1 / n)
    s = np.where(np.isfinite(s), s, 0.0)
    return np.clip(s, 0, 1)


def response_amplitude(responses, odorant_mask):
    """Mean response across selected odorants for each glomerulus."""
    if odorant_mask.sum() == 0:
        return np.zeros(responses.shape[1])
    return responses[odorant_mask].mean(axis=0)


def binary_cluster_membership(parent_clusters, facet_cluster=None):
    """1 if glom belongs to *facet_cluster*, else 0.

    Falls back to categorical cluster-ID colouring when *facet_cluster* is None.
    """
    if facet_cluster is None:
        return parent_clusters.astype(float)
    return (parent_clusters == facet_cluster).astype(float)


def nmf_class_selectivity(responses, parent_clusters, odorant_clusters):
    """For each glom in parent-cluster P: sum(|R[odor_cluster==P]|) / sum(|R|)."""
    abs_r = np.abs(responses)
    total = abs_r.sum(axis=0)
    result = np.zeros_like(total)
    for pc in np.unique(parent_clusters):
        glom_in_pc = parent_clusters == pc
        odor_in_pc = odorant_clusters == pc
        result[glom_in_pc] = abs_r[np.ix_(odor_in_pc, glom_in_pc)].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(total > 0, result / total, 0.0)
    return result


def chemical_class_selectivity(responses, groups):
    """For each glom: max over chemical groups of sum(|R[group==G]|) / sum(|R|)."""
    abs_r = np.abs(responses)
    total = abs_r.sum(axis=0)
    unique_groups = np.unique(groups)
    fractions = np.zeros((len(unique_groups), responses.shape[1]))
    for i, g in enumerate(unique_groups):
        fractions[i] = abs_r[groups == g].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = np.where(total > 0, fractions / total, 0.0)
    return fractions.max(axis=0)


def lifetime_sparseness(responses, odorant_mask):
    """Lifetime sparseness (rectified) over selected odorants."""
    data = responses[odorant_mask] if odorant_mask.any() else responses
    data = np.maximum(data, 0)
    return _lifetime_sparseness_raw(data)


def compute_metric(name, responses, parent_clusters, odorant_clusters,
                   groups, odorant_mask=None, facet_cluster=None):
    """Dispatch to the appropriate metric function.

    Parameters
    ----------
    name : str
        One of ``'response_amplitude'``, ``'binary_membership'``,
        ``'nmf_selectivity'``, ``'chemical_selectivity'``,
        ``'lifetime_sparseness'``.
    responses : ndarray (n_odorants, n_gloms)
    parent_clusters : ndarray (n_gloms,)
    odorant_clusters : ndarray (n_odorants,)
    groups : ndarray (n_odorants,)  — chemical group letters
    odorant_mask : bool ndarray (n_odorants,), optional
    facet_cluster : int or None
    """
    if odorant_mask is None:
        odorant_mask = np.ones(responses.shape[0], dtype=bool)

    dispatch = {
        "response_amplitude": lambda: response_amplitude(responses, odorant_mask),
        "binary_membership": lambda: binary_cluster_membership(parent_clusters, facet_cluster),
        "nmf_selectivity": lambda: nmf_class_selectivity(responses, parent_clusters, odorant_clusters),
        "chemical_selectivity": lambda: chemical_class_selectivity(responses, groups),
        "lifetime_sparseness": lambda: lifetime_sparseness(responses, odorant_mask),
    }
    if name not in dispatch:
        raise ValueError(f"Unknown metric: {name}")
    return dispatch[name]()
