"""Data loading and preprocessing for the omp8x glomerular dataset."""

import os
from collections import namedtuple
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import DATA_DIR, NMF_VARIANTS

__all__ = [
    "load_odorants",
    "load_glomerular_data",
    "load_xy_coords",
    "load_processed_data",
    "load_clustered_variant",
    "load_clustered_data",
]


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def load_xy_coords(data_dir=None):
    """Load X/Y coordinates for all four subjects.

    Returns
    -------
    dict
        ``{'S1': DataFrame, ..., 'S4': DataFrame}`` each with columns
        ``X`` and ``Y``.
    """
    data_dir = data_dir or DATA_DIR
    data_dict = {}
    for subject in range(1, 5):
        df = pd.read_csv(os.path.join(data_dir, f"glom_coords_{subject}.csv"), header=0)
        df = df.drop("ROI", axis=1)
        df.index.name = "glom#"
        data_dict[f"S{subject}"] = df
    return data_dict


def load_odorants(data_dir=None):
    """Load odorant names, SMILES, and chemical-group metadata.

    Returns
    -------
    DataFrame
        Columns: ``Odorant``, ``SMILES``, ``Name``, ``Group``, ``Group_num``.
    """
    data_dir = data_dir or DATA_DIR
    odorants = pd.concat(
        [
            pd.read_csv(os.path.join(data_dir, "omp8x_odornames.txt"),
                        header=None, names=["Odorant"]),
            pd.read_csv(os.path.join(data_dir, "omp8x_smiles.txt"),
                        header=None, names=["SMILES"]),
        ],
        axis=1,
    ).rename_axis("Odorant #")

    odorants["Name"] = odorants["Odorant"].apply(lambda x: " ".join(x.split()[1:]))
    odorants["Group"] = odorants["Odorant"].apply(lambda x: x.split()[0][0])

    # Fix known SMILES error for P3 odorant
    mask = odorants["Odorant"] == "P3 2-methoxy-3(5 or 6)-isopropylpyrazine"
    odorants.loc[mask, "SMILES"] = "CC(C)C1=CN=CC(=N1)OC"

    odorants["Group_num"] = LabelEncoder().fit_transform(odorants["Group"])
    return odorants


def load_glomerular_data(data_dir=None):
    """Load raw glomerular response matrices for all subjects and concentrations.

    Returns
    -------
    dict
        ``{'S1_low': DataFrame, 'S1_high': DataFrame, ...}`` each
        shaped (n_odorants, n_glomeruli).
    """
    data_dir = data_dir or DATA_DIR
    data_dict = {}
    for subject, conc in product(range(1, 5), ["high", "low"]):
        df = pd.read_csv(os.path.join(data_dir, f"omp8x_{subject}_{conc}.txt"),
                         header=None)
        data_dict[f"S{subject}_{conc}"] = df.T
    return data_dict


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _create_multiindex_data(data_dict):
    """Add (subject, glom#) MultiIndex columns to each DataFrame."""
    indexed = {}
    for key, df in data_dict.items():
        subject_id = key.split("_")[0][1]
        df_copy = df.copy()
        df_copy.columns = pd.MultiIndex.from_product(
            [[subject_id], df.columns.astype(str)], names=["subject", "gloms"],
        )
        df_copy.index.name = "odorant"
        indexed[key] = df_copy
    return indexed


def _filter_by_condition(data_dict, condition):
    """Concatenate DataFrames whose key contains *condition*."""
    d = {k: v for k, v in data_dict.items() if condition in k}
    return pd.concat(d.values(), axis=1)


# ---------------------------------------------------------------------------
# High-level loaders
# ---------------------------------------------------------------------------

def load_processed_data(data_dir=None):
    """Load and preprocess all odorant + glomerular data.

    Returns
    -------
    odorants : DataFrame
    low : DataFrame
        Low-concentration responses, concatenated across subjects.
    high : DataFrame
        High-concentration responses, concatenated across subjects.
    """
    data_dir = data_dir or DATA_DIR
    odorants = load_odorants(data_dir)
    raw = load_glomerular_data(data_dir)
    indexed = _create_multiindex_data(raw)
    low = _filter_by_condition(indexed, "low")
    high = _filter_by_condition(indexed, "high")
    return odorants, low, high


def _fix_column_types(df):
    """Ensure numeric column levels survive CSV round-trip."""
    names = df.columns.names
    arrays = []
    for name in names:
        vals = df.columns.get_level_values(name)
        if name in ("parent_cluster", "subcluster"):
            arrays.append(vals.astype(int))
        elif name in ("x", "y"):
            arrays.append(vals.astype(float))
        else:
            arrays.append(vals)
    df.columns = pd.MultiIndex.from_arrays(arrays, names=names)
    return df


def load_clustered_variant(variant="nonresponders_retained", data_dir=None):
    """Load pre-computed clustered low/high DataFrames for an NMF variant.

    Parameters
    ----------
    variant : str
        Key in ``NMF_VARIANTS`` (e.g. ``'nonresponders_retained'``).
    data_dir : path-like, optional
        Override the default data directory.

    Returns
    -------
    low, high : DataFrame
        Each with a 6-level column MultiIndex.
    """
    data_dir = data_dir or DATA_DIR
    files = NMF_VARIANTS[variant]["files"]
    low = _fix_column_types(
        pd.read_csv(os.path.join(data_dir, files[0]),
                     header=[0, 1, 2, 3, 4, 5], index_col=[0, 1])
    )
    high = _fix_column_types(
        pd.read_csv(os.path.join(data_dir, files[1]),
                     header=[0, 1, 2, 3, 4, 5], index_col=[0, 1])
    )
    return low, high


def load_clustered_data(method="nmf", threshold=0, data_dir=None):
    """Load raw data, drop inactive glomeruli, cluster, and sort.

    Returns
    -------
    ClusteredData
        Named tuple: ``odorants, low, high, low_sorted, high_sorted,
        row_labels, col_labels``.
    """
    from .clustering import assign_cluster_indices, apply_cluster_indices, permute_by_cluster_indices

    odorants, low, high = load_processed_data(data_dir)

    cols_to_retain = low.mean() > threshold
    n_dropped = (~cols_to_retain).sum()
    print(f"Dropped {n_dropped} of {low.shape[1]} columns (threshold={threshold})")
    low = low.loc[:, cols_to_retain]
    high = high.loc[:, cols_to_retain]

    row_labels, col_labels = assign_cluster_indices(low, method)
    low_sorted = permute_by_cluster_indices(apply_cluster_indices(low, row_labels, col_labels))
    high_sorted = permute_by_cluster_indices(apply_cluster_indices(high, row_labels, col_labels))

    ClusteredData = namedtuple(
        "ClusteredData",
        ["odorants", "low", "high", "low_sorted", "high_sorted", "row_labels", "col_labels"],
    )
    return ClusteredData(odorants, low, high, low_sorted, high_sorted, row_labels, col_labels)
