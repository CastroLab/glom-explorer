"""Data reshaping utilities."""

import pandas as pd

__all__ = ["melt_results_dict", "melt_all_results"]


def melt_results_dict(results_dict, index_name=None, iteration_name="iteration",
                      value_name="value", method_name="method"):
    """Convert a dictionary of DataFrames to long format.

    Parameters
    ----------
    results_dict : dict
        Keys are method names, values are DataFrames (index = component count,
        columns = iteration number).
    index_name : str, optional
        Name for the index column.  Falls back to the DataFrame's own index
        name or ``'index'``.
    iteration_name, value_name, method_name : str
        Column names in the output.

    Returns
    -------
    DataFrame
        Long-format with columns [*index_name*, *iteration_name*, *value_name*,
        *method_name*].
    """
    melted_dfs = []
    for method, df in results_dict.items():
        if index_name is None:
            idx_name = df.index.name if df.index.name is not None else "index"
        else:
            idx_name = index_name

        df_melted = df.reset_index().melt(
            id_vars=df.index.names[0] if df.index.names[0] is not None else "index",
            var_name=iteration_name,
            value_name=value_name,
        )
        df_melted.rename(columns={df_melted.columns[0]: idx_name}, inplace=True)
        df_melted[method_name] = method
        melted_dfs.append(df_melted)

    return pd.concat(melted_dfs, ignore_index=True)


def melt_all_results(results_dict):
    """Convert reconstruction results to long format (legacy wrapper)."""
    return melt_results_dict(
        results_dict,
        index_name="n_components",
        value_name="variance_explained",
    )
