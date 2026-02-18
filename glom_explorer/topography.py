"""Plotly facet-grid visualizations for glomerular spatial topography.

Replaces the Dash CSS-grid of individual ``dcc.Graph`` objects with a single
``plotly.subplots.make_subplots()`` figure suitable for Jupyter notebooks.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import NMF_VARIANTS, DATA_DIR
from .data import load_processed_data, load_clustered_variant
from .metrics import compute_metric

__all__ = [
    "topography_facet_grid",
    "topography_preset",
]

# ---------------------------------------------------------------------------
# Facet variable registry (simplified from Dash GridVar dataclass)
# ---------------------------------------------------------------------------

FACET_VARS = {
    "(none)":          {"type": None,      "field": None},
    "Subject":         {"type": "glom",    "field": "subject"},
    "Concentration":   {"type": "conc",    "field": "concentration"},
    "Parent cluster":  {"type": "glom",    "field": "parent_cluster"},
    "Subcluster":      {"type": "glom",    "field": "subcluster"},
    "Odorant cluster": {"type": "odorant", "field": "odorant_cluster"},
    "Odorant":         {"type": "odorant", "field": "odorant"},
}

# Metric -> (default_colorscale, odorant_aware)
METRIC_COLORS = {
    "response_amplitude":  {"colorscale": "OrRd",  "odorant_aware": True},
    "binary_membership":   {"colorscale": "OrRd",  "odorant_aware": False},
    "nmf_selectivity":     {"colorscale": "OrRd",  "odorant_aware": False},
    "chemical_selectivity": {"colorscale": "OrRd", "odorant_aware": False},
    "lifetime_sparseness": {"colorscale": "OrRd",  "odorant_aware": True},
}

PRESETS = {
    "cluster_map":        {"row_var": "(none)",        "col_var": "Parent cluster", "metric": "response_amplitude"},
    "cluster_by_subject": {"row_var": "Subject",       "col_var": "Parent cluster", "metric": "binary_membership"},
    "cluster_by_conc":    {"row_var": "Concentration", "col_var": "Parent cluster", "metric": "nmf_selectivity"},
    "subcluster_detail":  {"row_var": "Subcluster",    "col_var": "Parent cluster", "metric": "binary_membership"},
    "nmf_selectivity":    {"row_var": "(none)",        "col_var": "Parent cluster", "metric": "nmf_selectivity"},
    "chemical_tuning":    {"row_var": "(none)",        "col_var": "Parent cluster", "metric": "chemical_selectivity"},
    "odorant_groups":     {"row_var": "(none)",        "col_var": "Odorant cluster", "metric": "response_amplitude"},
    "group_by_conc":      {"row_var": "Concentration", "col_var": "Odorant cluster", "metric": "response_amplitude"},
    "sparseness_by_group": {"row_var": "(none)",       "col_var": "Odorant cluster", "metric": "lifetime_sparseness"},
}


# ---------------------------------------------------------------------------
# Internal: load and cache variant data
# ---------------------------------------------------------------------------

_VARIANT_CACHE: dict = {}


def _load_variant_data(variant):
    """Load (or return cached) numpy arrays + metadata for *variant*."""
    if variant in _VARIANT_CACHE:
        return _VARIANT_CACHE[variant]

    odorants, _, _ = load_processed_data()
    low_df, high_df = load_clustered_variant(variant)

    gmeta = low_df.columns.to_frame(index=False).copy()
    gmeta.columns = ["gloms", "subject", "parent_cluster", "subcluster", "x", "y"]

    ometa = low_df.index.to_frame(index=False).copy()
    ometa.columns = ["odorant", "odorant_cluster"]
    ometa["name"] = odorants["Name"].values
    ometa["group"] = odorants["Group"].values

    vdata = {
        "low": low_df.values,
        "high": high_df.values,
        "glom_meta": gmeta,
        "odorant_meta": ometa,
    }
    _VARIANT_CACHE[variant] = vdata
    return vdata


# ---------------------------------------------------------------------------
# Internal: facet-level helpers
# ---------------------------------------------------------------------------

def _get_levels(var_name, gmeta, ometa):
    """Return the list of levels for a facet variable."""
    spec = FACET_VARS[var_name]
    if spec["type"] is None:
        return [None]
    if spec["type"] == "conc":
        return ["low", "high"]
    if spec["type"] == "glom":
        return sorted(gmeta[spec["field"]].unique())
    if spec["type"] == "odorant":
        if spec["field"] == "odorant":
            return list(ometa["odorant"].values)
        return sorted(ometa[spec["field"]].unique())


def _level_label(var_name, level, ometa):
    """Human-readable label for one facet level."""
    if level is None:
        return ""
    labels = {
        "Subject": lambda l: f"S{l}",
        "Concentration": lambda l: l.capitalize(),
        "Parent cluster": lambda l: f"Cluster {l}",
        "Subcluster": lambda l: f"Sub {l}",
        "Odorant cluster": lambda l: f"Odor cluster {l}",
    }
    if var_name in labels:
        return labels[var_name](level)
    if var_name == "Odorant":
        row = ometa[ometa["odorant"] == level]
        return row.iloc[0]["name"] if len(row) else str(level)
    return str(level)


# ---------------------------------------------------------------------------
# Internal: resolve masks and compute cell data
# ---------------------------------------------------------------------------

def _resolve_masks(row_var, row_level, col_var, col_level,
                   concentration, glom_meta, odorant_meta,
                   skip_glom_fields=frozenset(), subjects=None):
    """Compute boolean masks and effective concentration for one cell."""
    glom_mask = np.ones(len(glom_meta), dtype=bool)
    odorant_mask = np.ones(len(odorant_meta), dtype=bool)
    conc = concentration

    for var_name, level in [(row_var, row_level), (col_var, col_level)]:
        spec = FACET_VARS[var_name]
        if spec["type"] is None:
            continue
        if spec["type"] == "glom" and spec["field"] not in skip_glom_fields:
            glom_mask &= glom_meta[spec["field"]].values == level
        elif spec["type"] == "odorant":
            odorant_mask &= odorant_meta[spec["field"]].values == level
        elif spec["type"] == "conc":
            conc = level

    # Subject filter
    if subjects is not None:
        glom_mask &= np.isin(glom_meta["subject"].values, subjects)

    return glom_mask, odorant_mask, conc


def _compute_cell(row_var, row_level, col_var, col_level, metric,
                  vdata, concentration, subjects):
    """Metric values and (x, y) for a single grid cell."""
    gmeta = vdata["glom_meta"]
    ometa = vdata["odorant_meta"]

    skip = {"parent_cluster"} if (
        metric == "binary_membership" and "Parent cluster" in (row_var, col_var)
    ) else set()

    glom_mask, odorant_mask, conc = _resolve_masks(
        row_var, row_level, col_var, col_level,
        concentration, gmeta, ometa, skip_glom_fields=skip, subjects=subjects,
    )

    if glom_mask.sum() == 0:
        return {"x": np.array([]), "y": np.array([]),
                "values": np.array([]), "pc": np.array([])}

    facet_cluster = None
    if metric == "binary_membership" and "Parent cluster" in (row_var, col_var):
        for v, lv in [(row_var, row_level), (col_var, col_level)]:
            if v == "Parent cluster":
                facet_cluster = lv

    responses = vdata[conc]
    pc = gmeta["parent_cluster"].values[glom_mask]
    oc = ometa["odorant_cluster"].values
    groups = ometa["group"].values

    values = compute_metric(
        metric, responses[:, glom_mask], pc, oc, groups,
        odorant_mask=odorant_mask, facet_cluster=facet_cluster,
    )
    return {
        "x": gmeta["x"].values[glom_mask],
        "y": gmeta["y"].values[glom_mask],
        "values": values,
        "pc": pc,
    }


def _build_scatter_trace(cell, vmin, vmax, colorscale, point_size,
                         show_colorbar=False):
    """Return a ``go.Scatter`` trace for foreground metric-coloured dots."""
    if len(cell["x"]) == 0:
        return go.Scatter(x=[], y=[], mode="markers", showlegend=False)

    order = np.argsort(cell["values"])
    x = cell["x"][order]
    y = cell["y"][order]
    vals = cell["values"][order]

    marker = dict(
        size=point_size, color=vals,
        colorscale=colorscale, cmin=vmin, cmax=vmax,
        showscale=show_colorbar,
    )
    if show_colorbar:
        marker["colorbar"] = dict(thickness=10, len=0.8, title="")

    return go.Scatter(
        x=x, y=y, mode="markers", marker=marker,
        hovertemplate="x=%{x:.0f}, y=%{y:.0f}<br>value=%{marker.color:.3f}<extra></extra>",
        showlegend=False,
    )


# ---------------------------------------------------------------------------
# Color-range helpers
# ---------------------------------------------------------------------------

def _color_range(metric, all_vals, cluster_is_facet):
    """Return ``(colorscale, vmin, vmax)`` for a metric."""
    if metric == "response_amplitude":
        return "OrRd", 0, max(float(np.percentile(all_vals, 95)), 0.01)
    if metric == "binary_membership":
        if cluster_is_facet:
            return [[0, "lightgrey"], [1, "steelblue"]], 0, 1
        return "Turbo", float(all_vals.min()), float(all_vals.max())
    # Default for nmf_selectivity, chemical_selectivity, lifetime_sparseness
    return "OrRd", 0, max(float(all_vals.max()), 0.01)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def topography_facet_grid(
    variant="nonresponders_retained",
    row_var="(none)",
    col_var="Parent cluster",
    metric="response_amplitude",
    concentration="low",
    colorscale=None,
    point_size=10,
    show_background=True,
    cell_size=250,
    subjects=None,
):
    """Build a faceted topography scatter grid as a single Plotly figure.

    Parameters
    ----------
    variant : str
        NMF variant name (key in ``NMF_VARIANTS``).
    row_var, col_var : str
        Facet variable names.  Choices: ``'(none)'``, ``'Subject'``,
        ``'Concentration'``, ``'Parent cluster'``, ``'Subcluster'``,
        ``'Odorant cluster'``, ``'Odorant'``.
    metric : str
        Colour metric.  Choices: ``'response_amplitude'``,
        ``'binary_membership'``, ``'nmf_selectivity'``,
        ``'chemical_selectivity'``, ``'lifetime_sparseness'``.
    concentration : {'low', 'high'}
        Default concentration (overridden when Concentration is a facet).
    colorscale : str, optional
        Plotly colorscale override.
    point_size : int
    show_background : bool
        Show grey background dots for spatial context.
    cell_size : int
        Height/width of each subplot in pixels.
    subjects : list of str, optional
        Subset of subjects (e.g. ``['1', '2']``).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    vdata = _load_variant_data(variant)
    gmeta = vdata["glom_meta"]
    ometa = vdata["odorant_meta"]

    row_levels = _get_levels(row_var, gmeta, ometa)
    col_levels = _get_levels(col_var, gmeta, ometa)
    n_rows = len(row_levels)
    n_cols = len(col_levels)

    # Build subplot titles
    titles = []
    for rl in row_levels:
        for cl in col_levels:
            parts = []
            if row_var != "(none)":
                parts.append(_level_label(row_var, rl, ometa))
            if col_var != "(none)":
                parts.append(_level_label(col_var, cl, ometa))
            titles.append(" | ".join(parts) if parts else "")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
    )

    # Precompute all cells
    cells = {}
    for i, rl in enumerate(row_levels):
        for j, cl in enumerate(col_levels):
            cells[(i, j)] = _compute_cell(
                row_var, rl, col_var, cl, metric, vdata, concentration, subjects,
            )

    # Global colour range
    all_vals = np.concatenate(
        [c["values"] for c in cells.values() if len(c["values"]) > 0]
    ) if any(len(c["values"]) > 0 for c in cells.values()) else np.array([0.0])

    cluster_is_facet = "Parent cluster" in (row_var, col_var)
    cs, vmin, vmax = _color_range(metric, all_vals, cluster_is_facet)

    if colorscale is not None and cs == "OrRd":
        cs = colorscale

    # Populate subplots
    for i, rl in enumerate(row_levels):
        for j, cl in enumerate(col_levels):
            r, c = i + 1, j + 1

            # Background dots
            if show_background:
                bg_mask = np.ones(len(gmeta), dtype=bool)
                for v, lv in [(row_var, rl), (col_var, cl)]:
                    if v == "Subject":
                        bg_mask = gmeta["subject"].values == lv
                if subjects is not None and "Subject" not in (row_var, col_var):
                    bg_mask &= np.isin(gmeta["subject"].values, subjects)

                fig.add_trace(go.Scatter(
                    x=gmeta["x"].values[bg_mask],
                    y=gmeta["y"].values[bg_mask],
                    mode="markers",
                    marker=dict(size=max(point_size - 2, 1), color="#e0e0e0"),
                    hoverinfo="skip", showlegend=False,
                ), row=r, col=c)

            # Foreground (coloured by metric)
            show_cb = (i == n_rows - 1 and j == n_cols - 1)
            trace = _build_scatter_trace(cells[(i, j)], vmin, vmax, cs,
                                         point_size, show_colorbar=show_cb)
            fig.add_trace(trace, row=r, col=c)

            # Axis formatting
            fig.update_xaxes(visible=False, scaleanchor=f"y{(i * n_cols + j) + 1}" if i == 0 and j == 0 else None,
                             row=r, col=c)
            fig.update_yaxes(visible=False, row=r, col=c)

    # Force equal aspect on all subplots by anchoring each y to its x
    for idx in range(n_rows * n_cols):
        ax_suffix = "" if idx == 0 else str(idx + 1)
        fig.update_layout(**{
            f"xaxis{ax_suffix}": dict(scaleanchor=f"y{ax_suffix}", scaleratio=1),
        })

    fig.update_layout(
        width=cell_size * n_cols + 100,
        height=cell_size * n_rows + 80,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font.size = 11

    return fig


def topography_preset(name, **overrides):
    """Convenience wrapper using named presets.

    Parameters
    ----------
    name : str
        One of: ``'cluster_map'``, ``'cluster_by_subject'``,
        ``'cluster_by_conc'``, ``'subcluster_detail'``,
        ``'nmf_selectivity'``, ``'chemical_tuning'``,
        ``'odorant_groups'``, ``'group_by_conc'``,
        ``'sparseness_by_group'``.
    **overrides
        Any keyword accepted by :func:`topography_facet_grid`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Choose from: {list(PRESETS)}")
    kwargs = {**PRESETS[name], **overrides}
    return topography_facet_grid(**kwargs)
