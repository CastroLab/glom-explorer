"""Plotly barplot visualizations for cluster tuning profiles."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu

from .utils import melt_results_dict

__all__ = [
    "mirrored_cluster_barplot",
    "plot_reconstruction",
]

# Plotly qualitative palette (replaces matplotlib tab10)
_QUALITATIVE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _nice_ceil(x):
    """Round *x* up to a visually clean number (10, 20, 50, 100, ...)."""
    if x <= 0:
        return 10
    magnitude = 10 ** np.floor(np.log10(x))
    residual = x / magnitude
    if residual <= 1.0:
        nice = 1.0
    elif residual <= 2.0:
        nice = 2.0
    elif residual <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * magnitude


def _prepare_barplot_data(df, cluster_num, odorants, cluster_level="parent_cluster",
                          subcluster=None, sort_by="nmf_cluster", group_column="Group"):
    """Select, sort, and melt cluster data for bar plotting.

    Returns
    -------
    long_df : DataFrame
        Long-format with columns ``'odorant'`` and ``'response'``.
    sorted_group_ids : ndarray
    n_glom : int
    """
    mask = df.columns.get_level_values(cluster_level) == cluster_num
    if subcluster is not None:
        mask = mask & (df.columns.get_level_values("subcluster") == subcluster)
    df_clust = df.loc[:, mask]
    n_glom = mask.sum()

    odorant_idx = df.index.get_level_values("odorant").astype(int).values
    odorant_clusters = df.index.get_level_values("odorant_cluster").astype(int).values

    if sort_by == "nmf_cluster":
        sort_order = np.argsort(odorant_clusters)
        sorted_group_ids = odorant_clusters[sort_order]
    elif sort_by == "chemical_class":
        groups = odorants[group_column].iloc[odorant_idx].values
        group_codes = pd.factorize(groups, sort=True)[0]
        sort_order = np.argsort(group_codes)
        sorted_group_ids = group_codes[sort_order]
    else:
        raise ValueError(f"sort_by must be 'nmf_cluster' or 'chemical_class', got '{sort_by}'")

    sorted_idx = odorant_idx[sort_order]
    sorted_names = odorants["Name"].iloc[sorted_idx].values

    long = pd.DataFrame(df_clust.values[sort_order], index=sorted_names)
    long = long.T.melt(var_name="odorant", value_name="response")
    return long, sorted_group_ids, n_glom


def _aggregate_for_plotly(long_df, estimator="mean", errorbar="se"):
    """Compute bar heights and error values from long-format data."""
    grouped = long_df.groupby("odorant", sort=False)["response"]
    values = grouped.mean() if estimator == "mean" else grouped.median()
    counts = grouped.count()
    stds = grouped.std()

    if errorbar == "se":
        errors = stds / np.sqrt(counts)
    elif errorbar == "sd":
        errors = stds
    elif errorbar == "ci":
        errors = 1.96 * stds / np.sqrt(counts)
    else:
        errors = pd.Series(0.0, index=values.index)

    return pd.DataFrame({
        "odorant": values.index,
        "value": values.values,
        "error": errors.values,
        "n": counts.values,
    })


def _compute_pvalues(low_long, high_long):
    """Mann-Whitney U per odorant, low vs. high."""
    pvals = {}
    for odorant in low_long["odorant"].unique():
        lv = low_long.loc[low_long["odorant"] == odorant, "response"].values
        hv = high_long.loc[high_long["odorant"] == odorant, "response"].values
        if len(lv) >= 1 and len(hv) >= 1:
            _, p = mannwhitneyu(lv, hv, alternative="two-sided")
            pvals[odorant] = p
        else:
            pvals[odorant] = np.nan
    return pvals


def _group_color_map(odorants, group_column="Group"):
    """Build group -> RGB string mapping from qualitative palette."""
    unique_groups = sorted(odorants[group_column].unique())
    return {
        g: _QUALITATIVE_COLORS[i % len(_QUALITATIVE_COLORS)]
        for i, g in enumerate(unique_groups)
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mirrored_cluster_barplot(
    low_df, high_df, cluster_num, odorants,
    low_color="#FFD700", high_color="#BA55D3",
    width=1000, height=700,
    cluster_level="parent_cluster",
    subcluster=None,
    sort_by="nmf_cluster",
    estimator="mean",
    errorbar="se",
    ylim="auto",
    group_column="Group",
    title=None,
):
    """Interactive mirrored barplot (low on top, high on bottom inverted).

    Parameters
    ----------
    low_df, high_df : DataFrame
        Clustered response matrices with MultiIndex columns.
    cluster_num : int
        Which parent cluster to display.
    odorants : DataFrame
        Odorant metadata (must contain ``'Name'`` and *group_column*).
    estimator : {'mean', 'median', 'distribution'}
        ``'distribution'`` shows box-and-whisker plots.
    errorbar : {'se', 'sd', 'ci', None}
        Ignored when *estimator* is ``'distribution'``.
    ylim : float or 'auto'

    Returns
    -------
    plotly.graph_objects.Figure
    """
    low_long, sorted_group_ids, n_glom = _prepare_barplot_data(
        low_df, cluster_num, odorants,
        cluster_level=cluster_level, subcluster=subcluster,
        sort_by=sort_by, group_column=group_column,
    )
    high_long, _, _ = _prepare_barplot_data(
        high_df, cluster_num, odorants,
        cluster_level=cluster_level, subcluster=subcluster,
        sort_by=sort_by, group_column=group_column,
    )

    pvals = _compute_pvalues(low_long, high_long)

    group_colors = _group_color_map(odorants, group_column)
    odorant_idx = low_df.index.get_level_values("odorant").astype(int).values
    name_to_group = dict(zip(
        odorants["Name"].iloc[odorant_idx],
        odorants[group_column].iloc[odorant_idx],
    ))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04)

    if estimator == "distribution":
        odorant_names = low_long["odorant"].unique()
        fig.add_trace(go.Box(x=low_long["odorant"], y=low_long["response"],
                             marker_color=low_color, line_color=low_color,
                             name="Low", boxmean=False, boxpoints=False), row=1, col=1)
        fig.add_trace(go.Box(x=high_long["odorant"], y=high_long["response"],
                             marker_color=high_color, line_color=high_color,
                             name="High", boxmean=False, boxpoints=False), row=2, col=1)

        if ylim == "auto":
            def _max_whisker(ldf):
                g = ldf.groupby("odorant", sort=False)["response"]
                q3 = g.quantile(0.75)
                iqr = q3 - g.quantile(0.25)
                upper = q3 + 1.5 * iqr
                return upper.clip(upper=g.max()).max()
            y_upper = float(_nice_ceil(max(_max_whisker(low_long), _max_whisker(high_long))))
        else:
            y_upper = float(ylim)
    else:
        low_agg = _aggregate_for_plotly(low_long, estimator=estimator, errorbar=errorbar)
        high_agg = _aggregate_for_plotly(high_long, estimator=estimator, errorbar=errorbar)
        odorant_names = low_agg["odorant"].values

        def _fmt_p(name):
            p = pvals.get(name, np.nan)
            if np.isnan(p):
                return "p = n/a"
            return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

        def _build_cd(agg):
            return [[name_to_group.get(r["odorant"], ""), int(r["n"]), _fmt_p(r["odorant"])]
                    for _, r in agg.iterrows()]

        ht = ("<b>%{x}</b><br>Group: %{customdata[0]}<br>"
              "Value: %{y:.1f} \u00b1 %{error_y.array:.1f}<br>"
              "n = %{customdata[1]}<br>%{customdata[2]}<extra></extra>")

        fig.add_trace(go.Bar(
            x=odorant_names, y=low_agg["value"].values,
            error_y=dict(type="data", array=low_agg["error"].values, visible=True),
            marker_color=low_color, name="Low",
            customdata=_build_cd(low_agg), hovertemplate=ht,
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=odorant_names, y=high_agg["value"].values,
            error_y=dict(type="data", array=high_agg["error"].values, visible=True),
            marker_color=high_color, name="High",
            customdata=_build_cd(high_agg), hovertemplate=ht,
        ), row=2, col=1)

        if ylim == "auto":
            y_upper = float(_nice_ceil(max(
                (low_agg["value"] + low_agg["error"]).max(),
                (high_agg["value"] + high_agg["error"]).max(),
            )))
        else:
            y_upper = float(ylim)

    fig.update_yaxes(range=[0, y_upper], title_text="Low", row=1, col=1)
    fig.update_yaxes(range=[y_upper, 0], title_text="High", row=2, col=1)

    # Coloured tick labels
    ticktext = []
    for name in odorant_names:
        grp = name_to_group.get(name, "")
        color = group_colors.get(grp, "black")
        ticktext.append(f'<span style="color:{color}">{name}</span>')
    fig.update_xaxes(
        tickvals=list(range(len(odorant_names))),
        ticktext=ticktext, tickangle=90, row=2, col=1,
    )

    # Group boundary lines
    boundaries = np.where(np.diff(sorted_group_ids) != 0)[0] + 0.5
    for b in boundaries:
        fig.add_vline(x=b, line_dash="dash", line_color="blue", opacity=0.3, line_width=0.8)

    # Title
    if title is None:
        title = f"Parent cluster {cluster_num}"
        if subcluster is not None:
            title += f", subcluster {subcluster}"
        title += f" ({n_glom} glomeruli)"
        sort_label = {"nmf_cluster": "NMF cluster", "chemical_class": "chemical class"}.get(sort_by, sort_by)
        if estimator == "distribution":
            agg_label = "distribution (box plot)"
        else:
            eb_label = {"se": "SE", "sd": "SD", "ci": "95% CI"}.get(errorbar, errorbar) if errorbar else ""
            agg_label = f"{estimator} \u00b1 {eb_label}" if errorbar else estimator
        title += f"<br><sub>odorants grouped by {sort_label}; {agg_label}</sub>"

    fig.update_layout(
        title_text=title, width=width, height=height,
        showlegend=True, bargap=0.15, plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Reconstruction plot (Plotly replacement for matplotlib version)
# ---------------------------------------------------------------------------

def plot_reconstruction(results_dict, width=600, height=450, title="Reconstruction Analysis"):
    """Mean +/- SD reconstruction curves using Plotly.

    Parameters
    ----------
    results_dict : dict
        Output of :func:`~glom_explorer.clustering.run_reconstruction_analysis`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    colors = {"PCA": "#b0b0b0", "PCA (Shuffled)": "#b0b0b0", "NMF": "#B255E8"}
    dashes = {"PCA (Shuffled)": "dash"}

    fig = go.Figure()
    for method, df in results_dict.items():
        x = df.index.values
        mean = df.mean(axis=1).values
        std = df.std(axis=1).values

        color = colors.get(method, "#333")
        dash = dashes.get(method, "solid")

        # SD band
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([mean + std, (mean - std)[::-1]]),
            fill="toself", fillcolor=color, opacity=0.15,
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        # Mean line
        fig.add_trace(go.Scatter(
            x=x, y=mean, mode="lines+markers",
            line=dict(color=color, dash=dash, width=2),
            marker=dict(size=5) if method == "NMF" else dict(size=0),
            name=method,
            hovertemplate=f"{method}<br>Components: %{{x}}<br>R²: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        title=title, width=width, height=height,
        xaxis_title="Number of Components",
        yaxis_title="Fraction of Variance Explained",
        plot_bgcolor="white",
        legend=dict(x=0.65, y=0.05),
    )
    return fig
