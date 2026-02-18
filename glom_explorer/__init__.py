"""glom-explorer: Interactive exploration of glomerular olfactory bulb data.

Usage::

    import glom_explorer as gx

    odorants, low, high = gx.load_processed_data()
    low_cl, high_cl = gx.load_clustered_variant()
    fig = gx.mirrored_cluster_barplot(low_cl, high_cl, 0, odorants)
    fig.show()
"""

__version__ = "0.1.0"

# -- Config --
from .config import (
    DATA_DIR,
    ANALYSIS_DEFAULTS,
    NMF_VARIANTS,
    load_variants,
    save_variant,
)

# -- Data loading --
from .data import (
    load_odorants,
    load_glomerular_data,
    load_xy_coords,
    load_processed_data,
    load_clustered_variant,
    load_clustered_data,
)

# -- Clustering & analysis --
from .clustering import (
    assign_cluster_indices,
    apply_cluster_indices,
    permute_by_cluster_indices,
    reorder_by_clusters,
    query_data_cluster,
    run_reconstruction_analysis,
    calc_nmf_basis_scores,
    consensus_clustering_stability,
)

# -- Metrics --
from .metrics import (
    compute_metric,
    response_amplitude,
    binary_cluster_membership,
    nmf_class_selectivity,
    chemical_class_selectivity,
    lifetime_sparseness,
)

# -- Tuning visualizations --
from .tuning import (
    mirrored_cluster_barplot,
    plot_reconstruction,
)

# -- Topography visualizations --
from .topography import (
    topography_facet_grid,
    topography_preset,
)

# -- Utilities --
from .utils import (
    melt_results_dict,
    melt_all_results,
)
