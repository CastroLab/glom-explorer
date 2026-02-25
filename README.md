# glom-explorer

Interactive exploration of glomerular olfactory bulb response data.

Analyze NMF clustering, tuning profiles, and spatial topography of glomerular
responses to a 59-odorant panel across 4 subjects — all from Jupyter notebooks.

## Quick start

```bash
git clone https://github.com/CastroLab/glom-explorer.git
cd glom-explorer
pip install -e .
jupyter lab notebooks/
```

## Notebooks

| Notebook | What it covers |
|----------|---------------|
| **01_nmf_basis_selection** | PCA vs NMF reconstruction, basis scores, consensus stability — justifies k=7 |
| **02_glomerular_tuning** | Mirrored barplots per cluster, distribution mode, subclusters, variant comparison |
| **03_glomerular_topography** | Spatial facet grids: cluster maps, selectivity, sparseness, odorant groups |

## Package API

```python
import glom_explorer as gx

# Load data
odorants, low, high = gx.load_processed_data()
low_cl, high_cl = gx.load_clustered_variant('nonresponders_retained')

# Tuning barplot
fig = gx.mirrored_cluster_barplot(low_cl, high_cl, cluster_num=0, odorants=odorants)
fig.show()

# Topography grid
fig = gx.topography_facet_grid(col_var='Parent cluster', metric='nmf_selectivity')
fig.show()

# Named presets
fig = gx.topography_preset('cluster_by_subject')
fig.show()
```

## Modules

- **config** — paths, analysis defaults, NMF variant registry (`variants.json`)
- **data** — load raw + pre-clustered response matrices
- **clustering** — NMF/PCA reconstruction, basis scores, consensus stability
- **metrics** — topography metrics (amplitude, selectivity, sparseness)
- **tuning** — interactive Plotly barplots for cluster tuning profiles
- **topography** — Plotly facet grids for spatial organisation
- **utils** — data reshaping helpers

## Dependencies

numpy, pandas, scipy, scikit-learn, plotly

## Data

All data files are bundled in `glom_explorer/data/` (~1.5 MB). Pre-computed NMF
variants are registered in `glom_explorer/data/variants.json`. To add your own
variant, run NMF with a new seed/k, save the clustered CSVs, and call:

```python
gx.save_variant('my_variant', seed=12345, n_clusters=7,
                sub_clusters={0:3, 1:3, 2:3, 3:3, 4:3, 5:3, 6:3},
                files=['my_low_clustered.csv', 'my_high_clustered.csv'])
```
