"""Paths, analysis defaults, and NMF variant registry."""

import json
from pathlib import Path

__all__ = [
    "DATA_DIR",
    "ANALYSIS_DEFAULTS",
    "NMF_VARIANTS",
    "load_variants",
    "save_variant",
]

# Resolve bundled data directory relative to this package
DATA_DIR = Path(__file__).resolve().parent / "data"

ANALYSIS_DEFAULTS = {
    "n_clusters": 7,
    "cluster_method": "nmf",
    "nmf_seed": 706231266,
    "threshold": 0,
    "nmf_max_iter": 1000,
    "n_shuffle_iterations": 50,
}

_VARIANTS_PATH = DATA_DIR / "variants.json"


def load_variants() -> dict:
    """Load NMF variant definitions from ``data/omp8x/variants.json``."""
    if not _VARIANTS_PATH.exists():
        return {}
    with open(_VARIANTS_PATH) as f:
        return json.load(f)


def save_variant(name: str, seed: int, n_clusters: int,
                 sub_clusters: dict, files: list[str]) -> None:
    """Register a new NMF variant (persists to ``variants.json``)."""
    variants = load_variants()
    variants[name] = {
        "seed": seed,
        "n_clusters": n_clusters,
        "sub_clusters": {str(k): v for k, v in sub_clusters.items()},
        "files": list(files),
    }
    with open(_VARIANTS_PATH, "w") as f:
        json.dump(variants, f, indent=2)
    # Update module-level dict
    global NMF_VARIANTS
    NMF_VARIANTS = variants


NMF_VARIANTS = load_variants()
