#!/usr/bin/env python3
"""
interactive_knn_segments.py
---------------------------
Visualise neighbour‑shell connectivity with **3‑D line segments** (one edge per
*k*‑nearest‑neighbour pair) coloured by the average ``q`` of the two nodes.
A Plotly + WebGL dashboard gives real‑time threshold filtering.

🔧 **What’s fixed in this version**
=================================
* Replaced ``matplotlib.cm.get_cmap`` (deprecated) with
  ``matplotlib.colormaps.get_cmap``.
* Dropped the private ``fig._get_traces()`` call – trace references are now
  stored explicitly when added, so the script runs on modern Plotly (>=5).
* Minor tidy‑ups and clearer CLI help.
* Added threshold filtering with slider and toggle switch

Run it like this
================
```bash
python interactive_knn_segments.py                  # default k = 1
python interactive_knn_segments.py --k 3            # denser graph
python interactive_knn_segments.py --cmap plasma    # different colour map
```
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import matplotlib
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# Utility routines
# -----------------------------------------------------------------------------

def _optimal_grid(n: int) -> Tuple[int, int]:
    """Return (n_rows, n_cols) arranged as square as possible."""
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)
    return n_rows, n_cols


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def _load_knn_files(input_dir: Path) -> Dict[int, pd.DataFrame]:
    """Read every ``*_ℓ.txt`` file into ``{ℓ: DataFrame}``."""
    pattern = re.compile(r".*?_(\d+)\.txt$", re.IGNORECASE)
    cols = ["idx", "x", "y", "z", "q"]
    data: Dict[int, pd.DataFrame] = {}
    for fname in sorted(input_dir.iterdir()):
        if fname.suffix.lower() != ".txt":
            continue
        m = pattern.match(fname.name)
        if not m:
            continue
        l_val = int(m.group(1))
        df = pd.read_csv(
            fname,
            sep=r"\t",
            header=None,
            names=cols,
            skiprows=2,
            usecols=range(5),
            engine="python",
        )
        data[l_val] = df
    if not data:
        raise FileNotFoundError(
            f"No '*_ℓ.txt' neighbour files found in directory '{input_dir}'."
        )
    return data


# -----------------------------------------------------------------------------
# Edge construction (k‑NN)
# -----------------------------------------------------------------------------

def _knn_edges(df: pd.DataFrame, k: int = 1) -> List[Tuple[int, int, float]]:
    """Return list of (i, j, q_avg) where *i < j* are row indices in *df*."""
    coords = df[["x", "y", "z"]].to_numpy()
    q_vals = df["q"].to_numpy()
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    inds = nn.kneighbors(return_distance=False)
    edges: List[Tuple[int, int, float]] = []
    for i, neighbours in enumerate(inds):
        for j in neighbours[1:]:  # skip self
            if i < j:  # avoid duplicates
                edges.append((i, j, 0.5 * (q_vals[i] + q_vals[j])))
    return edges


# -----------------------------------------------------------------------------
# Colour helper
# -----------------------------------------------------------------------------

def _colour_map(values: np.ndarray, cmap_name: str) -> List[str]:
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    return [mcolors.to_hex(cmap(norm(v))) for v in values]


# -----------------------------------------------------------------------------
# Plotly dashboard
# -----------------------------------------------------------------------------

def _build_plotly_dashboard(data: Dict[int, pd.DataFrame], k: int, cmap: str, mode: str, percentile: float):
    shells = sorted(data)
    n_rows, n_cols = _optimal_grid(len(shells))

    # Pre‑compute edges and q_avg list
    edge_dict: Dict[int, Dict[str, np.ndarray]] = {}
    global_qavg: List[float] = []
    for ℓ, df in data.items():
        edges = _knn_edges(df, k)
        if not edges:
            continue
        idx_i, idx_j, qavg = zip(*edges)
        edge_dict[ℓ] = {
            "idx_i": np.array(idx_i),
            "idx_j": np.array(idx_j),
            "qavg": np.array(qavg),
        }
        global_qavg.extend(qavg)

    global_qavg = np.array(global_qavg)

    # Filtering
    threshold = np.percentile(global_qavg, percentile)
    keep_if = (lambda q: q <= threshold) if mode == "up" else (lambda q: q >= threshold)

    # Colours
    colour_pool = _colour_map(global_qavg, cmap)
    colour_cursor = 0

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)],
        subplot_titles=[f"ℓ = {ℓ}" for ℓ in shells],
        horizontal_spacing=0.02,
        vertical_spacing=0.04,
    )

    for pos, ℓ in enumerate(shells, 1):
        row = (pos - 1) // n_cols + 1
        col = (pos - 1) % n_cols + 1
        df = data[ℓ]
        if ℓ not in edge_dict:
            continue
        ed = edge_dict[ℓ]
        coords = df[["x", "y", "z"]].to_numpy()
        qavg_vals = ed["qavg"]
        n_edges = len(qavg_vals)
        colours = colour_pool[colour_cursor: colour_cursor + n_edges]
        colour_cursor += n_edges

        for (i, j, q), col_hex in zip(zip(ed["idx_i"], ed["idx_j"], ed["qavg"]), colours):
            if not keep_if(q):
                continue
            trace = go.Scatter3d(
                x=[coords[i, 0], coords[j, 0]],
                y=[coords[i, 1], coords[j, 1]],
                z=[coords[i, 2], coords[j, 2]],
                mode="lines",
                line=dict(color=col_hex, width=2),
                hoverinfo="none",
                showlegend=False,
            )
            fig.add_trace(trace, row=row, col=col)

    fig.update_layout(
        title=f"Neighbour shells with percentile filter ({mode} {percentile}%)",
        margin=dict(l=0, r=0, b=0, t=40),
        height=300 * n_rows + 120,
    )
    fig.show(renderer="browser")

# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="3‑D neighbour‑shell segments with percentile filtering",
    )
    parser.add_argument(
        "--input_dir",
        default="knn_results",
        help="folder containing *_ℓ.txt files (default: knn_results)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="k in k‑NN graph (default: 1)",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap name (default: viridis)",
    )
    parser.add_argument(
        "--mode",
        choices=["up", "down"],
        default="down",
        help="Remove lines above or below the given percentile of q_avg",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=80.0,
        help="Percentile value (e.g., 80 means remove above 80th if mode is 'up')",
    )
    args = parser.parse_args()

    data = _load_knn_files(Path(args.input_dir))
    _build_plotly_dashboard(
        data,
        k=args.k,
        cmap=args.cmap,
        mode=args.mode,
        percentile=args.percentile,
    )

if __name__ == "__main__":
    main()
