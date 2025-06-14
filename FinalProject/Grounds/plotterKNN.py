#!/usr/bin/env python3
"""
make_knn_images.py
------------------
Create one 3-D scatter plot per neighbour shell `l` found in `knn_results/*.txt`.

Usage
-----
# save the figures to disk (default)
python make_knn_images.py

# view the figures interactively instead
python make_knn_images.py --live
"""

import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D             # noqa: F401 (side-effect sets 3-D backend)

def make_knn_images(input_dir="knn_results",
                    output_dir="knnImages",
                    live=False,
                    cmap="viridis"):
    """
    Parameters
    ----------
    input_dir : str
        Folder that holds the LAMMPS neighbour-list text files.
    output_dir : str
        Folder where PNGs will be saved (created if it doesn’t exist).
    live : bool
        If True, plots are shown interactively; if False, they are saved to disk.
    cmap : str
        Any valid Matplotlib colormap for `q_l_atom` (default "viridis").
    """
    os.makedirs(output_dir, exist_ok=True)
    cols = ["idx", "x", "y", "z", "q_l_atom"]
    pattern = re.compile(r".*?_(\d+)\.txt$", re.IGNORECASE)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".txt"):
            continue
        m = pattern.match(fname)
        if not m:
            # skip files that don’t follow the expected “*_l.txt” pattern
            continue

        l = int(m.group(1))
        df = pd.read_csv(
            os.path.join(input_dir, fname),
            sep=r"\t",
            header=None,
            names=cols,
            skiprows=2,
            usecols=range(5),
            engine='python'  # to handle regex in sep
        )

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            df["x"], df["y"], df["z"],
            c=df["q_l_atom"],
            cmap=cmap,
            s=6,           # point size; tweak if you like
            alpha=0.8
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"l = {l}")
        cbar = fig.colorbar(scatter, ax=ax, pad=0.10, shrink=0.75)
        cbar.set_label("q_l_atom")

        if live:
            plt.show()
        else:
            outfile = os.path.join(output_dir, f"l_{l}.png")
            fig.savefig(outfile, dpi=300, bbox_inches="tight")
            plt.close(fig)

def _parse_cli():
    parser = argparse.ArgumentParser(description="Generate 3-D scatter plots of knn_results.")
    parser.add_argument("--input_dir", default="knn_results",
                        help="directory that holds the *.txt neighbour files (default: knn_results)")
    parser.add_argument("--output_dir", default="knnImages",
                        help="directory where PNGs will be written (default: knnImages)")
    parser.add_argument("--live", action="store_true",
                        help="show plots on screen instead of saving them")
    parser.add_argument("--cmap", default="viridis",
                        help="matplotlib colormap for q_l_atom colouring (default: viridis)")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_cli()
    make_knn_images(args.input_dir, args.output_dir, args.live, args.cmap)
