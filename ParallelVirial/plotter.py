#!/usr/bin/env python3

import os
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Make plots look nicer
    sns.set_theme()

    # Reference data for B2/b and B3/b vs T*
    # (excerpted from the table in your posted image).
    # Only used for the T* values that exactly match your simulation runs.
    reference_data = {
        2: {  # B2/b
            0.625: -5.7578,
            0.75:  -4.1757,
            1.0:   -2.53812,
            1.2:   -1.8359,
            1.3:   -1.58421,
            1.4:   -1.3759,
            1.5:   -1.2010,
            2.0:   -0.6275,
            2.5:   -0.3126,
            5.0:    0.24332,
            10.0:   0.46087,
        },
        3: {  # B3/b
            0.625: -8.237,
            0.75:  -1.7923,
            1.0:    0.4299,
            1.2:    0.5922,
            1.3:    0.5881,
            1.4:    0.5682,
            1.5:    0.5433,
            2.0:    0.43703,
            2.5:    0.38100,
            5.0:    0.31507,
            10.0:   0.2860,
        }
    }

    # Dictionary for simulation results: results[n][T] = virial_coefficient
    results = {}

    # Read all result files from output/ folder
    for fname in os.listdir("output"):
        if not fname.startswith("results_T_"):
            continue
        filepath = os.path.join("output", fname)
        
        # Extract the T* value from the filename or from inside the file
        with open(filepath, "r") as f:
            lines = f.read().splitlines()
        
        # The line with T* = ...
        T_line = next((l for l in lines if l.startswith("T* = ")), None)
        if not T_line:
            continue

        # Parse out the temperature
        T_str = T_line.split("=")[1].strip()
        T_val = float(T_str)

        # Now parse out lines that look like: n=2:  -2.38656
        for line in lines:
            m = re.match(r"^n=(\d+):\s+([-\d.]+)", line)
            if m:
                n = int(m.group(1))
                val_str = m.group(2)
                val = float(val_str)
                if n not in results:
                    results[n] = {}
                results[n][T_val] = val

    if not results:
        print("No simulation data found in the output folder. Exiting.")
        return

    # Identify which orders we actually have
    max_order = max(results.keys())

    # Plot from n=2 up to the max order in the data
    for n in range(2, max_order + 1):
        if n not in results:
            continue

        # Gather the T* and Bn/b from the simulation
        Tvals_sim = sorted(results[n].keys())
        if not Tvals_sim:
            continue

        # Build arrays for the plot
        sim_x = np.array(Tvals_sim, dtype=float)
        sim_y = np.array([results[n][T] for T in Tvals_sim], dtype=float)

        # Reference data at the same T*, if available
        ref_x = []
        ref_y = []
        if n in reference_data:
            for T_ref, B_ref in reference_data[n].items():
                # Only plot if we actually have that T in sim
                if T_ref in sim_x:
                    ref_x.append(T_ref)
                    ref_y.append(B_ref)

        # Make a figure per order
        plt.figure(figsize=(6,4))
        plt.title(f"Virial Coefficient (Order n={n})")
        plt.xlabel("T*")
        plt.ylabel(f"B{n}/b")

        # Plot simulation points
        plt.scatter(sim_x, sim_y, color="blue", label="Simulation")

        # Plot reference points (if any)
        if ref_x:
            plt.scatter(ref_x, ref_y, color="red", label="Reference")

        # Fit a polynomial curve through the simulation data, up to degree 10
        # (If fewer points than 2, skip.)
        if len(sim_x) > 1:
            degree = min(5, len(sim_x) - 1)
            coeffs = np.polyfit(sim_x, sim_y, deg=degree)
            poly_x = np.linspace(sim_x.min(), sim_x.max(), 200)
            poly_y = np.polyval(coeffs, poly_x)
            # plt.plot(poly_x, poly_y, color="green", label="Poly fit")

        plt.legend()
        plt.tight_layout()

        # Save one plot per order
        out_plot = f"plot_order_{n}.png"
        plt.savefig(out_plot, dpi=150)
        plt.close()
        print(f"Saved {out_plot}.")

if __name__ == "__main__":
    main()
