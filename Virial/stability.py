import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Parameters for the simulation:
max_order = 3         
T = 1.0              
numSamples = 1000000
volumes = np.linspace(5,30, 4)  
num_simulations = 5

# Dictionary to store results.
# Structure: results[order] = {volume: [list of B_n values from each simulation]}
results = {order: {} for order in range(1, max_order + 1)}

# Loop over volume values and repeated simulations.
for vol in tqdm(volumes, desc="Volumes", unit="volume", total=len(volumes)):
    for sim in range(num_simulations):
        # Build the command arguments.
        cmd = [
            "./main.exe",   # assuming the executable is in the same directory
            str(max_order),
            str(T),
            str(vol),
            str(numSamples)
        ]
        # Launch the simulation.
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running simulation for volume={vol} on simulation #{sim+1}")
            print(e.stderr)
            continue

        # The program prints a line for each order like: "Computed virial coefficient for n = 3: <value>"
        # We'll extract the order and the computed coefficient from stdout.
        for line in proc.stdout.splitlines():
            if "Computed virial coefficient for n =" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    try:
                        left, value_str = parts[0], parts[1]
                        order_str = left.split("=")[-1].strip()
                        order = int(order_str)
                        Bn = float(value_str.strip())
                        if vol not in results[order]:
                            results[order][vol] = []
                        results[order][vol].append(Bn)
                    except Exception as e:
                        print("Error parsing line:", line, e)

# First plot: all data points (scatter) and average lines
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, max_order))

# Skip order 1 (and order 0 if it ever existed) by starting at order 2.
for order in range(2, max_order + 1):
    vol_vals = sorted(results[order].keys())
    legend_added = False  # flag to ensure we only label the first point for each order
    for vol in vol_vals:
        sim_values = results[order][vol]
        for val in sim_values:
            if not legend_added:
                plt.scatter(vol, val, color=colors[order-1], label=f"Order {order}")
                legend_added = True
            else:
                plt.scatter(vol, val, color=colors[order-1])
    # Connect the average values over simulations with a dashed line.
    avg_vals = [np.mean(results[order][v]) for v in vol_vals]
    plt.plot(vol_vals, avg_vals, color=colors[order-1], linestyle='--')

plt.xlabel("Volume (integration range)")
plt.ylabel("Virial Coefficient")
plt.title(f"Stability of Virial Coefficients with Changing Volume [numSamples={numSamples}]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Second plot: only the average for each order per volume
plt.figure(figsize=(10, 6))

# Dictionary to store the averages for printing and saving
averages = {}

for order in range(2, max_order + 1):
    vol_vals = sorted(results[order].keys())
    avg_vals = [np.mean(results[order][v]) for v in vol_vals]
    averages[order] = list(zip(vol_vals, avg_vals))
    plt.plot(vol_vals, avg_vals, marker='o', label=f"Order {order}")

plt.xlabel("Volume (integration range)")
plt.ylabel("Average Virial Coefficient")
plt.title(f"Average Virial Coefficients vs Volume [numSamples={numSamples}]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the average values to the terminal and write into a text file.
output_lines = []
for order in sorted(averages.keys()):
    output_lines.append(f"Order {order}:")
    print(f"Order {order}:")
    for vol, avg in averages[order]:
        line = f"  Volume: {vol:.4f} -> Average: {avg:.4f}"
        output_lines.append(line)
        print(line)
    output_lines.append("")  # blank line for readability

with open("averages.txt", "w") as f:
    f.write("\n".join(output_lines))
