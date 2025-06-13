import numpy as np
import subprocess
import sys

# Define the range of radii: 10 values evenly spaced between 1 and 5
radii = np.linspace(1.2, 6.0, 15)

# Path to the orderParameters script
SCRIPT = "orderParameters.py"

# Iterate through each radius and invoke the script
for r in radii:
    print(f"Running orderParameters.py with cutoff radius = {r:.3f}")
    result = subprocess.run([sys.executable, SCRIPT, "-r", f"{r:.6f}"],
                            capture_output=True,
                            text=True)
    # Print the script's stdout and stderr
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error (exit code {result.returncode}):", file=sys.stderr)
        print(result.stderr, file=sys.stderr)

print("All runs completed.")