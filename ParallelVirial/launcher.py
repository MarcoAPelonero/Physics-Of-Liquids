#!/usr/bin/env python3
import argparse
import subprocess
import os
import re
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List

# Maximum number of attempts per simulation run
MAX_RETRIES = 3

def run_simulation(cmd: List[str]) -> Tuple[int, str, str]:
    """Runs a simulation command via subprocess and returns (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def parse_simulation_output(output: str) -> dict:
    """
    Parse simulation output text.
    Expected lines (example):
      n = 2 virial coefficient (packing fraction): 0.12345
    Returns a dictionary mapping order (int) to coefficient (float).
    """
    pattern = re.compile(r"n\s*=\s*(\d+)\s+virial coefficient \(packing fraction\):\s*([\d\.\-Ee]+)")
    results = {}
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            order = int(match.group(1))
            value = float(match.group(2))
            results[order] = value
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Run Mayer simulation for multiple temperatures and average outputs."
    )
    parser.add_argument("--no-multithreading", action="store_true",
                        help="Disable multithreading and run simulations sequentially.")
    parser.add_argument("--runs-per-temp", type=int, default=5,
                        help="Number of simulation runs per temperature.")
    parser.add_argument("--executable", type=str, default="./main.exe",
                        help="Path to the simulation executable.")
    parser.add_argument("--nSamples", type=int, default=100000000,
                        help="Number of Monte Carlo samples (2nd argument to the executable).")
    parser.add_argument("--dimension", type=int, default=3,
                        help="Dimension (3rd argument to the executable).")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Sigma parameter (4th argument to the executable).")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Epsilon parameter (5th argument to the executable).")
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.625, 0.75, 1.0, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 5.0, 10.0],
                        help="List of temperatures to run simulations.")
    parser.add_argument("--max-workers", type=int, default=3,
                        help="Maximum number of workers for multithreading.")
    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs("output", exist_ok=True)

    # Build a queue of jobs.
    # For each temperature we run multiple simulations.
    # Each individual run writes its own temporary file (which we aggregate later).
    jobs = deque()
    for T_val in args.temperatures:
        for run_index in range(1, args.runs_per_temp + 1):
            # The command-line arguments for the simulation executable are:
            # [executable, graph_file, nSamples, dimension, sigma, epsilon, T, outfilename]
            outfilename = f"output/results_T_{T_val}_run_{run_index}.txt"
            cmd = [
                args.executable,
                "graphs.dat",  # Supply the graph file as expected by your simulation
                str(args.nSamples),
                str(args.dimension),
                str(args.sigma),
                str(args.epsilon),
                str(T_val),
                outfilename
            ]
            jobs.append((cmd, T_val, run_index, 0))
    total_jobs = len(jobs)
    completed_jobs = 0

    # Dictionary to collect outputs by temperature.
    # For each temperature we'll store the stdout from each successful simulation run.
    results_by_temp = {T: [] for T in args.temperatures}

    print("Launching simulations...")
    if args.no_multithreading:
        # Run jobs sequentially.
        while jobs:
            cmd, T_val, run_number, retries = jobs.popleft()
            ret, stdout, stderr = run_simulation(cmd)
            if ret == 0:
                completed_jobs += 1
                print(f"Simulation for T={T_val}, run {run_number} completed successfully "
                      f"({completed_jobs}/{total_jobs}).")
                results_by_temp[T_val].append(stdout)
            else:
                if retries < MAX_RETRIES:
                    new_retries = retries + 1
                    print(f"Simulation for T={T_val}, run {run_number} failed (attempt {new_retries}). Retrying...")
                    jobs.append((cmd, T_val, run_number, new_retries))
                else:
                    print(f"Simulation for T={T_val}, run {run_number} failed (attempt {retries+1}) "
                          f"and reached max retries. Error:\n{stderr}")
    else:
        # Run jobs in parallel using ThreadPoolExecutor.
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            running_futures = {}
            while jobs or running_futures:
                # Fill available workers
                while len(running_futures) < args.max_workers and jobs:
                    cmd, T_val, run_number, retries = jobs.popleft()
                    future = executor.submit(run_simulation, cmd)
                    running_futures[future] = (cmd, T_val, run_number, retries)
                # Process completed futures
                done_futures = [f for f in running_futures if f.done()]
                for future in done_futures:
                    cmd, T_val, run_number, retries = running_futures.pop(future)
                    ret, stdout, stderr = future.result()
                    if ret == 0:
                        completed_jobs += 1
                        print(f"Simulation for T={T_val}, run {run_number} completed successfully "
                              f"({completed_jobs}/{total_jobs}).")
                        results_by_temp[T_val].append(stdout)
                    else:
                        if retries < MAX_RETRIES:
                            new_retries = retries + 1
                            print(f"Simulation for T={T_val}, run {run_number} failed (attempt {new_retries}). Retrying...")
                            jobs.append((cmd, T_val, run_number, new_retries))
                        else:
                            print(f"Simulation for T={T_val}, run {run_number} failed (attempt {retries+1}) "
                                  f"and reached max retries. Error:\n{stderr}")

    # Aggregate the results from all runs per temperature.
    averaged_results = {}
    for T_val, outputs in results_by_temp.items():
        coefficients = defaultdict(list)
        for output in outputs:
            parsed = parse_simulation_output(output)
            for order, value in parsed.items():
                coefficients[order].append(value)
        averages = {}
        for order, values in coefficients.items():
            if values:
                avg = sum(values) / len(values)
                averages[order] = avg
        averaged_results[T_val] = averages

    # Write one output file per temperature (readable by your plotter)
    for T_val, averages in averaged_results.items():
        filename = f"output/results_T_{T_val}.txt"
        with open(filename, "w") as f:
            f.write(f"T* = {T_val}\n")
            for order in sorted(averages.keys()):
                f.write(f"n={order}: {averages[order]}\n")
        print(f"Averaged result written to {filename}")

if __name__ == "__main__":
    main()
