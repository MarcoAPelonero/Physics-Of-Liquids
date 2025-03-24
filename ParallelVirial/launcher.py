#!/usr/bin/env python3

import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from tqdm import tqdm

def run_simulation(cmd):
    """Runs one simulation command via subprocess, returns (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return (result.returncode, result.stdout, result.stderr)

def main():
    # Basic parameters
    order = 3
    nSamples = 1000000
    dimension = 3
    sigma = 1.0
    epsilon = 1.0
    T_list = [0.625, 0.75, 1.0, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 5.0, 10.0]

    # If it exists delte previous output folder
    '''
    if os.path.exists("output"):
        import shutil
        shutil.rmtree("output")
    '''
    os.makedirs("output", exist_ok=True)
    
    # Build initial queue of jobs. Each item is (cmd, T_val, retries).
    # We'll allow up to MAX_RETRIES attempts per T.
    MAX_RETRIES = 3
    jobs = deque()
    for T_val in T_list:
        outfilename = f"output/results_T_{T_val}.txt"
        cmd = [
            "./mayerSimulation.exe", 
            str(order),
            str(nSamples),
            str(dimension),
            str(sigma),
            str(epsilon),
            str(T_val),
            outfilename
        ]
        jobs.append((cmd, T_val, 0))

    print("Launching simulations in parallel with re-try on failure...")

    # We'll track total simulations we do so we know the progress
    total_runs = len(jobs)
    completed_runs = 0  # how many jobs have successfully finished

    # We process jobs in batches of up to 3 in parallel:
    with ThreadPoolExecutor(max_workers=3) as executor:
        # We'll keep track of the future->(cmd,T_val,retries) so we know what just finished
        running_futures = {}

        # We load up our initial batch of jobs
        while jobs or running_futures:
            # Fill up to 3 concurrent tasks
            while len(running_futures) < 3 and jobs:
                cmd, T_val, retries = jobs.popleft()
                future = executor.submit(run_simulation, cmd)
                running_futures[future] = (cmd, T_val, retries)

            # As each future completes, handle success/fail
            done_futures = []
            for future in running_futures:
                if future.done():
                    done_futures.append(future)

            for future in done_futures:
                cmd, T_val, retries = running_futures.pop(future)
                returncode, stdout, stderr = future.result()

                if returncode == 0:
                    # Success
                    completed_runs += 1
                    print(f"Simulation for T={T_val} completed successfully. "
                          f"({completed_runs}/{total_runs} done)")
                else:
                    # Failure
                    if retries < MAX_RETRIES:
                        new_retries = retries + 1
                        print(f"Simulation for T={T_val} failed (attempt {new_retries}). Retrying...")
                        jobs.append((cmd, T_val, new_retries))
                    else:
                        print(f"Simulation for T={T_val} failed (attempt {retries+1}) "
                              f"and reached max retries. Error:\n{stderr}")

            # Optional: use a small sleep or so to avoid busy looping
            # time.sleep(0.1)

    print("All simulations done (or reached max retries).")

if __name__ == "__main__":
    main()
