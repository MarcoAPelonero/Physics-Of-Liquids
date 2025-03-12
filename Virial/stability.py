#### filepath: c:\Users\marcu\OneDrive\Desktop\Stuff\github_repos\Physics-Of-Liquids\Virial\launcher.py
import sys
import subprocess

if len(sys.argv) < 5:
    print(f"Usage: {sys.argv[0]} <max_order> <T> <volume> <range>")
    sys.exit(1)

max_order = sys.argv[1]
T         = sys.argv[2]
volume    = sys.argv[3]
r         = sys.argv[4]

cmd = ["./main.exe", max_order, T, volume, r]
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)