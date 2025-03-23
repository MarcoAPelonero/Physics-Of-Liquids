import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load the data from the file; each row has three columns: x, prop, ref
    data = np.loadtxt("analysis.txt")
    x = data[:, 0]
    prop = data[:, 1]
    ref = data[:, 2]
    
    # Create a figure with 2 subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot prop vs. x in the left subplot
    axs[0].plot(x, prop, marker='o', linestyle='-', color='blue')
    axs[0].set_title("Prop vs X")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Prop Value")
    
    # Plot ref vs. x in the right subplot
    axs[1].plot(x, ref, marker='o', linestyle='-', color='red')
    axs[1].set_title("Ref vs X")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Ref Value")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
