import re
import numpy as np  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
from matplotlib.animation import FuncAnimation

def read_data(file_path):
    """
    Reads the simulation output file and parses data into a list of steps.
    Each step is separated by a blank line.
    """
    with open(file_path, 'r') as f:
        content = f.read().strip()
    
    # Split by double newlines (assumes each step is separated by an empty line)
    step_blocks = content.split("\n\n")
    frames = []
    
    for block in step_blocks:
        lines = block.strip().splitlines()
        step_info = {}
        particles = []
        for line in lines:
            if line.startswith("Step:"):
                step_info['step'] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Number of Particles:"):
                step_info['num_particles'] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Time Step:"):
                step_info['time_step'] = float(line.split(":", 1)[1].strip())
            elif line.startswith("Gravity:"):
                # Expecting format: Gravity: (0, 0, -9.81)
                gravity_str = line.split("Gravity:", 1)[1].strip().strip("()")
                gravity = tuple(float(x.strip()) for x in gravity_str.split(","))
                step_info['gravity'] = gravity
            elif line.startswith("Box Dimensions:"):
                # Expecting format: Box Dimensions: (10, 10, 10)
                box_str = line.split("Box Dimensions:", 1)[1].strip().strip("()")
                box = tuple(float(x.strip()) for x in box_str.split(","))
                step_info['box'] = box
            elif line.startswith("Particle Position:"):
                # Use regex to parse:
                # Particle Position: (x, y, z) Velocity: (vx, vy, vz) Acceleration: (ax, ay, az) Radius: r
                pattern = r"Particle Position:\s*\(([^)]+)\)\s*Velocity:\s*\(([^)]+)\)\s*Acceleration:\s*\(([^)]+)\)\s*Radius:\s*(\S+)"
                match = re.search(pattern, line)
                if match:
                    pos_str, vel_str, acc_str, radius_str = match.groups()
                    pos = tuple(float(x.strip()) for x in pos_str.split(","))
                    # velocity and acceleration are available if needed in the future
                    vel = tuple(float(x.strip()) for x in vel_str.split(","))
                    acc = tuple(float(x.strip()) for x in acc_str.split(","))
                    radius = float(radius_str)
                    particle = {"position": pos, "velocity": vel, "acceleration": acc, "radius": radius}
                    particles.append(particle)
        step_info['particles'] = particles
        frames.append(step_info)
    
    return frames

def plot_sphere(ax, center, radius, color='b', resolution=20):
    """
    Plots a sphere on the given 3D axis.
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, shade=True)

def animate(frame_index, frames, ax):
    """
    Clears the axes and plots the spheres for the given frame.
    """
    ax.clear()
    frame = frames[frame_index]
    box = frame.get('box', (10, 10, 10))
    particles = frame.get('particles', [])
    
    # Set axis limits based on the box dimensions
    ax.set_xlim(0, box[0])
    ax.set_ylim(0, box[1])
    ax.set_zlim(0, box[2])

    ax.set_box_aspect((box[0], box[1], box[2]))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Step: {frame.get('step', frame_index)}")
    
    # Plot each particle as a sphere
    for particle in particles:
        pos = particle["position"]
        radius = particle["radius"]
        plot_sphere(ax, pos, radius)

def main():
    # Read the simulation data from output.txt
    frames = read_data("output.txt")
    
    # Create the figure and 3D axis for animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create an animation that updates the plot for each frame.
    anim = FuncAnimation(fig, animate, frames=len(frames), fargs=(frames, ax), interval=500, repeat=True)
    
    # Display the animation window.
    plt.show()

if __name__ == "__main__":
    main()
