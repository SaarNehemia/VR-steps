from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from json_utils import load_json

# Define sensor layout (x,y) coordinates
coords = []

# First line: 3 sensors
coords += [(i, 0) for i in range(3)]

# Next 5 lines of 4 sensors
for row in range(1, 6):
    coords += [(i, row) for i in range(4)]

# Next 3 lines of 2 sensors
for row in range(6, 9):
    coords += [(i, row) for i in range(2)]

# Next 3 lines of 3 sensors
for row in range(9, 12):
    coords += [(i, row) for i in range(3)]

# Last line: 2 sensors
coords += [(i, 12) for i in range(2)]

coords = np.array(coords)


# --- Animation Function ---
def animate_feet(left_data, right_data, coords, name, save_as=None):
    fig, axes = plt.subplots(1, 2, figsize=(8, 12))

    # Initial scatter plots
    sc_left = axes[0].scatter(coords[:, 0], coords[:, 1], c=left_data[0],
                              cmap='viridis', s=500, marker='s')
    axes[0].set_title("Left Foot")
    axes[0].invert_yaxis()
    axes[0].axis('equal')

    sc_right = axes[1].scatter(-coords[:, 0], coords[:, 1], c=right_data[0],
                               cmap='viridis', s=500, marker='s')
    axes[1].set_title("Right Foot")
    axes[1].invert_yaxis()
    axes[1].axis('equal')

    # Shared colorbar
    cbar = fig.colorbar(sc_left, ax=axes, orientation='horizontal', fraction=0.05)
    cbar.set_label("Sensor Value")

    def update(frame):
        sc_left.set_array(left_data[frame])
        sc_right.set_array(right_data[frame])
        fig.suptitle(f"Name: {name}\nFrame: {frame + 1}/{len(left_data)}")
        return sc_left, sc_right

    ani = FuncAnimation(fig, update, frames=len(left_data), interval=100, blit=False)

    # Save if requested
    if save_as:
        if save_as.endswith(".mp4"):
            writer = FFMpegWriter(fps=10, bitrate=1800)
            ani.save(save_as, writer=writer)
        elif save_as.endswith(".gif"):
            writer = PillowWriter(fps=10)
            ani.save(save_as, writer=writer)
            print(f"Animation saved as {save_as}")
    else:
        plt.show()


def load_data(save_name=None):
    json_folder = Path(
        r"G:\My Drive\הקוצ'ינים הצעירים\israeli-Indian Hackathon\Info for Participants\VR steps\Data\New data 18.08.25")
    json_name = r"Copy of pedisol_segment_0-8-sitdown"
    # json_name = r"Copy of pedisol_segment_0-49-standup"
    # json_name = r"Copy of pedisol_segment_0-48"
    # json_name = r"Copy of pedisol_segment_0-603"

    save_name = f"{json_name}.gif"  # put None if you don't wont to save.

    json_path = json_folder.joinpath(json_name + '.json')
    data = load_json(json_path)

    # List of samples where each sample is a dict with the following keys:
    # id (string)
    # Session (string)
    # Expire (seconds number, nanoseconds number)
    # R (list)
    # L (list)
    # T (number)

    t = np.sort([data[i]["T"] for i in range(len(data))])
    right_data = np.vstack([data[i]["R"] for i in range(len(data))])
    left_data = np.vstack([data[i]["L"] for i in range(len(data))])

    return left_data, right_data, json_name, save_name


if __name__ == '__main__':
    to_load = False
    if to_load:
        # Load data
        left_data, right_data, name, save_name = load_data()
    else:
        # Generate data (shape: (time_samples, 40))
        time_samples = 100
        left_data = np.random.rand(time_samples, 40)
        right_data = np.random.rand(time_samples, 40)
        name = 'example'
        save_name = None

    animate_feet(left_data=left_data, right_data=right_data, coords=coords,
                 name=name, save_as=save_name)
