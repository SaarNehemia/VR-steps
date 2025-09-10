import matplotlib.pyplot as plt
import numpy as np

# Example data: replace with your own matrices
# shape: (time_samples, 40)
time_samples = 100
left_data = np.random.rand(time_samples, 40)
right_data = np.random.rand(time_samples, 40)

# Define sensor layout (x,y) coordinates
# Following your description: top to bottom
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


def plot_feet(left_values, right_values, coords, title="Foot Sole Sensors"):
    fig, axes = plt.subplots(1, 2, figsize=(8, 12))

    # Left foot
    sc1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=left_values,
                          cmap='viridis', s=500, marker='s')
    axes[0].set_title("Left Foot")
    axes[0].invert_yaxis()
    axes[0].axis('equal')

    # Right foot (mirror X axis)
    sc2 = axes[1].scatter(-coords[:, 0], coords[:, 1], c=right_values,
                          cmap='viridis', s=500, marker='s')
    axes[1].set_title("Right Foot")
    axes[1].invert_yaxis()
    axes[1].axis('equal')

    fig.colorbar(sc1, ax=axes, orientation='horizontal', fraction=0.05)
    fig.suptitle(title)
    plt.show()


# Example: plot the first time sample
plot_feet(left_data[0], right_data[0], coords)
