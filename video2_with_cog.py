import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Example data: replace with your real data
time_samples = 200
left_data = np.random.rand(time_samples, 40)
right_data = np.random.rand(time_samples, 40)

# Define sensor layout (same as before)
coords = []
coords += [(i, 0) for i in range(3)]  # 1st row
for row in range(1, 6): coords += [(i, row) for i in range(4)]  # 5 rows of 4
for row in range(6, 9): coords += [(i, row) for i in range(2)]  # 3 rows of 2
for row in range(9, 12): coords += [(i, row) for i in range(3)]  # 3 rows of 3
coords += [(i, 12) for i in range(2)]  # last row
coords = np.array(coords)


def compute_cog(sensor_values, coords, mirror_x=False):
    if mirror_x:
        xs = -coords[:, 0]
    else:
        xs = coords[:, 0]
    ys = coords[:, 1]
    weights = sensor_values
    if np.sum(weights) == 0:
        return np.mean(xs), np.mean(ys)
    cx = np.sum(xs * weights) / np.sum(weights)
    cy = np.sum(ys * weights) / np.sum(weights)
    return cx, cy


def run_game_with_feet(left_data, right_data, coords):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # --- Left foot ---
    sc_left = axes[0].scatter(coords[:, 0], coords[:, 1], c=left_data[0],
                              cmap='viridis', s=300, marker='s')
    axes[0].set_title("Left Foot")
    axes[0].invert_yaxis()
    axes[0].axis('equal')

    # --- Right foot (mirrored) ---
    sc_right = axes[1].scatter(-coords[:, 0], coords[:, 1], c=right_data[0],
                               cmap='viridis', s=300, marker='s')
    axes[1].set_title("Right Foot")
    axes[1].invert_yaxis()
    axes[1].axis('equal')

    # --- Character movement ---
    char_pos = np.array([0.0, 0.0])
    char_dot, = axes[2].plot([], [], 'ro', markersize=12)
    trail, = axes[2].plot([], [], 'b-', alpha=0.5)  # path trail
    path_x, path_y = [], []
    axes[2].set_xlim(-10, 10)
    axes[2].set_ylim(-10, 10)
    axes[2].set_title("Character Movement")
    axes[2].grid(True)

    # Shared colorbar
    fig.colorbar(sc_left, ax=axes[:2], orientation='horizontal', fraction=0.05)

    def update(frame):
        nonlocal char_pos, path_x, path_y

        # Update feet
        sc_left.set_array(left_data[frame])
        sc_right.set_array(right_data[frame])

        # Compute CoG
        cog_left = compute_cog(left_data[frame], coords, mirror_x=False)
        cog_right = compute_cog(right_data[frame], coords, mirror_x=True)
        cog = ((cog_left[0] + cog_right[0]) / 2,
               (cog_left[1] + cog_right[1]) / 2)

        # Move character
        velocity = np.array(cog) * 0.05
        char_pos = char_pos + velocity
        char_dot.set_data([char_pos[0]], [char_pos[1]])

        # Trail
        path_x.append(char_pos[0])
        path_y.append(char_pos[1])
        trail.set_data(path_x, path_y)

        fig.suptitle(f"Frame {frame+1}/{len(left_data)} | CoG=({cog[0]:.2f},{cog[1]:.2f})")
        return sc_left, sc_right, char_dot, trail

    ani = FuncAnimation(fig, update, frames=len(left_data), interval=100, blit=False)
    plt.show()
    return ani


# Run
ani = run_game_with_feet(left_data, right_data, coords)
