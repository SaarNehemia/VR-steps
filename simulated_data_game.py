import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Sole mask layout (row-major indexing, top-left first) ---
def sole_mask():
    mask = np.full((13, 4), np.nan)  # 13 rows, max 4 sensors
    idx = 0

    # row 0: 3 sensors
    for c in range(3):
        mask[0, c] = idx; idx += 1
    # rows 1–5: 4 sensors
    for r in range(1,6):
        for c in range(4):
            mask[r, c] = idx; idx += 1
    # rows 6–8: 2 sensors (centered)
    for r in range(6,9):
        for c in range(1,3):
            mask[r, c] = idx; idx += 1
    # rows 9–11: 3 sensors (centered)
    for r in range(9,12):
        for c in range(3):
            mask[r, c] = idx; idx += 1
    # row 12: 2 sensors
    for c in range(2):
        mask[12, c] = idx; idx += 1

    return mask

mask = sole_mask()
rows, cols = mask.shape
n_sensors = int(np.nanmax(mask) + 1)
print("Sensors per foot:", n_sensors)

# --- Mapping from sensor index to row/col ---
idx_map = {int(mask[r,c]): (r,c)
           for r in range(rows) for c in range(cols)
           if not np.isnan(mask[r,c])}

def frame_to_grid(values, mirror=False):
    """Fill a 2D grid from sensor values, optionally mirrored."""
    grid = np.full((rows, cols), np.nan)
    for i, val in enumerate(values):
        r,c = idx_map[i]
        if mirror:  # flip horizontally
            c = cols-1-c
        grid[r,c] = val
    return grid

# --- Synthetic circular CoG data ---
n_frames = 200
angles = np.linspace(0, 2*np.pi, n_frames)
circle_x = 5*np.cos(angles)
circle_y = 5*np.sin(angles)

def generate_frame(cx, cy):
    vals = []
    for i in range(n_sensors):
        r,c = idx_map[i]
        d = np.hypot(c-2, r-cy)  # rough distance to blob center
        vals.append(np.exp(-(d**2)/10)*100)
    return np.array(vals)

left_data = [generate_frame(x, y) for x,y in zip(circle_x,circle_y)]
right_data = [generate_frame(x, y) for x,y in zip(circle_x,circle_y)]
left_data = np.array(left_data)
right_data = np.array(right_data)

# --- Compute CoG ---
def compute_cog(lvals, rvals):
    lpos = np.array([idx_map[i] for i in range(n_sensors)])
    rpos = np.array([idx_map[i] for i in range(n_sensors)])
    # mirror left foot this time
    lpos[:,1] = cols-1-lpos[:,1]
    # shift apart visually
    lpos[:,1] -= 5
    rpos[:,1] += 5
    all_vals = np.concatenate([lvals, rvals])
    all_pos = np.vstack([lpos, rpos])
    total = all_vals.sum()
    return (all_pos*all_vals[:,None]).sum(0)/total if total>0 else np.zeros(2)

cogs = np.array([compute_cog(l,r) for l,r in zip(left_data,right_data)])

# --- Plot soles + character ---
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,4))

# Now: left foot is mirrored, right foot is not
im_left = ax1.imshow(frame_to_grid(left_data[0], mirror=True),
                     cmap="hot", vmin=0, vmax=100)
ax1.set_title("Left Foot")
ax1.axis("off")

im_right = ax2.imshow(frame_to_grid(right_data[0], mirror=False),
                      cmap="hot", vmin=0, vmax=100)
ax2.set_title("Right Foot")
ax2.axis("off")

char, = ax3.plot([], [], "ro", markersize=12)
ax3.set_xlim(-15,15)
ax3.set_ylim(-5,15)
ax3.set_aspect("equal")
ax3.set_title("Character")

def update(frame):
    im_left.set_data(frame_to_grid(left_data[frame], mirror=True))
    im_right.set_data(frame_to_grid(right_data[frame], mirror=False))
    char.set_data([cogs[frame,1]], [rows-cogs[frame,0]])  # flip y for nicer view
    return im_left, im_right, char

ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
plt.show()
