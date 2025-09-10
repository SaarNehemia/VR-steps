import numpy as np
import matplotlib.pyplot as plt

"""
Playable soles + CoG demo
- Left and right feet have their natural human layout (left foot mirrored naturally)
- Both feet in global coordinate system
- Arrow keys control right foot CoG (up=toes, down=heel, left/right horizontal)
- WASD keys control left foot CoG (W=up, S=down, A=left, D=right)
- Gaussians and red dot move consistently in all directions
- Default Matplotlib key bindings that interfere (like 's' for save) are disabled
"""

# Disable conflicting key bindings
plt.rcParams['keymap.save'] = ''
plt.rcParams['keymap.fullscreen'] = ''
plt.rcParams['keymap.home'] = ''
plt.rcParams['keymap.back'] = ''
plt.rcParams['keymap.forward'] = ''

# ---------------------- Layout / Indexing ----------------------
def sole_mask():
    mask = np.full((13, 4), np.nan)
    idx = 0
    for c in range(3): mask[0, c] = idx; idx += 1
    for r in range(1, 6):
        for c in range(4): mask[r, c] = idx; idx += 1
    for r in range(6, 9):
        for c in range(1, 3): mask[r, c] = idx; idx += 1
    for r in range(9, 12):
        for c in range(3): mask[r, c] = idx; idx += 1
    for c in range(2): mask[12, c] = idx; idx += 1
    return mask

mask = sole_mask()
ROWS, COLS = mask.shape
N = int(np.nanmax(mask)) + 1

# Sensor positions
rc = np.array([(r,c) for r in range(ROWS) for c in range(COLS) if not np.isnan(mask[r,c])])
order = np.lexsort((rc[:,1], rc[:,0]))
rc = rc[order]
assert len(rc) == N

# Physical positions (global coordinates) - left foot naturally mirrored
x_left_phys  = (COLS-1 - rc[:,1]).astype(float)   # left foot X reversed naturally
y_left_phys  = rc[:,0].astype(float)
x_right_phys = rc[:,1].astype(float)             # right foot X normal
y_right_phys = rc[:,0].astype(float)

FOOT_GAP = 5.0
SIGMA = 1.4
AMP = 100.0
STEP = 0.4

# ---------------------- Helpers ----------------------
def frame_to_grid(values, left=False):
    grid = np.full((ROWS, COLS), np.nan)
    for i, (r,c) in enumerate(rc):
        cc = int(x_left_phys[i]) if left else int(x_right_phys[i])
        grid[int(r), cc] = values[i]
    return grid

def generate_frame(cx, cy, x_phys, y_phys):
    def blob(x, y):
        dx = x - cx
        dy = y - cy
        return AMP * np.exp(-(dx**2 + dy**2)/(2*SIGMA*SIGMA))
    vals = blob(x_phys, y_phys)
    return vals

def compute_cog(left_vals, right_vals):
    # Use true physical coordinates for CoG calculation (no FOOT_GAP shift)
    x_world = np.concatenate([x_left_phys, x_right_phys])
    y_world = np.concatenate([y_left_phys, y_right_phys])
    vals = np.concatenate([left_vals, right_vals])
    tot = vals.sum()
    if tot <= 1e-9: return 0.0, 0.0
    cx = np.dot(x_world, vals) / tot
    cy = np.dot(y_world, vals) / tot
    return cx, cy

# ---------------------- Initial blob ----------------------
left_cx, left_cy = (COLS-1)/2.0, (ROWS-1)/2.0
right_cx, right_cy = (COLS-1)/2.0, (ROWS-1)/2.0
left_vals = generate_frame(left_cx, left_cy, x_left_phys, y_left_phys)
right_vals = generate_frame(right_cx, right_cy, x_right_phys, y_right_phys)
cog = compute_cog(left_vals, right_vals)

# ---------------------- Plot ----------------------
fig, (axL, axR, axC) = plt.subplots(1, 3, figsize=(13,5))
im_left  = axL.imshow(frame_to_grid(left_vals, left=True),  cmap='hot', vmin=0, vmax=AMP)
axL.set_title('Left Foot');  axL.axis('off')
im_right = axR.imshow(frame_to_grid(right_vals, left=False), cmap='hot', vmin=0, vmax=AMP)
axR.set_title('Right Foot'); axR.axis('off')

char, = axC.plot([], [], 'ro', markersize=10)
axC.set_xlim(-FOOT_GAP-1, COLS-1+FOOT_GAP+1)
axC.set_ylim(-1, ROWS)
axC.set_aspect('equal'); axC.grid(True, linestyle='--', alpha=0.3)
axC.set_title('Character (CoG)')
char.set_data([cog[0]], [ROWS - cog[1]])

# ---------------------- Key controls ----------------------
def clamp(v, lo, hi): return max(lo, min(hi, v))

def on_key(event):
    global left_cx, left_cy, right_cx, right_cy, left_vals, right_vals, cog
    # Left foot WASD
    if event.key == 'w': left_cy -= STEP
    elif event.key == 's': left_cy += STEP
    elif event.key == 'a': left_cx -= STEP
    elif event.key == 'd': left_cx += STEP
    # Right foot arrows
    elif event.key == 'up': right_cy -= STEP
    elif event.key == 'down': right_cy += STEP
    elif event.key == 'left': right_cx -= STEP
    elif event.key == 'right': right_cx += STEP

    left_cx = clamp(left_cx, 0.0, COLS-1.0)
    left_cy = clamp(left_cy, 0.0, ROWS-1.0)
    right_cx = clamp(right_cx, 0.0, COLS-1.0)
    right_cy = clamp(right_cy, 0.0, ROWS-1.0)

    left_vals = generate_frame(left_cx, left_cy, x_left_phys, y_left_phys)
    right_vals = generate_frame(right_cx, right_cy, x_right_phys, y_right_phys)
    cog = compute_cog(left_vals, right_vals)

    im_left.set_data(frame_to_grid(left_vals, left=True))
    im_right.set_data(frame_to_grid(right_vals, left=False))
    char.set_data([cog[0]], [ROWS - cog[1]])
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key)
plt.tight_layout()
plt.show()
