import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# ---- Disable Matplotlib conflicting keys so 's' won't save ----
plt.rcParams['keymap.save'] = ''
plt.rcParams['keymap.fullscreen'] = ''
plt.rcParams['keymap.home'] = ''
plt.rcParams['keymap.back'] = ''
plt.rcParams['keymap.forward'] = ''

# ---- Sensor layout (row-major mask) ----
def sole_mask():
    mask = np.full((13, 4), np.nan)
    idx = 0
    for c in range(3):
        mask[0, c] = idx; idx += 1
    for r in range(1, 6):
        for c in range(4):
            mask[r, c] = idx; idx += 1
    for r in range(6, 9):
        for c in range(1, 3):
            mask[r, c] = idx; idx += 1
    for r in range(9, 12):
        for c in range(3):
            mask[r, c] = idx; idx += 1
    for c in range(2):
        mask[12, c] = idx; idx += 1
    return mask

mask = sole_mask()
ROWS, COLS = mask.shape
rc = np.array([(r, c) for r in range(ROWS) for c in range(COLS) if not np.isnan(mask[r, c])])
order = np.lexsort((rc[:, 1], rc[:, 0]))
rc = rc[order]

# ---- Physical layout (human-like) ----
hspace = 0.5
vspace = 0.5
x_local = (rc[:, 1] - (COLS - 1) / 2.0) * hspace
y_local = ((ROWS - 1) / 2.0 - rc[:, 0]) * vspace

foot_sep = 4.0
left_offset = -foot_sep / 2.0
right_offset = +foot_sep / 2.0

# No artificial mirroring — left foot uses x_local + left_offset (looks like left foot by shape)
x_left_phys  = x_local + left_offset
y_left_phys  = y_local.copy()
x_right_phys = x_local + right_offset
y_right_phys = y_local.copy()

# ---- Simulation parameters ----
SIGMA = 1.4
AMP = 100.0
MOVE_STEP = 0.25
DOT_GAIN = 0.50     # base movement gain for red dot
LATERAL_BOOST = 2.0 # boost horizontal effect on red dot
DEADZONE = 0.01
OBSTACLE_SPEED = 0.12
SPAWN_PROB = 0.08
COLLIDE_RADIUS = 0.5

# ---- Helpers ----
def gaussian_blob(cx, cy, x_phys, y_phys):
    dx = x_phys - cx
    dy = y_phys - cy
    return AMP * np.exp(-(dx*dx + dy*dy) / (2 * SIGMA * SIGMA))

def compute_cog(left_vals, right_vals):
    x_world = np.concatenate([x_left_phys, x_right_phys])
    y_world = np.concatenate([y_left_phys, y_right_phys])
    vals = np.concatenate([left_vals, right_vals])
    tot = vals.sum()
    if tot <= 1e-9:
        return 0.0, 0.0
    cx = np.dot(x_world, vals) / tot
    cy = np.dot(y_world, vals) / tot
    return cx, cy

def clamp(v, lo, hi): return max(lo, min(hi, v))

# ---- Initial centers (in world coords) ----
left_cx, left_cy   = left_offset, 0.0
right_cx, right_cy = right_offset, 0.0

left_vals  = gaussian_blob(left_cx, left_cy, x_left_phys,  y_left_phys)
right_vals = gaussian_blob(right_cx, right_cy, x_right_phys, y_right_phys)
cog_x, cog_y = compute_cog(left_vals, right_vals)

# ---- Game state ----
dot_pos = np.array([0.0, 0.0])   # red dot starts at center (0,0)
obstacles = []
score = 0
game_over = False

# ---- Plot setup ----
fig, (axFeet, axGame) = plt.subplots(1, 2, figsize=(14, 6))

x_min = min(x_left_phys.min(), x_right_phys.min()) - 1.0
x_max = max(x_left_phys.max(), x_right_phys.max()) + 1.0
y_min = y_local.min() - 1.0
y_max = y_local.max() + 1.0

# Feet plot
axFeet.set_xlim(x_min, x_max)
axFeet.set_ylim(y_min, y_max)
axFeet.set_aspect('equal')
axFeet.set_title('Feet (blue=left, green=right). Red = CoG')
left_scatter  = axFeet.scatter(x_left_phys,  y_left_phys,  c=left_vals,  cmap='Blues',  vmin=0, vmax=AMP, s=120, marker='s')
right_scatter = axFeet.scatter(x_right_phys, y_right_phys, c=right_vals, cmap='Greens', vmin=0, vmax=AMP, s=120, marker='s')
cog_marker, = axFeet.plot([], [], 'ro', markersize=10)
axFeet.axhline(0, color='gray', ls='--', lw=0.8)
axFeet.axvline(0, color='gray', ls='--', lw=0.8)

# Game plot (same extents so dot can reach all)
axGame.set_xlim(x_min, x_max)
axGame.set_ylim(y_min, y_max)
axGame.set_aspect('equal')
axGame.set_title('Game (red dot) - Score: 0')
char_marker, = axGame.plot([], [], 'ro', markersize=10)
obstacles_scatter, = axGame.plot([], [], 'ks', markersize=8)

# ---- Key handling (continuous) ----
key_state = {'w': False, 'a': False, 's': False, 'd': False,
             'up': False, 'down': False, 'left': False, 'right': False}

def _normalize_key(ev):
    if ev.key is None:
        return None
    k = ev.key.lower()
    # matplotlib sometimes reports 'left', sometimes 'arrow left' etc.
    k = k.replace('arrow ', '')
    return k

def on_key_press(ev):
    k = _normalize_key(ev)
    if k in key_state:
        key_state[k] = True

def on_key_release(ev):
    k = _normalize_key(ev)
    if k in key_state:
        key_state[k] = False

fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('key_release_event', on_key_release)

# ---- Animation update ----
def update(frame):
    global left_cx, left_cy, right_cx, right_cy, left_vals, right_vals
    global cog_x, cog_y, dot_pos, obstacles, score, game_over

    if game_over:
        return left_scatter, right_scatter, cog_marker, char_marker, obstacles_scatter

    # --- Move foot centers according to keys (intuitive physical directions) ---
    # Left foot controls (WASD)
    if key_state['w']:
        left_cy += MOVE_STEP
    if key_state['s']:
        left_cy -= MOVE_STEP
    if key_state['a']:
        left_cx -= MOVE_STEP
    if key_state['d']:
        left_cx += MOVE_STEP

    # Right foot controls (arrows)
    if key_state['up']:
        right_cy += MOVE_STEP
    if key_state['down']:
        right_cy -= MOVE_STEP
    if key_state['left']:
        right_cx -= MOVE_STEP
    if key_state['right']:
        right_cx += MOVE_STEP

    # --- Clamp foot centers so they stay near their nominal offsets ---
    left_cx  = clamp(left_cx,  left_offset - 1.0, left_offset + 1.0)
    left_cy  = clamp(left_cy,  y_min + 0.5, y_max - 0.5)
    right_cx = clamp(right_cx, right_offset - 1.0, right_offset + 1.0)
    right_cy = clamp(right_cy, y_min + 0.5, y_max - 0.5)

    # --- Recompute sensor intensities (blobs) using centers ---
    left_vals  = gaussian_blob(left_cx, left_cy, x_left_phys,  y_left_phys)
    right_vals = gaussian_blob(right_cx, right_cy, x_right_phys, y_right_phys)
    left_scatter.set_array(left_vals)
    right_scatter.set_array(right_vals)

    # --- Compute CoG from sensors (physical coords, no mirroring math) ---
    cog_x, cog_y = compute_cog(left_vals, right_vals)
    cog_marker.set_data([cog_x], [cog_y])

    # --- Move red dot toward CoG (boost horizontal effect so lateral movement is noticeable) ---
    vec = np.array([cog_x, cog_y]) - dot_pos
    # apply lateral boost (scale X component)
    vec[0] *= LATERAL_BOOST
    dist = np.hypot(vec[0], vec[1])
    if dist > DEADZONE:
        vel = (DOT_GAIN * vec / dist) * min(dist, 5.0)  # normalized direction * speed scaled by dist (caps)
        dot_pos += vel

    dot_pos[0] = clamp(dot_pos[0], x_min, x_max)
    dot_pos[1] = clamp(dot_pos[1], y_min, y_max)

    # --- Obstacles: spawn at top, fall downwards ---
    if random.random() < SPAWN_PROB:
        spawn_x = random.uniform(x_min + 0.5, x_max - 0.5)
        spawn_y = y_max + 0.5
        obstacles.append([spawn_x, spawn_y])
    for o in obstacles:
        o[1] -= OBSTACLE_SPEED
    obstacles = [o for o in obstacles if o[1] > (y_min - 1.0)]

    # --- Collision detection ---
    collided = any(np.hypot(dot_pos[0] - ox, dot_pos[1] - oy) < COLLIDE_RADIUS for ox, oy in obstacles)
    if collided:
        game_over = True
        axGame.set_title(f"GAME OVER! Final Score: {score}", color='red')
    else:
        score += 1
        axGame.set_title(f"Score: {score}   CoG Δ = (x={cog_x:.2f}, y={cog_y:.2f})")

    # --- Update visuals ---
    char_marker.set_data([dot_pos[0]], [dot_pos[1]])
    if obstacles:
        oxs, oys = zip(*obstacles)
        obstacles_scatter.set_data(oxs, oys)
    else:
        obstacles_scatter.set_data([], [])

    return left_scatter, right_scatter, cog_marker, char_marker, obstacles_scatter

ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
plt.tight_layout()
plt.show()
