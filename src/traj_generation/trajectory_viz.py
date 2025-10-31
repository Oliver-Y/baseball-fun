## Pull random data and visualize
from src import utils
import logging 
from datetime import date, timedelta
from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

G = 32.174  # ft/s^2 (use in z if you want gravity added)
@dataclass
class Trajectory9P:
    # initial position (ft), velocity (ft/s), and acceleration (ft/s^2)
    r0: np.ndarray          # shape (3,) -> [x0, y0, z0]
    v0: np.ndarray          # shape (3,) -> [vx0, vy0, vz0]
    a:  np.ndarray          # shape (3,) -> [ax, ay, az]  (non-gravity if using Statcast's az)

    include_gravity: bool = False  # True if you want total az = az - G

    def __call__(self, t):
        """
        Evaluate (x,y,z) at times t.
        t can be a scalar or a numpy array.
        Returns array of shape (len(t), 3) if t is array; otherwise (3,)
        """
        t = np.asarray(t)
        # choose vertical acceleration with or without gravity
        ax, ay, az = self.a
        if self.include_gravity:
            az = az - G  # Statcast az is "gravity-removed"; add gravity back downward

        # r(t) = r0 + v0*t + 0.5*a*t^2 (applies elementwise to each axis)
        # broadcast over t
        r = self.r0 + self.v0 * t[..., None] + 0.5 * np.array([ax, ay, az]) * (t[..., None] ** 2)
        return r  # shape: (N,3) or (3,)

    # convenience accessors
    def x(self, t): return self(t)[..., 0]
    def y(self, t): return self(t)[..., 1]
    def z(self, t): return self(t)[..., 2]

    def plate_intercept_time(self):
        """
        Solve y(t) = 0 for the first positive root.
        Uses constant-accel in y: y = y0 + vy0*t + 0.5*ay*t^2
        """
        x0, y0, z0 = self.r0
        vx0, vy0, vz0 = self.v0
        ax, ay, az = self.a
        # quadratic: 0.5*ay*t^2 + vy0*t + y0 = 0
        A = 0.5 * ay
        B = vy0
        C = y0
        disc = B*B - 4*A*C
        if disc < 0:
            return None
        roots = [(-B + np.sqrt(disc)) / (2*A) if A != 0 else None,
                 (-B - np.sqrt(disc)) / (2*A) if A != 0 else None]
        # pick smallest positive root
        cand = [r for r in roots if r is not None and r > 0]
        return min(cand) if cand else None

#vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'release_pos_x', 'release_pos_z',
def viz_trajectory(pitch_data: Optional[dict] = None):
    r0 = np.array([pitch_data['release_pos_x'], 60.5 - pitch_data['release_extension'], pitch_data['release_pos_z']])
    v0 = np.array([pitch_data['vx0'], pitch_data['vy0'], pitch_data['vz0']])
    a  = np.array([pitch_data['ax'], pitch_data['ay'], pitch_data['az']])
    traj = Trajectory9P(
        r0=r0,
        v0=v0,
        a=a,
        include_gravity=False  # Statcast az is gravity-removed
    )

    t = np.linspace(0.0, 0.6, 300)
    xyz = traj(t)                      
    x, y, z = xyz.T

    # === 3D Plot ===
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Keep axes proportional
    ax.set_box_aspect([1, 3, 0.5])  # x : y : z scaling
    # roughly: y-axis is longest (mound→plate), z is short (height)

    # Or, more physically: use data ranges
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(55, -5)     # invert so plate (0) is front
    ax.set_zlim(0, 7)

    # Optional: make ground plane clearer
    ax.plot([-2, 2], [0, 0], [0, 0], 'k--', alpha=0.3)

    ax.plot(x, y, z, lw=2)
    ax.scatter(*r0, color='green', s=50, label='Release')
    t_hit = traj.plate_intercept_time()
    x_pred, y_pred, z_pred = traj(t_hit) if t_hit is not None else (np.nan, np.nan, np.nan)
    ax.scatter(x_pred, y_pred, z_pred, color='red', s=60, label='Pred @ plate')

    # Add small time labels along the curve every 0.1 s
    for i in range(0, len(t), 50):
        ax.text(x[i], y[i], z[i], f"{t[i]:.2f}s", fontsize=8, color="black")

    # Aesthetics
    ax.set_xlabel('X (horizontal break, ft)')
    ax.set_ylabel('Y (toward plate, ft)')
    ax.set_zlabel('Z (height, ft)')

    # --- Statcast truth at plate ---
    x_true = pitch_data.get('plate_x', np.nan)
    z_true = pitch_data.get('plate_z', np.nan)

    # truth plate point (if present)
    if not (np.isnan(x_true) or np.isnan(z_true)):
        ax.scatter(x_true, 0.0, z_true, color='cyan', s=60, edgecolors='k', label='Statcast plate')

        # connect predicted → truth
        ax.plot([x_pred, x_true], [0.0, 0.0], [z_pred, z_true], linestyle='--', color='gray', lw=1)

        dx = x_true - x_pred
        dz = z_true - z_pred
        ax.text(x_true, 0.0, z_true, f"Δx={dx:.3f} ft\nΔz={dz:.3f} ft", fontsize=8)

    # Safely pull values (use .get to handle missing keys)
    name = pitch_data.get("player_name", "Unknown Pitcher")
    date = pitch_data.get("game_date", "Unknown Date")
    pitch_type = pitch_data.get("pitch_type", "Unknown Pitch")

    # Build a multi-line title string
    title_str = (
        f"{name} – {date}\n"
        f"Pitch Type: {pitch_type}\n"
        "3D Baseball Pitch Trajectory (Statcast 9P model)"
    )

    # Optional: draw a strike-zone box using sz_bot/sz_top (if available)
    sz_bot = pitch_data.get('sz_bot', None)
    sz_top = pitch_data.get('sz_top', None)
    if sz_bot is not None and sz_top is not None:
        # half plate width ≈ 0.7083 ft
        hw = 0.7083
        # 4 bottom corners then close the loop
        zx = np.array([-hw,  hw,  hw, -hw, -hw])
        zz = np.array([sz_bot, sz_bot, sz_top, sz_top, sz_bot])
        yy = np.zeros_like(zx)
        ax.plot(zx, yy, zz, color='k', alpha=0.25, label='Strike zone')

    ax.set_title(title_str, fontsize=10, pad=12)
    ax.legend()
    ax.view_init(elev=20, azim=-60)  # nice viewing angle

    plt.tight_layout()
    plt.show()

    #Answers: 'pfx_x', 'pfx_z', 'plate_x', 'plate_z'



#TODO have an abstract data class that parses data into the relevant and right format

# Get a module-level logger
logger = logging.getLogger(__name__)

#print the columns
test_date, stat = utils.pull_single_random_pitch_data()
logger.info("Columns: %s", stat.columns.to_list())
logger.info(f"Date: {test_date}, Stat: {stat}")
#Take the A-z, a-y, a-x and plot it into the 9P form 
pitch_sample = stat.iloc[0]  # take first row as sample
viz_trajectory(pitch_sample.to_dict())


