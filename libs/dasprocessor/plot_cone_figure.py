
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# User parameters (EDIT THESE)
# -----------------------------
p0 = np.array([0.0, 0.0, 0.0])      # apex position p0 (3,)
a  = np.array([0.0, 0.0, 1.0])      # axis direction vector a (3,) (doesn't need to be unit)
theta_deg = 25.0                    # half-angle theta in degrees

height = 3.0                        # how far to draw the cone along the axis
n_h = 60                            # samples along height
n_phi = 120                         # samples around the cone

# Visual tweaks
axis_len = 1.2 * height
arc_len  = 0.8 * height             # radius used for the theta arc visualization

# -----------------------------
# Helpers
# -----------------------------
def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Axis vector 'a' must be non-zero.")
    return v / n

def orthonormal_basis_from_axis(u: np.ndarray):
    """
    Given a unit vector u (axis), return two unit vectors v,w such that
    {v,w,u} is a right-handed orthonormal basis.
    """
    # Pick a vector not parallel to u
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.cross(u, tmp)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    return v, w

# -----------------------------
# Build cone surface
# -----------------------------
u = normalize(a)
theta = np.deg2rad(theta_deg)

v, w = orthonormal_basis_from_axis(u)

hs = np.linspace(0.0, height, n_h)
phis = np.linspace(0.0, 2*np.pi, n_phi)

H, PHI = np.meshgrid(hs, phis, indexing="ij")  # (n_h, n_phi)

R = H * np.tan(theta)  # radius grows linearly with height

# points: p = p0 + H*u + R*(cos(phi)*v + sin(phi)*w)
P = (p0[None, None, :]
     + H[..., None] * u[None, None, :]
     + R[..., None] * (np.cos(PHI)[..., None] * v[None, None, :]
                       + np.sin(PHI)[..., None] * w[None, None, :]))

X, Y, Z = P[..., 0], P[..., 1], P[..., 2]

# -----------------------------
# Build axis + theta arc
# -----------------------------
axis_end = p0 + axis_len * u

# Choose a generator direction (one specific cone ray) in the plane spanned by u and v
# At angle theta away from axis: dir = cos(theta)*u + sin(theta)*v
gen_dir = np.cos(theta) * u + np.sin(theta) * v
gen_end = p0 + axis_len * gen_dir

# Arc showing theta in the (u,v) plane
t = np.linspace(0.0, theta, 100)
arc_pts = p0 + arc_len * (np.cos(t)[:, None] * u[None, :] + np.sin(t)[:, None] * v[None, :])

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

# Cone surface
ax.plot_surface(X, Y, Z, alpha=0.35, linewidth=0, antialiased=True)

# Apex point
ax.scatter([p0[0]], [p0[1]], [p0[2]], s=60, marker="o", color = "red", label="Apex $\\mathbf{p}_0$")

# Axis line
ax.plot([p0[0], axis_end[0]], [p0[1], axis_end[1]], [p0[2], axis_end[2]],
        linewidth=2.5, color = "orange", label="Axis $\\mathbf{a}$")

# One cone generator ray (helps show the angle)
ax.plot([p0[0], gen_end[0]], [p0[1], gen_end[1]], [p0[2], gen_end[2]],
        linewidth=2.0, color = "blue")

# Theta arc
ax.plot(arc_pts[:, 0], arc_pts[:, 1], arc_pts[:, 2], linewidth=2.5, color = "green", label="Half-angle $\\theta$")

# Label theta near the arc end
theta_label_pos = arc_pts[len(arc_pts)//2]
ax.text(theta_label_pos[0], theta_label_pos[1], theta_label_pos[2], r"$\theta$", fontsize=14)

# Improve aspect / view
ax.set_title("3D Cone with Apex, Axis, and Half-angle")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Equal-ish scaling
all_pts = np.vstack([P.reshape(-1, 3), p0, axis_end, gen_end, arc_pts])
mins = all_pts.min(axis=0)
maxs = all_pts.max(axis=0)
ranges = maxs - mins
center = (maxs + mins) / 2
max_range = ranges.max()
ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)

ax.grid(False)
ax.set_axis_off()

ax.legend()
plt.tight_layout()
plt.show()
