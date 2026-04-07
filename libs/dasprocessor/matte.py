import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_phi = 200
n_theta = 300

phi = np.linspace(0, np.pi, n_phi)         # polar angle from +z axis
theta = np.linspace(0, 2*np.pi, n_theta)   # azimuth angle around z axis

Phi, Theta = np.meshgrid(phi, theta, indexing="ij")

# Radius as a function of phi
R = 6 - 3*np.cos(Phi)

# Spherical -> Cartesian (phi from z-axis)
X = R * np.sin(Phi) * np.cos(Theta)
Y = R * np.sin(Phi) * np.sin(Theta)
Z = R * np.cos(Phi)

# Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.9)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title(r"Surface: $r(\phi)=4-\cos(\phi)$, $\phi\in[0,\pi]$, $\theta\in[0,2\pi]$")

# Make axes have equal scale so it doesn't look squished
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
mid_x = (X.max() + X.min()) / 2
mid_y = (Y.max() + Y.min()) / 2
mid_z = (Z.max() + Z.min()) / 2
ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

plt.show()
