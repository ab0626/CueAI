# physics/table_surface.py
import numpy as np

def get_curvature_force(x, y):
    # Gaussian warp in bottom-right corner
    center_x = 2.2
    center_y = 1.0
    radius = 0.5

    dx = x - center_x
    dy = y - center_y
    dist = np.hypot(dx, dy)

    if dist > radius:
        return (0, 0)

    strength = 0.005 * (1 - dist / radius)
    fx = -strength * dx
    fy = -strength * dy
    return (fx, fy)
