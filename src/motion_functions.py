import numpy as np


def circular_motion(t: float, radius: float, centerX: float, centerY: float, f: float) -> np.ndarray:
            """
            Return the (x, y) coordinates of a circle with radius centered at (centerX, centerY)
            as a function of time t and frequency f.
            """
            x = radius * np.cos(2 * np.pi * f * t/2) + centerX/2
            y = radius * np.sin(2 * np.pi * f * t/2) + centerY/2
            return np.array([x, y])

def clifford_attractor(t: float, a: float, b: float, c: float, d: float) -> np.ndarray:
        """Return the (x, y) coordinates of a point in the Clifford Attractor system
        as a function of time t and parameters a, b, c, d."""
        z = 0.3 + 0.15 * (np.sin(a * t) + c * np.cos(a * t))
        y = 0.2 + 0.1 * (np.sin(b * t) + d * np.cos(b * t))
        x = 0.4 * np.cos(z + y) 
        return np.array([x, y, z])