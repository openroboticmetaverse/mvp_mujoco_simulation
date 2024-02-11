import numpy as np


def circular_motion(t: float, radius: float, centerX: float, centerY: float, f: float) -> np.ndarray:
            """
            Return the (x, y) coordinates of a circle with radius centered at (centerX, centerY)
            as a function of time t and frequency f.
            """
            x = radius * np.cos(2 * np.pi * f * t) + centerX
            y = radius * np.sin(2 * np.pi * f * t) + centerY
            return np.array([x, y])