import numpy as np


def circular_motion(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
            """
            Return the (x, y) coordinates of a circle with radius r centered at (h, k)
            as a function of time t and frequency f.
            """
            x = r * np.cos(2 * np.pi * f * t) + h
            y = r * np.sin(2 * np.pi * f * t) + k
            return np.array([x, y])