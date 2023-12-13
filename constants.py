import numpy as np

# UR5e constants.
_UR5E_VELOCITY_LIMITS_DEG = np.asarray((180, 180, 180, 180, 180, 180))
UR5E_VELOCITY_LIMITS = np.deg2rad(_UR5E_VELOCITY_LIMITS_DEG)  # rad/s.
UR5E_TORQUE_LIMITS = (150, 150, 150, 280, 28, 28)  # Nm.

# iiwa14 constants.
IIWA_VELOCITY_LIMITS = np.deg2rad((85, 85, 100, 75, 130, 135, 135))  # rad/s.
