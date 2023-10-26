"""Differential inverse kinematics.

Move the mocap body (red square) position and orientation (right and left click
respectively) and watch the UR5e end-effector track its position and orientation.

Prerequisites:

    pip install mujoco dm_control dm_robotics-transformations

Usage:

    mjpython diffik.py (macOS)
    python diffik.py (Linux)
"""

import mujoco
import numpy as np
import mujoco.viewer
import time
from dm_control import mjcf
from pathlib import Path
from typing import Optional
from dm_robotics.transformations import transformations as tr

# Type annotations.
MjcfElement = mjcf.element._ElementImpl

_HERE = Path(__file__).parent

# Constants.
# XML_PATH = _HERE / "universal_robots_ur5e" / "scene.xml"
XML_PATH = _HERE / "kuka_iiwa_14" / "scene.xml"
DAMPING = 1e-4
INTEGRATION_TIME = 1.0


def diff_ik(
    physics: mjcf.Physics,
    site: MjcfElement,
    dof_ids: np.ndarray,
    pos: Optional[np.ndarray] = None,
    ori: Optional[np.ndarray] = None,
    damping: float = 0.0,
) -> np.ndarray:
    """Computes a joint velocity that will realize the desired end-effector pose.

    Args:
        physics: The physics object.
        site: The end-effector site of the robot. If using a Menagerie robot, this will
            usually be a site called `attachment_site`.
        dof_ids: The dof ids of the joints to control.
        pos: The desired position of the site in world frame. If not specified, uses
            the current position of the site.
        ori: The desired orientation of the site in world frame. If not specified, uses
            the current orientation of the site.
        damping: Regularization term for the damped least squares solver.
    """
    if pos is None:
        x_des = physics.bind(site).xpos.copy()
    else:
        x_des = np.asarray(pos)
    if ori is None:
        xmat = physics.bind(site).xmat.copy()
        quat_des = tr.mat_to_quat(xmat.reshape((3, 3)))
    else:
        ori = np.asarray(ori)
        if ori.shape == (3, 3):
            quat_des = tr.mat_to_quat(ori)
        else:
            quat_des = ori

    # Compute Jacobian of the eef site in world frame.
    J_v = np.zeros((3, len(dof_ids)), dtype=np.float64)
    J_w = np.zeros((3, len(dof_ids)), dtype=np.float64)
    mujoco.mj_jacSite(
        physics.model.ptr,
        physics.data.ptr,
        J_v,
        J_w,
        physics.bind(site).element_id,
    )
    J_v = J_v[:, dof_ids]
    J_w = J_w[:, dof_ids]
    J = np.concatenate([J_v, J_w], axis=0)

    # Translation error.
    dx = x_des - physics.bind(site).xpos.copy()

    # Orientation error.
    quat = tr.mat_to_quat(physics.bind(site).xmat.copy().reshape((3, 3)))
    err_quat = tr.quat_diff_active(quat, quat_des)
    dw = tr.quat_to_axisangle(err_quat)

    # Compute end-effector velocity using damped least squares.
    twist = np.concatenate([dx, dw], axis=0)
    if damping > 0.0:
        hess_approx = J.T @ J + np.eye(J.shape[1]) * damping
        jac_pinv = np.linalg.solve(hess_approx, J.T)
        return jac_pinv @ twist
    else:
        return np.linalg.lstsq(J, twist, rcond=None)[0]


def main() -> None:
    mjcf_root = mjcf.from_path(XML_PATH.as_posix())

    physics = mjcf.Physics.from_mjcf_model(mjcf_root)
    physics.legacy_step = False

    jnts = mjcf_root.find_all("joint")
    dof_ids = np.array(physics.bind(jnts).dofadr)
    eef_site = mjcf_root.find("site", "attachment_site")
    jnt_range = physics.bind(jnts).range.copy()

    with mujoco.viewer.launch_passive(
        physics.model.ptr,
        physics.data.ptr,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        physics.reset(0)
        while viewer.is_running():
            step_start = time.time()

            pos_des = physics.data.mocap_pos[0].copy()
            quat_des = physics.data.mocap_quat[0].copy()

            # Compute spatial velocity.
            dq = diff_ik(
                physics=physics,
                site=eef_site,
                dof_ids=dof_ids,
                pos=pos_des,
                ori=quat_des,
                damping=DAMPING,
            )

            # Integrate dq.
            q = physics.data.qpos.copy()
            mujoco.mj_integratePos(physics.model.ptr, q, dq, INTEGRATION_TIME)
            np.clip(q, jnt_range[:, 0], jnt_range[:, 1], out=q)

            # Set the control and step the simulation.
            physics.data.ctrl = q
            physics.step(nstep=2)

            viewer.sync()
            time_until_next_step = physics.timestep() - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
