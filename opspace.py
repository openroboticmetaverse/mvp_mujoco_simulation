"""Operational space control.

Prerequisites:

    pip install mujoco dm_control dm_robotics-transformations

Usage:

    mjpython opspace.py (macOS)
    python opspace.py (Linux)
"""

import mujoco
import numpy as np
import mujoco.viewer
import time
from dm_control import mjcf
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
from dm_robotics.transformations import transformations as tr

# Type annotations.
MjcfElement = mjcf.element._ElementImpl

_HERE = Path(__file__).parent

# Constants.
XML_PATH = _HERE / "kuka_iiwa_14" / "scene.xml"
HOME_QPOS = np.asarray([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0])


def pd_control(
    x: np.ndarray,
    x_des: np.ndarray,
    dx: np.ndarray,
    kp_kv: np.ndarray,
    ddx_max: float = 0.0,
) -> np.ndarray:
    # Compute error.
    x_err = x - x_des
    dx_err = dx

    # Apply gains.
    x_err *= -kp_kv[:, 0]
    dx_err *= -kp_kv[:, 1]

    # Limit maximum error.
    if ddx_max > 0.0:
        x_err_sq_norm = np.sum(x_err**2)
        ddx_max_sq = ddx_max**2
        if x_err_sq_norm > ddx_max_sq:
            x_err *= ddx_max / np.sqrt(x_err_sq_norm)

    return x_err + dx_err


def _orientation_error(
    quat: np.ndarray,
    quat_des: np.ndarray,
) -> np.ndarray:
    quat_err = tr.quat_mul(quat, tr.quat_conj(quat_des))
    quat_err /= np.linalg.norm(quat_err)
    axis_angle = tr.quat_to_axisangle(quat_err)
    if quat_err[0] < 0.0:
        angle = np.linalg.norm(axis_angle) - 2 * np.pi
    else:
        angle = np.linalg.norm(axis_angle)
    return axis_angle * angle


def pd_control_orientation(
    quat: np.ndarray,
    quat_des: np.ndarray,
    w: np.ndarray,
    kp_kv: np.ndarray,
    dw_max: float = 0.0,
) -> np.ndarray:
    # Compute error.
    # ori_err = tr.quat_to_axisangle(tr.quat_diff_active(quat, quat_des))
    ori_err = _orientation_error(quat, quat_des)
    w_err = w

    # Apply gains.
    ori_err *= -kp_kv[:, 0]
    w_err *= -kp_kv[:, 1]

    # Limit maximum error.
    if dw_max > 0.0:
        ori_err_sq_norm = np.sum(ori_err**2)
        dw_max_sq = dw_max**2
        if ori_err_sq_norm > dw_max_sq:
            ori_err *= dw_max / np.sqrt(ori_err_sq_norm)

    return ori_err + w_err


def opspace(
    physics: mjcf.Physics,
    site: MjcfElement,
    dof_ids: np.ndarray,
    joints: Sequence[MjcfElement],
    pos: Optional[np.ndarray] = None,
    ori: Optional[np.ndarray] = None,
    joint: Optional[np.ndarray] = None,
    pos_gains: Union[Tuple[float, float, float], np.ndarray] = (200.0, 200.0, 200.0),
    ori_gains: Union[Tuple[float, float, float], np.ndarray] = (200.0, 200.0, 200.0),
    damping_ratio: float = 1.0,
    nullspace_stiffness: float = 0.5,
    max_pos_acceleration: Optional[float] = None,
    max_ori_acceleration: Optional[float] = None,
    gravity_comp: bool = True,
) -> np.ndarray:
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
    if joint is None:
        q_des = physics.bind(joints).qpos.copy()
    else:
        q_des = np.asarray(joint)

    kp = np.asarray(pos_gains)
    kd = damping_ratio * 2 * np.sqrt(kp)
    kp_kv_pos = np.stack([kp, kd], axis=-1)

    kp = np.asarray(ori_gains)
    kd = damping_ratio * 2 * np.sqrt(kp)
    kp_kv_ori = np.stack([kp, kd], axis=-1)

    kp_joint = np.full((len(dof_ids),), nullspace_stiffness)
    kd_joint = damping_ratio * 2 * np.sqrt(kp_joint)
    kp_kv_joint = np.stack([kp_joint, kd_joint], axis=-1)

    ddx_max = max_pos_acceleration if max_pos_acceleration is not None else 0.0
    dw_max = max_ori_acceleration if max_ori_acceleration is not None else 0.0

    # Get current state.
    q = physics.bind(joints).qpos.copy()
    dq = physics.bind(joints).qvel.copy()

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

    # Compute position PD control.
    x = physics.bind(site).xpos.copy()
    dx = J_v @ dq
    ddx = pd_control(
        x=x,
        x_des=x_des,
        dx=dx,
        kp_kv=kp_kv_pos,
        ddx_max=ddx_max,
    )

    # Compute orientation PD control.
    quat = tr.mat_to_quat(physics.bind(site).xmat.copy().reshape((3, 3)))
    if quat @ quat_des < 0.0:
        quat *= -1.0
    w = J_w @ dq
    dw = pd_control_orientation(
        quat=quat,
        quat_des=quat_des,
        w=w,
        kp_kv=kp_kv_ori,
        dw_max=dw_max,
    )

    # Compute inertia matrix in joint space.
    M = np.zeros((len(dof_ids), len(dof_ids)), dtype=np.float64)
    mujoco.mj_fullM(physics.model.ptr, M, physics.data.qM)
    M = M[dof_ids, :][:, dof_ids]

    # Compute inertia matrix in task space.
    M_inv = np.linalg.inv(M)
    Mx_inv = J @ M_inv @ J.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

    # Compute generalized forces.
    ddx_dw = np.concatenate([ddx, dw], axis=0)
    tau = J.T @ Mx @ ddx_dw

    # Add joint task in nullspace.
    ddq = pd_control(
        x=q,
        x_des=q_des,
        dx=dq,
        kp_kv=kp_kv_joint,
        ddx_max=0.0,
    )
    Jnull = M_inv @ J.T @ Mx
    tau += (np.eye(len(q)) - J.T @ Jnull.T) @ ddq

    # We're technically doing more than just gravity compensation here. We're also
    # compensating for Coriolis/centrifugal forces. This is because MuJoCo lumps
    # together all the terms into qfrc_bias and there's no easy way to get the terms
    # individually.
    if gravity_comp:
        tau += physics.data.qfrc_bias.copy()[dof_ids]

    return tau


def main() -> None:
    mjcf_root = mjcf.from_path(XML_PATH.as_posix())

    physics = mjcf.Physics.from_mjcf_model(mjcf_root)
    physics.legacy_step = False

    jnts = mjcf_root.find_all("joint")
    dof_ids = np.array(physics.bind(jnts).dofadr)
    eef_site = mjcf_root.find("site", "attachment_site")

    with mujoco.viewer.launch_passive(physics.model.ptr, physics.data.ptr) as viewer:
        physics.reset(0)
        while viewer.is_running():
            step_start = time.time()

            pos_des = physics.data.mocap_pos[0].copy()
            quat_des = physics.data.mocap_quat[0].copy()

            # Compute spatial velocity.
            tau = opspace(
                physics=physics,
                site=eef_site,
                dof_ids=dof_ids,
                joints=jnts,
                pos=pos_des,
                ori=quat_des,
                joint=HOME_QPOS,
                gravity_comp=True,
            )

            # Set the control and step the simulation.
            physics.data.ctrl = tau @ np.linalg.pinv(physics.data.actuator_moment)
            physics.step()

            viewer.sync()
            time_until_next_step = physics.timestep() - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
