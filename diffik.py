"""Differential inverse kinematics for the UR5e.

Move the mocap body (red square) with your mouse (left click to rotate, right click to
translate) and watch the UR5e end-effector track its position and orientation.
"""

import mujoco
import numpy as np
import mujoco.viewer
import time


def jacobian(
    model: mujoco.MjModel, data: mujoco.MjData, site_id: int, dof_ids: np.ndarray
) -> np.ndarray:
    jac = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    return jac[:, dof_ids]


def spatial_velocity(
    pos: np.ndarray,
    pos_des: np.ndarray,
    quat: np.ndarray,
    quat_des: np.ndarray,
    dt: float,
) -> np.ndarray:
    # Angular velocity.
    quat_conj = np.zeros(4)
    mujoco.mju_negQuat(quat_conj, quat)
    error_quat = np.zeros(4)
    mujoco.mju_mulQuat(error_quat, quat_des, quat_conj)
    dw = np.zeros(3)
    mujoco.mju_quat2Vel(dw, error_quat, 1.0)

    # Linear velocity.
    dx = pos_des - pos

    return np.concatenate([dx, dw], axis=0) / dt


def diff_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    dof_ids: np.ndarray,
    pos: np.ndarray,
    ori: np.ndarray,
    control_dt: float,
    damping: float = 0.0,
) -> np.ndarray:
    # We only get access to xmat for sites, so we need to convert it to a quaternion.
    # Note: mju_mat2Quat returns a unit quaternion so no need to normalize.
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site(site_id).xmat)

    twist = spatial_velocity(
        pos=data.site(site_id).xpos,
        pos_des=pos,
        quat=quat,
        quat_des=ori,
        dt=control_dt,
    )

    jac = jacobian(model, data, site_id, dof_ids)

    # Damped least-squares.
    if damping > 0.0:
        # v = (J^T * J + lambda * I)^+ * J^T * V.
        return np.linalg.solve(
            jac.T @ jac + np.eye(jac.shape[1]) * damping,
            jac.T @ twist,
        )
    # Pseudoinverse: v = J^+ * V.
    return np.linalg.lstsq(jac, twist, rcond=None)[0]


def main() -> None:
    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene.xml")
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    # Control parameters.
    control_dt = 5 * dt  # Control timestep (seconds).
    damping = 1e-4  # Damping term for the pseudoinverse (unitless).

    # Compute the number of simulation steps needed per control step.
    n_steps = int(round(control_dt / dt))

    # End-effector site we wish to control.
    site_name = "attachment_site"
    site_id = model.site(site_name).id

    # Joint names we wish to control.
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])

    # Joint limits.
    jnt_limits = model.jnt_range.copy()

    # Actuator names we wish to control.
    actuator_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
    actuator_ids = np.array([model.actuator(name).id for name in actuator_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id

    # Mocap body we will control with our mouse.
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    # Reset the simulation to the initial joint configuration.
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Compute joint velocities needed to realize the desired spatial velocity.
            dq = diff_ik(
                model=model,
                data=data,
                site_id=site_id,
                dof_ids=dof_ids,
                control_dt=control_dt,
                damping=damping,
                pos=data.mocap_pos[mocap_id],
                ori=data.mocap_quat[mocap_id],
            )

            # The UR5 uses position actuators, so we need to integrate the joint
            # velocities to feed it desired joint positions.
            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, control_dt)
            np.clip(q, *jnt_limits.T, out=q)

            data.ctrl[actuator_ids] = q
            mujoco.mj_step(model, data, n_steps)

            viewer.sync()
            time_until_next_step = control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
