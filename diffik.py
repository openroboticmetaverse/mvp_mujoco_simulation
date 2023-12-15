import mujoco
import numpy as np
import time


def main() -> None:
    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene.xml")
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    # Control parameters.
    control_dt = 1 * dt  # Control timestep (seconds).
    integration_dt = 1.0  # Integration timestep (seconds).
    damping = 1e-5  # Damping term for the pseudoinverse (unitless).
    Kps = np.asarray([4000.0, 4000.0, 4000.0, 1000.0, 1000.0, 1000.0])
    Kds = np.asarray([400.0, 400.0, 400.0, 200.0, 200.0, 200.0])

    # Set PD gains.
    model.actuator_gainprm[:, 0] = Kps
    model.actuator_biasprm[:, 1] = -Kps
    model.actuator_biasprm[:, 2] = -Kds

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
    actuator_names = [name[:-6] for name in joint_names]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in actuator_names])

    # Joint limits.
    jnt_limits = model.jnt_range[dof_ids]

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id

    # Mocap body we will control with our mouse.
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    # Reset the simulation to the initial joint configuration.
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv), dtype=np.float64)
    site_quat = np.zeros(4, dtype=np.float64)
    site_quat_conj = np.zeros(4, dtype=np.float64)
    error_quat = np.zeros(4, dtype=np.float64)
    dw = np.zeros(3, dtype=np.float64)
    diag = damping * np.eye(model.nv)

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        while viewer.is_running():
            step_start = time.time()

            dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(dw, error_quat, 1.0)
            twist = np.concatenate([dx, dw], axis=0) / integration_dt

            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            if damping > 0.0:
                dq = np.linalg.solve(jac.T @ jac + diag, jac.T @ twist)
            else:
                dq = np.linalg.lstsq(jac, twist, rcond=None)[0]

            q = data.qpos.copy()  # Note the copy here is important.
            mujoco.mj_integratePos(model, q, dq, integration_dt)
            ctrl = q[dof_ids]
            np.clip(ctrl, *jnt_limits.T, out=ctrl)

            data.ctrl[actuator_ids] = ctrl
            mujoco.mj_step(model, data, n_steps)

            viewer.sync()
            time_until_next_step = control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
