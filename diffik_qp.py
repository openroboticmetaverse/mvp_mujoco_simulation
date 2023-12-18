import mujoco
import numpy as np
import time


def main() -> None:
    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene.xml")
    data = mujoco.MjData(model)

    # Control parameters.
    dt = 0.005  # Simulation timestep (seconds).
    control_dt = 5 * dt  # Control timestep (seconds).
    integration_dt = 1.0  # Integration timestep (seconds).
    damping = 1e-5  # Damping term for the pseudoinverse (unitless).
    Kp = np.asarray([4000.0, 4000.0, 4000.0, 1000.0, 1000.0, 1000.0])
    Kd = np.asarray([200.0, 200.0, 200.0, 50.0, 50.0, 50.0])

    # Set PD gains.
    model.actuator_gainprm[:, 0] = Kp
    model.actuator_biasprm[:, 1] = -Kp
    model.actuator_biasprm[:, 2] = -Kd

    # Compute the number of simulation steps needed per control step.
    model.opt.timestep = dt
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
    actuator_names = [name[:-5] + "position" for name in joint_names]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in actuator_names])

    # Limits.
    jnt_limits = model.jnt_range
    vel_limits = np.full((model.nq,), np.pi)  # 180 deg/s.

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id

    # Mocap body we will control with our mouse.
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    # Reset the simulation to the initial joint configuration.
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    dw = np.zeros(3)
    dq = np.zeros(model.nv)
    r = np.zeros((model.nv, model.nv + 7))
    index = np.zeros(model.nv, np.int32)
    diag = damping * np.eye(model.nv)

    def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
        """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
        as a function of time t and frequency f."""
        x = r * np.cos(2 * np.pi * f * t) + h
        y = r * np.sin(2 * np.pi * f * t) + k
        return np.array([x, y])

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        while viewer.is_running():
            step_start = time.time()

            data.mocap_pos[mocap_id, 0:2] = circle(data.time, 0.1, 0.5, 0.0, 0.5)

            # Spatial velocity (aka twist).
            dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(dw, error_quat, 1.0)
            twist = np.hstack([dw, dx]) / integration_dt

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac[3:], jac[:3], site_id)

            # Solve QP:
            # min_v ||J * v - V||_2^2
            # s.t. v_lower <= v <= v_upper
            #      q_lower <= q + dt * v <= q_upper
            # Rewrite as:
            # min_v 1/2 * v^T * H * v + g^T * v
            # s.t. v_lower <= v <= v_upper
            #      q_lower <= q + dt * v <= q_upper
            # where H = J^T * J + diag(damping)
            #       g = -J^T * V
            H = jac.T @ jac + diag
            g = -jac.T @ twist
            q_limits = (jnt_limits - data.qpos.reshape(-1, 1)) / integration_dt
            lower = np.maximum(-vel_limits, q_limits[:, 0])
            upper = np.minimum(vel_limits, q_limits[:, 1])
            mujoco.mju_boxQP(dq, r, index, H, g, lower, upper)

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()  # Note the copy here is important.
            mujoco.mj_integratePos(model, q, dq, integration_dt)
            np.clip(q, *jnt_limits.T, out=q)
            ctrl = q[dof_ids]

            # Set the control signal and step the simulation.
            data.ctrl[actuator_ids] = ctrl
            mujoco.mj_step(model, data, n_steps)

            viewer.sync()
            time_until_next_step = control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
