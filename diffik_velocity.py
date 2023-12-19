import mujoco
import numpy as np
import time


def main() -> None:
    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene_velocity.xml")
    data = mujoco.MjData(model)

    # Control parameters.
    dt = 0.002  # Simulation timestep (seconds).
    control_dt = 5 * dt  # Control timestep (seconds).
    damping = 1e-5  # Damping term for the pseudoinverse (unitless).
    Kv = np.asarray([50.0] * model.nu)  # Velocity gains (rad/s).

    # Set PD gains.
    model.actuator_gainprm[:, 0] = Kv
    model.actuator_biasprm[:, 2] = -Kv

    # Enable gravity compensation.
    model.body_gravcomp[:] = 1.0

    # Compute the number of simulation steps needed per control step.
    model.opt.timestep = dt
    n_steps = int(round(control_dt / dt))

    # End-effector site we wish to control.
    site_name = "attachment_site"
    site_id = model.site(site_name).id

    # Joint names we wish to control.
    joint_names = [f"joint{i}" for i in range(1, 8)]
    actuator_names = [f"actuator{i}" for i in range(1, 8)]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in actuator_names])

    # Limits.
    vel_limits = model.actuator_ctrlrange  # rad/s.

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
            twist = np.hstack([dw, dx]) / 0.05

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac[3:], jac[:3], site_id)

            # Solve J * v = V with damped least squares to obtain joint velocities.
            if damping > 0.0:
                dq = np.linalg.solve(jac.T @ jac + diag, jac.T @ twist)
            else:
                dq = np.linalg.lstsq(jac, twist, rcond=None)[0]

            # Clip the joint velocities to the velocity limits.
            np.clip(dq, *vel_limits.T, out=dq)
            ctrl = dq[dof_ids]

            # Only set the control signal every n_steps.
            data.ctrl[actuator_ids] = ctrl
            mujoco.mj_step(model, data, n_steps)

            viewer.sync()
            time_until_next_step = control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
