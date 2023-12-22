import mujoco
import numpy as np
import time


def main() -> None:
    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene.xml")
    data = mujoco.MjData(model)

    # ======================================================================== #
    # Hyperparameters for the controller.
    # ======================================================================== #
    dt = 0.002  # Simulation timestep (seconds).
    control_dt = 0.01  # Control timestep (seconds).

    # Amount of time the joint velocities are integrated over to obtain the joint
    # positions. This should be set to a value that is large enough to allow the
    # joints to move to the target pose, but not so large that the joints overshoot.
    integration_dt = 1.0  # (seconds).

    damping = 1e-5  # Damping term for the pseudoinverse (unitless).

    Kp = np.asarray([2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0])
    Kd = np.asarray([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    # ======================================================================== #

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

    # Joint limits.
    jnt_limits = model.jnt_range

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
    twist = np.zeros(6)
    site_quat = np.zeros(4)
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
            twist[:3] = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_subQuat(twist[3:], site_quat, data.mocap_quat[mocap_id])
            mujoco.mju_rotVecQuat(twist[3:], twist[3:], site_quat)
            twist[3:] *= -1.0

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Solve J * v = V with damped least squares to obtain joint velocities.
            if damping > 0.0:
                dq = np.linalg.solve(jac.T @ jac + diag, jac.T @ twist)
            else:
                dq = np.linalg.lstsq(jac, twist, rcond=None)[0]

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
