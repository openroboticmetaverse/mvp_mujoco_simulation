import mujoco
import numpy as np
import cvxpy as cp
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
    diag = damping * np.eye(model.nv)
    v = cp.Variable(model.nv)
    G1 = np.vstack([-np.eye(model.nv), np.eye(model.nv)])
    h1 = np.hstack([vel_limits, vel_limits])
    G2 = np.vstack(
        [-integration_dt * np.eye(model.nv), integration_dt * np.eye(model.nv)]
    )

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
            # min_v 1/2 * v^T * P * v + q^T * v
            # s.t. G v <= h
            # P = J^T * J + lambda * I
            # q = -J^T * twist
            # (see diffik_qp.py for derivation of P and q)
            # Constraints:
            # Rewrite v_min <= v <= v_max as:
            # v_min - v <= 0 --> -v <= -v_min
            # -v[0] <= -v_min[0]
            # -v[1] <= -v_min[1]
            # ...
            # -v[nv-1] <= -v_min[nv-1]
            # Same for v - v_max <= 0 --> v <= v_max
            # Thus G1 = [[-I], [I]]
            # and h1 = [[-v_min], [v_max]] (2nv, 1)
            # Rewrite q_lower <= q + dt * v <= q_upper as:
            # q_lower - q - dt * v <= 0 --> - dt * v <= q - q_lower
            # -dt * v[0] <= -q_lower[0] + q[0]
            # -dt * v[1] <= -q_lower[1] + q[1]
            # ...
            # -dt * v[nv-1] <= -q_lower[nv-1] + q[nv-1]
            # Same for q_upper - q - dt * v <= 0 --> dt * v <= q_upper - q
            # Thus G2 = [[-dt * I], [dt * I]]
            h2 = np.hstack(
                [
                    data.qpos[dof_ids] - jnt_limits[:, 0],
                    jnt_limits[:, 1] - data.qpos[dof_ids],
                ]
            )
            G = np.vstack([G1, G2])
            h = np.hstack([h1, h2])
            P = jac.T @ jac + diag
            q = -jac.T @ twist
            prob = cp.Problem(cp.Minimize(cp.quad_form(v, P) + q.T @ v), [G @ v <= h])
            prob.solve(solver=cp.CVXOPT)
            dq = v.value

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
