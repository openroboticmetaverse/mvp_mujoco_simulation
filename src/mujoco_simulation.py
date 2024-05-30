import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse
from motion_functions import circular_motion, clifford_attractor

class MuJocoSimulation:
    """
    Class for the Mujoco Simulation.
    """
    
    def __init__(self, robot_type):
        # Integration timestep in seconds. This corresponds to the amount of time the joint
        # velocities will be integrated for to obtain the desired joint positions. 
        # Source values: 6-DoF robot: 1.0, 7-DoF robot: 0.1
        self.integration_dt: float = 1.0

        # Damping term for the pseudoinverse. This is used to prevent joint velocities from
        # becoming too large when the Jacobian is close to singular.
        # Source values: 6-DoF robot: 1e-5, 7-DoF robot: 1e-4
        self.damping: float = 1e-5

        # Gains for the twist computation. These should be between 0 and 1. 0 means no
        # movement, 1 means move the end-effector to the target in one integration step.
        self.Kpos: float = 0.95
        self.Kori: float = 0.95

        # Nullspace P gain - used only for 7-DoF robots
        self.Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

        # Whether to enable gravity compensation.
        self.gravity_compensation: bool = True

        # Simulation timestep in seconds.
        self.dt: float = 0.01   #0.002

        # Maximum allowable joint velocity in rad/s.
        self.max_angvel = 0.785

        # Robot type configuration
        self.robot_type = robot_type
        self.configure_robot(robot_type)

    def configure_robot(self, robot_type):
        """
        Configure robot-specific settings.
        """
        if robot_type == "kuka":
            self.robot_path = "../config/kuka_iiwa_14/scene.xml"
            self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
            self.joint_name_prefix = "kuka_"
            self.name_home_pose = "home"
        elif robot_type == "franka":
            self.robot_path = "../config/franka_emika_panda/scene.xml"
            self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
            self.joint_name_prefix = "panda_"
            self.name_home_pose = "home"
        elif robot_type == "ur5e":
            self.robot_path = "../config/universal_robots_ur5e/scene.xml"
            self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
            self.joint_name_prefix = "ur5e_"
            self.name_home_pose = "home"
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")

    def setupRobotConfigs(self):
        """
        Setup robot configurations
        """
        full_path = self.robot_path
        print(f"Loading model from: {full_path}")
        self.model = mujoco.MjModel.from_xml_path(full_path)
        self.data = mujoco.MjData(self.model)

        self.model.body_gravcomp[:] = float(self.gravity_compensation)
        self.model.opt.timestep = self.dt

        # End-effector site we wish to control.
        site_name = "attachment_site"
        self.site_id = self.model.site(site_name).id

        self.dof_ids = np.array([self.model.joint(name).id for name in self.joint_names])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in self.joint_names])

        # Initial joint configuration saved as a keyframe in the XML file.
        self.key_id = self.model.key(self.name_home_pose).id
        self.q0 = self.model.key(self.name_home_pose).qpos

        # Mocap body we will control with our mouse.
        mocap_name = "target"
        self.mocap_id = self.model.body(mocap_name).mocapid[0]

        # Pre-allocate numpy arrays.
        self.jac = np.zeros((6, self.model.nv))
        self.diag = self.damping * np.eye(self.model.nv)
        self.eye = np.eye(self.model.nv)
        self.twist = np.zeros(6)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)

    def runSimulation(self):
        """
        Run the simulation and visualize it locally.
        """
        self.setupRobotConfigs()
        
        # Launch the viewer.
        simViewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        with simViewer as viewer:
            # Reset the simulation.
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
            
            # Reset the free camera.
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
            # Enable site frame visualization.
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            while viewer.is_running():
                step_start = time.time()

                # Get demo motion function
                a = 1.5
                b = -1.8
                c = 1.6
                d = 0.9
                self.data.mocap_pos[self.mocap_id, 0:3] = clifford_attractor(self.data.time, a, b, c, d)
                #self.data.mocap_pos[self.mocap_id, 0:3] = np.array([0.514, 0.55, 0.5])
                #self.data.mocap_pos[self.mocap_id, 0:2] = circular_motion(self.data.time, 0.1, 0.5, 0.0, 0.5)

                # Spatial velocity (aka twist).
                dx = self.data.mocap_pos[self.mocap_id] - self.data.site(self.site_id).xpos
                self.twist[:3] = self.Kpos * dx / self.integration_dt
                mujoco.mju_mat2Quat(self.site_quat, self.data.site(self.site_id).xmat)
                mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
                mujoco.mju_mulQuat(self.error_quat, self.data.mocap_quat[self.mocap_id], self.site_quat_conj)
                mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
                self.twist[3:] *= self.Kori / self.integration_dt

                # Jacobian
                mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)

                # Solve for joint velocities: J * dq = twist using damped least squares.
                dq = np.linalg.solve(self.jac.T @ self.jac + self.diag, self.jac.T @ self.twist)

                # Nullspace control biasing joint velocities towards the home configuration.
                if len(self.joint_names) == 7:
                    dq += (self.eye - np.linalg.pinv(self.jac) @ self.jac) @ (self.Kn * (self.q0 - self.data.qpos[self.dof_ids]))

                # Clamp maximum joint velocity.
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > self.max_angvel:
                    dq *= self.max_angvel / dq_abs_max

                # Integrate joint velocities to obtain joint positions - copying is important
                q = self.data.qpos.copy()

                # Adds a vector in the format of qvel (scaled by dt) to a vector in the format of qpos.
                mujoco.mj_integratePos(self.model, q, dq, self.integration_dt)
                np.clip(q, *self.model.jnt_range.T, out=q)

                # Set the control signal and step the simulation.
                self.data.ctrl[self.actuator_ids] = q[self.dof_ids]
                mujoco.mj_step(self.model, self.data)

                viewer.sync() # used for local visualisation
                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MuJoCo simulation for different robots.")
    parser.add_argument("--robot", type=str, choices=["kuka", "franka", "ur5e"], required=True, help="Type of robot to simulate")
    args = parser.parse_args()

    simulation = MuJocoSimulation(args.robot)
    simulation.runSimulation()
