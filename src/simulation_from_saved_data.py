import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse
from motion_functions import circular_motion, clifford_attractor
import pandas as pd
import json

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

    def save_simulation(self, simulation_data):
        """
        Extract all the simulation data from mjdata object and save them as a dictionary with the name of each simulation variable as its name. 
        The data is then appended to the siumlation_data object and saved as time series

        Parameters:
        simulation_data (dict): a dictionary containing the simulation data saved as time series

        Returns:
        dict: The updated simulation data
        """
        # Collect all MjData attributes at this timestep
        timestep_data = self.collect_mjdata_attributes()
        
        # Add the current simulation time
        timestep_data['time'] = self.data.time
        
        # Append the collected data to the list
        simulation_data.append(timestep_data)
        return simulation_data
    
    def collect_mjdata_attributes(self):
        """
        Dynamically extract all the attribues of the simulation data from mjdata to a dictionary.
        """
        data_keys = {'main_keys':['ctrl', 'qpos', 'time'], 
                     'additional_keys':['actuator_', 'cam', 'q']}
        attributes = {}
        for attr_name in dir(self.data):
            if (any(key == attr_name for key in data_keys["main_keys"]) 
                or any(attr_name.startswith(key) for key in data_keys['additional_keys'])):
                attr_value = getattr(self.data, attr_name)
                try:
                    if isinstance(attr_value, np.ndarray):
                        attributes[attr_name] = attr_value.tolist()  # Convert numpy arrays to lists
                    else:
                        attributes[attr_name] = attr_value
                except Exception as e:
                    print(f"Could not process attribute {attr_name}: {e}")
        return attributes
    
    def runSimulation(self, save_simulation_data=False):
        """
        Run the simulation and visualize it locally.
        """
        self.setupRobotConfigs()
        if save_simulation_data:
            simulation_data = []
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

            df = pd.read_csv('simulation_data_ur5e.csv')
            for ctrl in df['ctrl']:
                step_start = time.time()
                self.data.ctrl = json.loads(ctrl)
                mujoco.mj_step(self.model, self.data)

                viewer.sync() # used for local visualisation
                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MuJoCo simulation for different robots.")
    parser.add_argument("--robot", type=str, choices=["kuka", "franka", "ur5e"], required=True, help="Type of robot to simulate")
    parser.add_argument('-s', action='store_true', help='Save simulation data')
    args = parser.parse_args()

    simulation = MuJocoSimulation(args.robot)
    simulation.runSimulation(args.s)
