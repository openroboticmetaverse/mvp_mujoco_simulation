import mujoco
import mujoco.viewer
import numpy as np
import xml.etree.ElementTree as ET
import time
import argparse
from motion_functions import circular_motion, clifford_attractor
import pandas as pd
import json
import copy
from get_coordinates import *
class MuJocoSimulation:
    """
    Class for the Mujoco Simulation.
    """
    
    def __init__(self, robot_models, num_robots, coordinates):
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
        self.robot_path = []
        self.create_new_scene(robot_models, num_robots, coordinates)

    def find_duplicates(self, strings):
        seen = {}
        duplicates = []
        for string in strings:
            if string in seen:
                seen[string] += 1
                duplicates.append(f"{string}_{seen[string]}")
            else:
                seen[string] = 0
                duplicates.append(string)
        return duplicates
    
    def create_new_scene(self, robot_type, num_robots, coordinates):
        """
        Configure robot-specific settings.
        """
        if robot_type == "kuka":
            self.base_path = "../config/kuka_iiwa_14/"
            self.robot_path = self.base_path + "iiwa14.xml"
            self.scene_path = self.base_path + "scene.xml"
            joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
            self.joint_name_prefix = "kuka_"
            self.name_home_pose = "home"
        elif robot_type == "franka":
            self.base_path = "../config/franka_emika_panda/"
            self.robot_path = self.base_path + "panda_nohand.xml"
            self.scene_path = self.base_path + "scene.xml"
            joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
            self.joint_name_prefix = "panda_"
            self.name_home_pose = "home"
        elif robot_type == "ur5e":
            self.base_path = "../config/universal_robots_ur5e/"
            self.robot_path = self.base_path + "ur5e.xml"
            self.scene_path = self.base_path + "scene.xml"
            joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
            self.joint_name_prefix = "ur5e_"
            self.name_home_pose = "home"
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")
        
        #configure joint names
        self.joint_names = []
        for id in range(num_robots):
            for joint in joint_names:
                self.joint_names.append(joint+'_copy_'+str(id))
        
        # load scene XML file
        scene_tree = ET.parse(self.scene_path)
        scene_root = scene_tree.getroot()

        # load model XML file
        robot_tree = ET.parse(self.robot_path)
        robot_root = robot_tree.getroot()

        # Create a new XML structure for the combined model
        combined_root = ET.Element('mujoco')

        # List of top-level tags to be copied
        top_level_tags = [
            'compiler', 'option', 'size', 'visual', 'asset', 'worldbody', 'tendon',
            'actuator', 'sensor', 'keyframe', 'contact', 'equality', 'custom', 'default']
        
        # Copy each top-level tag from both models
        for tag in top_level_tags:
            scene_element = scene_root.find(tag)
            robot_element = robot_root.find(tag)
            if tag == 'worldbody':
                # create new worldbody subelement
                new_element = ET.SubElement(combined_root, tag)
                #copy scene body elements into combined scene
                if scene_element is not None:
                    self.copy_elements(scene_element, new_element)
                # copy model body elements into combined scene
                for i, pos in enumerate(coordinates):
                    # adjust body names and positions
                    robot_copy = self.create_model_copy(robot_root.find('.//worldbody/body'), i, tag, pos)
                    new_element.append(robot_copy)
            elif tag == 'actuator':
                # create new actuator subelement
                new_element = ET.SubElement(combined_root, tag)
                # copy scene actuator elements into combined scene
                if scene_element is not None:
                    self.copy_elements(scene_element, new_element)
                # copy model actuator elements into combined scene
                for i, pos in enumerate(coordinates):
                    # adjust names and corresponding joints
                    robot_copy = self.create_model_copy(robot_root.find('.//actuator'), i, tag)
                    self.copy_elements(robot_copy, new_element)
            elif tag == 'contact':
                if robot_root.find('.//contact') is None:
                    continue
                # create new actuator subelement
                new_element = ET.SubElement(combined_root, tag)
                # copy scene actuator elements into combined scene
                if scene_element is not None:
                    self.copy_elements(scene_element, new_element)
                # copy model actuator elements into combined scene
                for i, pos in enumerate(coordinates):
                    # adjust names and corresponding joints
                    robot_copy = self.create_model_copy(robot_root.find('.//contact'), i, tag)
                    self.copy_elements(robot_copy, new_element)
            else:
                # copy scene elements into combined scene
                if scene_element is not None:
                    combined_root.append(scene_element)
                # copy model elements into combined scene
                if robot_element is not None:
                    combined_root.append(robot_element)
        
        # adjust qpos and control settings
        key_element =  combined_root.find('.//keyframe/key')
        qpos = key_element.get('qpos')
        ctrl = key_element.get('ctrl')
        key_element.set('qpos', ' '.join([qpos for i in range(len(coordinates))]))
        key_element.set('ctrl', ' '.join([ctrl for i in range(len(coordinates))]))

        # Save the combined model to a new XML file
        self.combined_scene_path = self.base_path + "combined_model.xml"
        tree_combined = ET.ElementTree(combined_root)
        tree_combined.write(self.combined_scene_path)
    
    def setupRobotConfigs(self):
        """
        Setup robot configurations
        """
        print(f"Loading model from: {self.combined_scene_path}")
        self.model = mujoco.MjModel.from_xml_path(self.combined_scene_path)
        self.data = mujoco.MjData(self.model)

        self.model.body_gravcomp[:] = float(self.gravity_compensation)
        self.model.opt.timestep = self.dt

        # End-effector site we wish to control.
        site_name = "attachment_site_copy_0"
        self.site_id = self.model.site(site_name).id
        self.key_id = self.model.key(self.name_home_pose).id
        return
        

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
        attributes = {}
        for attr_name in dir(self.data):
            if not attr_name.startswith('_'):
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
            while viewer.is_running():
                step_start = time.time()
                continue
                
    
    def copy_elements(self, source_element, target_element, exclude_tags=None):
        """
        Copy the element of a source xml file into the element of a target xml file.

        Parameters:
        source_element: source element to be copied
        target_element: element in which source elements are copied
        exclude_tags: tags that do not need to be copied
        """
        if exclude_tags is None:
            exclude_tags = []
        if source_element is not None:
            for element in source_element:
                if element.tag not in exclude_tags:
                    target_element.append(element)
        return
    
    def create_model_copy(self, root, instance_id, tag, position=None):
        """
        Create a copy of an element of model and adjust the names of the elements accordingly.
        
        Parameters:
        root: source element to be copied
        instance_id: simulation_id of the model
        tag: element type (worldbody, actuator or contact)
        position: coordinates of the model
        """
        # Deep copy the root element
        robot_copy = copy.deepcopy(root)
        if tag == 'worldbody':
            # update position
            robot_copy.set('pos', ' '.join(map(str, position)))
            robot_copy.set('name', f"{robot_copy.get('name')}_copy_{instance_id}")
            # Update names
            for element in robot_copy.findall('.//body') + robot_copy.findall('.//joint') + robot_copy.findall('.//site'):
                if 'name' in element.attrib:
                    element.attrib['name'] = f"{element.attrib['name']}_copy_{instance_id}"
        elif tag == 'actuator':
            # update names
            for pos_element in robot_copy.findall('.//position'):
                pos_element.set('name', f"{pos_element.get('name')}_copy_{instance_id}")
                pos_element.set('joint', f"{pos_element.get('joint')}_copy_{instance_id}")

            for general in robot_copy.findall('.//general'):
                general.set('name', f"{general.get('name')}_copy_{instance_id}")
                general.set('joint', f"{general.get('joint')}_copy_{instance_id}")
        elif tag == 'contact':
            # update names
            for pos_element in robot_copy.findall('.//exclude'):
                pos_element.set('body1', f"{pos_element.get('body1')}_copy_{instance_id}")
                pos_element.set('body2', f"{pos_element.get('body2')}_copy_{instance_id}")
        else:
            raise ValueError(f"Invalid element: {tag}")
        return robot_copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MuJoCo simulation for different robots.")
    parser.add_argument("--robot", type=str, choices=["kuka", "franka", "ur5e"], required=True, help="Type of robot to simulate")
    parser.add_argument('-n', type=int, help='select number of robots')
    args = parser.parse_args()
    if args.n < 0:
        raise ValueError('Number of robots should be greater than 0')
    coordinates = get_coordinates(args.n)

    simulation = MuJocoSimulation(args.robot, args.n, coordinates)
    simulation.runSimulation(False)
