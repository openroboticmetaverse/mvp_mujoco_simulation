import mujoco
import mujoco.viewer
import numpy as np
import xml.etree.ElementTree as ET
import time
import argparse
from motion_functions import circular_motion, clifford_attractor
#import pandas as pd
#import json
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
        elif robot_type == "anymal_b":
            self.base_path = "../config/anybotics_anymal_b/"
            self.robot_path = self.base_path + "anymal_b.xml"
            self.scene_path = self.base_path + "scene.xml"
            joint_names = []
            self.joint_name_prefix = "anymal_b_"
            self.name_home_pose = "home"
        elif robot_type == "anymal_c":
            self.base_path = "../config/anybotics_anymal_c/"
            self.robot_path = self.base_path + "anymal_c.xml"
            self.scene_path = self.base_path + "scene.xml"
            joint_names = []
            self.joint_name_prefix = "anymal_c_"
            self.name_home_pose = "home"
        elif robot_type == "cf2":
            self.base_path = "../config/bitcraze_crazyflie_2/"
            self.robot_path = self.base_path + "cf2.xml"
            self.scene_path = self.base_path + "scene.xml"
            joint_names = []
            self.joint_name_prefix = "cf2_"
            self.name_home_pose = "home"
        elif robot_type == "boston_d":
            self.base_path = "../config/boston_dynamics_spot/"
            self.robot_path = self.base_path + "spot.xml"
            self.scene_path = self.base_path + "scene.xml"
            joint_names = []
            self.joint_name_prefix = "bd_"
            self.name_home_pose = "home"
        elif robot_type == "fruitfly":
            self.base_path = "../config/flybody/"
            self.robot_path = self.base_path + "fruitfly.xml"
            self.scene_path = self.base_path + "scene.xml"
            joint_names = []
            self.joint_name_prefix = "fly_"
            self.name_home_pose = "home"
        elif robot_type == "aloha":
            self.base_path = "../config/aloha/"
            self.robot_path = self.base_path + "aloha.xml"
            self.scene_path = self.base_path + "scene.xml"
            joint_names = []
            self.joint_name_prefix = "aloha_"
            self.name_home_pose = "home"
        elif robot_type == "cassie":
            self.base_path = "../config/agility_cassie/"
            self.robot_path = self.base_path + "cassie.xml"
            self.scene_path = self.base_path + "scene.xml"
            joint_names = []
            self.joint_name_prefix = "aloha_"
            self.name_home_pose = "home"
        elif robot_type == "franka3":
            self.base_path = "../config/franka_fr3/"
            self.robot_path = self.base_path + "fr3.xml"
            self.scene_path = self.base_path + "scene.xml"
            joint_names = []
            self.joint_name_prefix = "franka3_"
            self.name_home_pose = "home"
        elif robot_type == "google_robot":
            self.base_path = "../config/google_robot/"
            self.robot_path = self.base_path + "robot.xml"
            self.scene_path = self.base_path + "scene.xml"
            joint_names = []
            self.joint_name_prefix = "franka3_"
            self.name_home_pose = "home"
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")
        
        if num_robots == 1:
            self.joint_names = joint_names
            self.combined_scene_path = self.scene_path
            return
        
        # configure joint names
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
            robot_element = robot_root.find(tag)
            if tag == 'worldbody':
                # create new worldbody subelement
                new_element = ET.SubElement(combined_root, tag)
                
                # copy model body elements into combined scene
                for i, pos in enumerate(coordinates):
                    # adjust body names and positions
                    for body in robot_root.findall('.//worldbody/body'):
                        robot_copy = self.create_model_copy(body, i, tag, pos)
                        new_element.append(robot_copy)
            elif tag in ['actuator', 'contact', 'tendon', 'sensor', 'equality']:
                if robot_root.find('.//'+tag) is None:
                    continue
                # create new actuator subelement
                new_element = ET.SubElement(combined_root, tag)
            
                # copy model actuator elements into combined scene
                for i, pos in enumerate(coordinates):
                    # adjust names and corresponding joints
                    robot_copy = self.create_model_copy(robot_root.find('.//'+tag), i, tag)
                    self.copy_elements(robot_copy, new_element)
            elif tag == 'keyframe':
                if robot_element is not None:
                    combined_root.append(robot_element)
                    # adjust qpos and control settings
                    key_element =  combined_root.find('.//keyframe/key')
                    if key_element is not None:
                        for attr in ['qpos', 'ctrl']:
                            item = key_element.get(attr)
                            if item is not None:
                                key_element.set(attr, ' '.join([item for i in range(len(coordinates))]))
            else:
                # copy model elements into combined scene
                if robot_element is not None:
                    combined_root.append(robot_element)

        # Save the combined model to a new XML file
        self.combined_model_path = self.base_path + "combined_model.xml"
        tree_combined = ET.ElementTree(combined_root)
        tree_combined.write(self.combined_model_path)

        # modifiy scene file to include new combined model
        scene_root.find('include').set('file', 'combined_model.xml')
        self.combined_scene_path = self.base_path + "combined_scene.xml"
        scene_tree.write(self.combined_scene_path)
    
    def setupRobotConfigs(self, num_robots):
        """
        Setup robot configurations
        """
        print(f"Loading model from: {self.combined_scene_path}")
        self.model = mujoco.MjModel.from_xml_path(self.combined_scene_path)
        self.data = mujoco.MjData(self.model)

        self.model.body_gravcomp[:] = float(self.gravity_compensation)
        self.model.opt.timestep = self.dt

        # End-effector site we wish to control.
        if True:
            site_name = [""]
        elif num_robots > 1:
            site_name = "attachment_site_copy_0"
        else:
            site_name = "attachment_site"
        self.site_id = None#self.model.site(site_name).id
        self.key_id = None#self.model.key(self.name_home_pose).id
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
    
    def runSimulation(self, num_robots, save_simulation_data=False):
        """
        Run the simulation and visualize it locally.
        """
        self.setupRobotConfigs(num_robots)
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
            #mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
            
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
            elements = robot_copy.findall('.//body') + robot_copy.findall('.//joint') + robot_copy.findall('.//freejoint') + robot_copy.findall('.//site') + robot_copy.findall('.//camera') + robot_copy.findall('.//light') + robot_copy.findall('.//geom')
            for element in elements:
                if 'name' in element.attrib:
                    element.attrib['name'] = f"{element.attrib['name']}_copy_{instance_id}"
        elif tag == 'actuator':
            # update names
            subsections = ['position', 'general', 'adhesion', 'motor']
            for subsection in subsections:
                for element in robot_copy.findall('.//'+subsection):
                    for attr in ['name', 'joint', 'tendon', 'motor', 'body', 'site']:
                        if attr in element.attrib:
                            element.set(attr, f"{element.get(attr)}_copy_{instance_id}") 

        elif tag == 'contact':
            # update names
            attributes = ['body1', 'body2', 'name']
            for element in robot_copy.findall('.//exclude'):
                for attr in attributes:
                    if attr in element.attrib: 
                        element.set(attr, f"{element.get(attr)}_copy_{instance_id}")

        elif tag == 'sensor':
            # update names
            for sensor in robot_copy:
                for attr in ['name', 'objname', 'site', 'actuator', 'joint']:
                    if attr in sensor.attrib:
                        sensor.set(attr, f"{sensor.attrib[attr]}_copy_{instance_id}")

        elif tag == 'tendon':
            # update names
            for element in robot_copy:
                if 'name' in element.attrib:
                    element.set('name', f"{element.attrib['name']}_copy_{instance_id}")
                for joint in element.findall('.//joint'):
                    joint.set('joint', f"{joint.get('joint')}_copy_{instance_id}")

        elif tag == 'equality':
            attributes = ['joint1', 'joint2']
            for element in robot_copy:
                for attr in attributes:
                    if attr in element.attrib:
                        element.set(attr, f"{element.attrib[attr]}_copy_{instance_id}")
        else:
            raise ValueError(f"Invalid element: {tag}")
        return robot_copy

if __name__ == "__main__":
    choices = ["kuka", "franka", "ur5e", "anymal_b", "anymal_c", 
               "cf2", "boston_d", "fruitfly", "aloha", "cassie",
               "franka3", "google_robot"]
    parser = argparse.ArgumentParser(description="Run MuJoCo simulation for different robots.")
    parser.add_argument("--robot", type=str, choices=choices, required=True, help="Type of robot to simulate")
    parser.add_argument('-n', type=int, help='select number of robots')
    args = parser.parse_args()
    if args.n < 0:
        raise ValueError('Number of robots should be greater than 0')
    coordinates = get_coordinates(args.n)

    simulation = MuJocoSimulation(args.robot, args.n, coordinates)
    simulation.runSimulation(args.n, False)
