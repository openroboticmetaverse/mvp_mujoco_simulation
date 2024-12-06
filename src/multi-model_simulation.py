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
import json
import requests
import os
import zipfile
import subprocess
import shutil
import logging
from definitions import LOG_PATH
import traceback
import sys
from collections import Counter
class MuJocoSimulation:
    """
    Class for the Mujoco Simulation.
    """
    # List of top-level Mujoco XML tags
    TOP_LEVEL_TAGS = [
        "compiler",
        "option",
        "size",
        "visual",
        "statistic",
        "default",
        "asset",
        "worldbody",
        "contact",
        "equality",
        "tendon",
        "actuator",
        "sensor",
        "keyframe",
        "custom"
    ]
    def __init__(self):
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

        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.setup_logger()

        # setup paths
        self.scene_path = "./scene/scene.xml"
        self.combined_model_path = "./scene/combined_models.xml"

        # Robot type configuration
        self.robot_path = []
        try:
            self.create_new_scene()
            # self.clean_scene_folder()
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            _, _, tb = sys.exc_info()
            lineno = tb.tb_lineno
            filename = tb.tb_frame.f_code.co_filename
            
            self.logger.error(f"An error occurred in {filename} at line {lineno}")
            self.logger.error(f"Error Type: {error_type}")
            self.logger.error(f"Error Message: {error_message}")
            self.logger.error("Traceback:")
            self.logger.error(traceback.format_exc())

    def setup_logger(self, log_level=logging.INFO, log_file_path=LOG_PATH):
        """
        Setup the logger for the simulation.
        
        Args:
            log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG)
            log_file_path (str): Path to the log file. If None, only console logging is used.
        """
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create console handler and set level
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        
        # Add console handler to logger
        self.logger.addHandler(ch)
        
        # If log_file_path is provided, set up file logging
        if log_file_path:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            
            # Create file handler and set level
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            
            # Add file handler to logger
            self.logger.addHandler(fh)
        
        self.logger.info(f"Logging setup complete. Level: {logging.getLevelName(log_level)}, File: {log_file_path if log_file_path else 'Console only'}")

    
    def create_new_scene(self, scene_id=1):
        """
        Creates an XML file containing all the models present in the specified scene.

        This function performs the following steps:
        1. Imports models based on the given scene_id.
        2. Creates a new XML structure for the combined scene.
        3. Processes each robot in the scene, extracting and modifying relevant XML sections.
        4. Combines all processed sections into a single XML file.

        The function handles various XML sections including compiler, option, asset, default,
        worldbody, contact, actuator, keyframe, and tendon. It ensures that robot-specific
        elements are properly named and positioned within the combined scene.

        Args:
            scene_id (int): The ID of the scene to create. Defaults to 1.

        The resulting XML file is stored at 'scene/combined_models.xml'.

        Note:
        - This function relies on several helper methods like parse_section and copy_elements.
        - It handles robot-specific modifications and ensures uniqueness of elements across different robots.
        - Special handling is implemented for sections like keyframe, where data from multiple robots needs to be combined.
        """

        # import models
        self.import_models(scene_id)
        self.base_names = []
        # create combined_models.xml
        combined_model = ET.Element("mujoco")

        # count existing robots and their tags
        existing_robots = {key:False for key in Counter(robot['robot_reference']["name"] for robot in self.robots).keys()}
        tags = {tag: {'exists': False, 'section': None} for tag in self.TOP_LEVEL_TAGS}

        for robot in self.robots:
            # extract robot name and count
            robot_name = robot['robot_reference']["name"]
            robot_tree = ET.parse(f"./scene/{robot_name}.xml")
            robot_root = robot_tree.getroot()

            for tag in self.TOP_LEVEL_TAGS:
                if tag == 'compiler':
                    # check if section exists
                    if not tags[tag]['exists']:
                        tags[tag]['exists'] = True
                        tags[tag]['section'] = ET.SubElement(combined_model, tag)

                        tags[tag]['section'].set("angle", "radian")
                        tags[tag]['section'].set("autolimits", "true")

                elif tag == 'option':
                    # check if section exists
                    if not tags[tag]['exists']:
                        tags[tag]['exists'] = True
                        tags[tag]['section'] = ET.SubElement(combined_model, tag)

                        tags[tag]['section'].set("integrator", "implicitfast")
                        tags[tag]['section'].set("cone", "elliptic")
                        tags[tag]['section'].set("impratio", "100")

                elif tag == 'asset':
                    # check if section exists
                    if not tags[tag]['exists']:
                        tags[tag]['exists'] = True
                        tags[tag]['section'] = ET.SubElement(combined_model, tag)

                    # check if robot assets exist
                    if existing_robots[robot_name]:
                        continue

                    # extract robot assets section
                    asset_section = robot_root.find("asset")

                    # parse assets section
                    self.parse_section(asset_section, tag, robot_name)

                    # copy assets section to combined model
                    self.copy_elements(asset_section, tags[tag]['section'])

                elif tag == 'default':
                    # check if section exists
                    if not tags[tag]['exists']:
                        tags[tag]['exists'] = True
                        tags[tag]['section'] = ET.SubElement(combined_model, tag)

                    # check if robot default section exists
                    if existing_robots[robot_name]:
                        continue

                    # extract robot default section
                    default_section = robot_root.find("default")

                    # parse default section
                    self.parse_section(default_section, tag, robot_name)

                    # copy default section to combined model
                    self.copy_elements(default_section, tags[tag]['section'])

                elif tag == 'worldbody':
                    # check if section exists
                    if not tags[tag]['exists']:
                        tags[tag]['exists'] = True
                        tags[tag]['section'] = ET.SubElement(combined_model, tag)

                    # extract robot worldbody section
                    worldbody_section = robot_root.find("worldbody")

                    # parse worldbody section
                    self.parse_section(worldbody_section, tag, robot_name, id=robot["id"], upper_level=True, pos=robot["position"])

                    # copy worldbody section to combined model
                    self.copy_elements(worldbody_section, tags[tag]['section'], exclude_tags=['light'])

                elif tag == 'contact':
                    # check if section exists
                    if not tags[tag]['exists']:
                        tags[tag]['exists'] = True
                        tags[tag]['section'] = ET.SubElement(combined_model, tag)

                    # extract robot contact section
                    contact_section = robot_root.find("contact")

                    # parse contact section
                    self.parse_section(contact_section, tag, robot_name, id=robot["id"])

                    # copy contact section to combined model
                    self.copy_elements(contact_section, tags[tag]['section'])

                elif tag == 'actuator':
                    # check if section exists
                    if not tags[tag]['exists']:
                        tags[tag]['exists'] = True
                        tags[tag]['section'] = ET.SubElement(combined_model, tag)

                    # extract robot actuator section
                    actuator_section = robot_root.find("actuator")

                    # parse actuator section
                    self.parse_section(actuator_section, tag, robot_name, id=robot["id"])

                    # copy actuator section to combined model
                    self.copy_elements(actuator_section, tags[tag]['section'])

                elif tag == 'keyframe':
                    # check if section exists
                    if not tags[tag]['exists']:
                        tags[tag]['exists'] = True
                        tags[tag]['section'] = ET.SubElement(combined_model, tag)
                        key_section = ET.SubElement(tags[tag]['section'], "key")
                        key_section.set("name", "home")
                        key_section.set("qpos", "")
                        key_section.set("ctrl", "")

                    # extract robot keyframe section
                    keyframe_section = robot_root.find("./keyframe/key")
                    if keyframe_section is not None:
                        # adjust qpos array
                        qpos = keyframe_section.get('qpos')
                        if qpos:
                            qpos_list = qpos.split(" ")
                            qpos_list = tags[tag]['section'].find('key').get('qpos').split(" ") + qpos_list
                            tags[tag]['section'].find('key').set('qpos', ' '.join(qpos_list))

                        # adjust ctrl array
                        ctrl = keyframe_section.get('ctrl')
                        if ctrl:
                            ctrl_list = ctrl.split(" ")
                            ctrl_list = tags[tag]['section'].find('key').get('ctrl').split(" ") + ctrl_list
                            tags[tag]['section'].find('key').set('ctrl', ' '.join(ctrl_list))

                    else:
                        # Count the number of joints (excluding the free joint for the base)
                        joint_count = len(robot_root.findall('.//joint'))

                        # Count the number of actuators
                        actuator_count = len(robot_root.findall('.//actuator/position'))

                        # Calculate qpos size (7 for base + number of joints)
                        qpos_size = 10 + joint_count
                        
                        # Adjust qpos size
                        qpos_list = tags[tag]['section'].find('key').get('ctrl').split(" ") + ['0'] * qpos_size
                        tags[tag]['section'].find('key').set('qpos', ' '.join(qpos_list))

                        # Calculate ctrl size (equal to the number of actuators)
                        ctrl_size = actuator_count

                        # adjust ctrl size
                        ctrl_list = tags[tag]['section'].find('key').get('ctrl').split(" ") + ['0'] * ctrl_size
                        tags[tag]['section'].find('key').set('ctrl', ' '.join(ctrl_list))

                elif tag == 'tendon':
                    # check if section exists
                    if not tags[tag]['exists']:
                        tags[tag]['exists'] = True
                        tags[tag]['section'] = ET.SubElement(combined_model, tag)

                    # extract robot tendon section
                    tendon_section = robot_root.find("tendon")

                    # parse tendon section
                    self.parse_section(tendon_section, tag, robot_name, id=robot["id"])

                    # copy tendon section to combined model
                    self.copy_elements(tendon_section, tags[tag]['section'])

                elif tag == 'equality':
                    # check if section exists
                    if not tags[tag]['exists']:
                        tags[tag]['exists'] = True
                        tags[tag]['section'] = ET.SubElement(combined_model, tag)

                    # extract robot equality section
                    equality_section = robot_root.find("equality")

                    # parse equality section
                    self.parse_section(equality_section, tag, robot_name, id=robot["id"])

                    # copy equality section to combined model
                    self.copy_elements(equality_section, tags[tag]['section'])
                                       
            existing_robots[robot_name] = True
                                
        # save combined models in a new XML
        tree = ET.ElementTree(combined_model)
        tree.write(self.combined_model_path, encoding="utf-8", xml_declaration=True)

    def parse_section(self, section, tag, robot_name="", id=0, upper_level=False, pos=None):
        """
        Parse and modify XML sections for robot models.

        This function parses different sections of an XML file representing a robot model,
        adjusting names, classes, and other attributes to ensure uniqueness when combining
        multiple robot models.

        Parameters:
        section (xml.etree.ElementTree.Element): The XML section to parse.
        tag (str): The tag name of the section being parsed (e.g., 'asset', 'default', 'worldbody').
        robot_name (str, optional): The name of the robot model. Defaults to "".
        id (int, optional): An identifier for the robot instance. Defaults to 0.
        upper_level (bool, optional): Flag indicating if this is a top-level element. Defaults to False.
        pos (list, optional): Position coordinates for adjusting element positions. Defaults to None.

        Returns:
        None. The function modifies the input XML section in-place.
        """
        if section is None:
            return

        if tag == 'asset':
            for element in section:
                # adjust element name
                if element.get('name') is None:
                    file_path = element.get('file')
                    if file_path:
                        file_name = os.path.basename(file_path)
                        name_without_extension = os.path.splitext(file_name)[0]
                        element.set('name', f"{name_without_extension}_{robot_name}")
                elif element.get('name') is not None and robot_name not in element.get('name'):
                    element.set('name', f"{element.get('name')}_{robot_name}")

                # adjust class names
                if element.get('class') is not None and robot_name not in element.get('class'):
                    element.set('class', f"{element.get('class')}_{robot_name}")

                # adjust texture paths
                if element.get('texture') is not None:
                    texture_name = element.get('texture')
                    element.set('texture', f"{texture_name}_{robot_name}")

                # adjust mesh paths
                if element.get('file') is not None:
                    mesh_file = element.get('file')
                    element.set('file', f"assets_{robot_name}/{mesh_file}")

        elif tag == 'default':
            # adjust section name
            if section.get('class') is not None and robot_name not in section.get('class'):
                section.set('class', f"{section.get('class')}_{robot_name}")

            # adjust mesh names
            for geom_section in section.findall('./geom'):
                if geom_section.get('material') is not None and robot_name not in geom_section.get('material'):
                    geom_section.set('material', f"{geom_section.get('material')}_{robot_name}")

            # recursively parse subsections
            subsections = section.findall('./default')
            if subsections is not None:
                for subsection in subsections:
                    self.parse_section(subsection, tag, robot_name, id)
            else:
                return

        elif tag == 'worldbody':
            for element in section:
                if element.tag == 'light':
                    continue
                # adjust element name
                if element.get('name') is not None:
                    if element.get('name') not in robot_name:
                        element.set('name', f"{element.get('name')}_{robot_name}_{id}")
                    else:
                        element.set('name', f"{element.get('name')}_{id}")

                # adjust class names
                if element.get('class') is not None and robot_name not in element.get('class'):
                    element.set('class', f"{element.get('class')}_{robot_name}")

                # adjust child class names
                if element.get('childclass') is not None and robot_name not in element.get('childclass'):
                    element.set('childclass', f"{element.get('childclass')}_{robot_name}")

                # adjust joint names
                if element.get('joint') is not None and robot_name not in element.get('joint'):
                    element.set('joint', f"{element.get('joint')}_{robot_name}_{id}")

                # adjust material names
                if element.get('material') is not None and robot_name not in element.get('material'):
                    element.set('material', f"{element.get('material')}_{robot_name}")

                # adjust mesh names
                if element.get('mesh') is not None:
                    mesh_file = element.get('mesh')
                    element.set('mesh', f"{mesh_file}_{robot_name}")

                # adjust position
                if upper_level:
                    if element.get('pos') is not None:
                        base_pos = element.get('pos').split(" ")
                        curr_pos = [str(float(p) + float(base_pos[i])) for i, p in enumerate(pos)]
                    else:
                        curr_pos = [str(p) for p in pos]
                    element.set('pos', " ".join(curr_pos))

                    # append base names
                    self.base_names.append(element.get('name'))

            # recursively parse subsections
            subsections = section.findall('body')
            if subsections is not None:
                for subsection in subsections:
                    self.parse_section(subsection, tag, robot_name, id)
            else:
                return

        elif tag == 'contact':
            for element in section:
                element.set('body1', f"{element.get('body1')}_{robot_name}_{id}")
                element.set('body2', f"{element.get('body2')}_{robot_name}_{id}")

        elif tag == 'actuator':
            for element in section:
                # Remove the 'gainprm' attribute if it exists
                if element.tag == 'general':
                    if 'gainprm' in element.attrib:
                        del element.attrib['gainprm']

                    if 'biasprm' in element.attrib:
                        del element.attrib['biasprm']
                    
                    # Convert to 'position' actuator if conditions are met
                    if element.get('tendon') is None:
                        element.tag = 'position'

                    # adjust tendon names
                    if element.get('tendon') is not None:
                        tendon_name = element.get('tendon')
                        element.set('tendon', f"{tendon_name}_{robot_name}_{id}")

                # adjust class names
                if element.get('class') is not None and robot_name not in element.get('class'):
                    element.set('class', f"{element.get('class')}_{robot_name}")

                # adjust joint names
                if element.get('joint') is not None:
                    element.set('joint', f"{element.get('joint')}_{robot_name}_{id}")

                # adjust names
                if element.get('name') is not None:
                    if element.get('name') not in robot_name:
                        element.set('name', f"{element.get('name')}_{robot_name}_{id}")
                    else:
                        element.set('name', f"{element.get('name')}_{id}")

        elif tag == 'tendon':
            section.find('fixed').set('name', f"{section.find('fixed').get('name')}_{robot_name}_{id}")
            for element in section.findall('.//joint'):
                if element.get('joint') is not None:
                    element.set('joint', f"{element.get('joint')}_{robot_name}_{id}")

        elif tag == 'equality':
            for element in section:
                element.set('joint1', f"{element.get('joint1')}_{robot_name}_{id}")
                element.set('joint2', f"{element.get('joint2')}_{robot_name}_{id}")


    def copy_elements(self, source_element, target_element, exclude_tags=None, pos=None):
        """
        Copy elements from a source XML element to a target XML element.

        This function iterates through the child elements of the source element
        and appends them to the target element, excluding specified tags.

        Parameters:
        source_element (xml.etree.ElementTree.Element): The source XML element from which to copy.
        target_element (xml.etree.ElementTree.Element): The target XML element to which elements are copied.
        exclude_tags (list, optional): A list of tag names to exclude from copying. Defaults to None.
        pos (list, optional): Unused parameter. Defaults to None.

        Returns:
        None
        """
        if exclude_tags is None:
            exclude_tags = []
        if source_element is not None:
            for element in source_element:
                if element.tag not in exclude_tags:
                    target_element.append(element)
        return

    def import_models(self, scene_id):
        """
        Import the source files of the models in the specified scene from the GitHub repository.

        This function reads a JSON file containing scene information, filters robots based on the
        given scene_id, and downloads the necessary XML files and assets for each robot from the
        GitHub repository if they don't already exist locally.

        Parameters:
        scene_id (int): The ID of the scene for which to import robot models.

        Returns:
        None

        Raises:
        ConnectionError: If there's a failure in downloading the robot XML file.
        FileNotFoundError: If the assets folder for a robot cannot be found in the repository.

        Side effects:
        - Modifies self.robots list to include robots for the specified scene.
        - Downloads and saves robot XML files and assets to the local file system.
        - Logs information and warnings about the import process.
        """
        # Open the JSON file and load its contents
        robots_list_path = "./scene/scene.json"
        with open(robots_list_path, 'r') as f:
            scene = json.load(f)
        self.robots = []
        for robot in scene["robots"]:
            # check if robot belongs to the scene
            if robot["scene_id"] == scene_id:
                self.robots.append(robot)

            # check if robot files are already imported
            if not os.path.exists(f"scene/assets_{robot['robot_reference']['name']}"):
                repo_owner = "openroboticmetaverse"
                repo_name = "robot-description"

                                # Download XML file using GitHub API
                repo_owner = "openroboticmetaverse"
                repo_name = "robot-description"
                file_path = robot["robot_reference"]["file"].split("main/")[1]
                api_url = self.get_api_url(repo_owner, repo_name, file_path)
                
                headers = {"Accept": "application/vnd.github.v3.raw"}
                response = requests.get(api_url, headers=headers)
                
                if response.status_code != 200:
                    raise ConnectionError(f"Failed to download file {robot['robot_reference']['file']}")
                
                with open(f"./scene/{robot['robot_reference']['name']}.xml", "wb") as f:
                    f.write(response.content)
                
                self.logger.info(f"Successfully downloaded {robot['robot_reference']['name']}.xml")

                # Download assets folder using GitHub API
                
                mujoco_index = robot['robot_reference']['file'].index("mujoco")
                assets_path = robot['robot_reference']['file'][mujoco_index:].rsplit('/', 1)[0] + "/assets"

                assets_found = False
                api_url = self.get_api_url(repo_owner, repo_name, assets_path)
                self.logger.info(f"Attempting to fetch folder contents from: {api_url}")

                headers = {"Accept": "application/vnd.github.v3+json"}
                response = requests.get(api_url, headers=headers, timeout=10)

                if response.status_code == 200:
                    assets_found = True
                    folder_contents = response.json()
                else:
                    self.logger.warning(f"Failed to find assets at {api_url}")

                if not assets_found:
                    raise FileNotFoundError(f"Could not find assets folder for {robot['robot_reference']['name']}")

                os.makedirs(f"scene/assets_{robot['robot_reference']['name']}", exist_ok=True)

                for item in folder_contents:
                    if item['type'] == 'file':
                        file_url = item['download_url']
                        file_name = item['name']
                        file_path = f"scene/assets_{robot['robot_reference']['name']}/{file_name}"

                        file_response = requests.get(file_url, timeout=10)
                        file_response.raise_for_status()

                        with open(file_path, 'wb') as f:
                            f.write(file_response.content)

                self.logger.info(f"Successfully downloaded assets for {robot['robot_reference']['name']}")

    def get_api_url(self, repo_owner, repo_name, file_path):
        return f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"

    def clean_scene_folder(self):
        """
        Cleans the scene folder by removing all files and directories except 'scene.json' and 'scene.xml'.

        This function iterates through all items in the './scene' directory and removes them,
        with the exception of 'scene.json' and 'scene.xml'. It removes files using os.remove()
        and directories using shutil.rmtree().

        Parameters:
        None

        Returns:
        None

        Side effects:
        - Deletes files and directories in the './scene' folder
        - Logs information about removed files/directories using self.logger
        """
        scene_folder = './scene'
        for filename in os.listdir(scene_folder):
            filepath = os.path.join(scene_folder, filename)
            if filename not in ['scene.json', 'scene.xml']:
                if os.path.isfile(filepath):
                    os.remove(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                self.logger.info(f"Successfully Removed {filepath}")

    def setupRobotConfigs(self, num_robots=0):
        """
        Setup robot configurations
        """
        self.logger.info(f"Loading model from: {self.scene_path}")
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)

        # Set the integrator to semi-implicit Euler
        # self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER


        # Alternatively, for even more stability (but slower simulation), use RK4:
        # self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4

        self.model.body_gravcomp[:] = float(self.gravity_compensation)
        self.model.opt.timestep = self.dt



        # End-effector site we wish to control.
        tree = ET.parse(self.combined_model_path)
        root = tree.getroot()
        sites = [element.get('name') for element in root.findall('.//site') if element.get('name')]

        # self.site_id = self.model.site(site_name).id
        # self.key_id = self.model.key(self.name_home_pose).id
        return  

    def runSimulation(self):
        """
        Run the simulation and visualize it locally, with slower, more controlled robot movements.
        """
        self.setupRobotConfigs()
    
        # Launch the viewer.
        simViewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
    
        # Reduce the maximum joint velocity for slower movements
        max_joint_vel = 0.1  # radians per second
    
        # Increase the number of sub-steps for smoother motion
        n_substeps = 20
        substep_time = self.dt / n_substeps
    
        # Identify actuators for each robot
        tree = ET.parse(self.combined_model_path)
        root = tree.getroot()
        robot_actuators = []
        for robot in self.robots:
            actuator_names = [element.get('name') for element in root.findall('./actuator/position') if element.get('name').endswith(str(robot['id']))]
            robot_actuators.append([self.model.actuator(name).id for name in actuator_names])
        
        # Initialize target positions for each robot
        target_positions = [self.data.ctrl[actuators].copy() for actuators in robot_actuators]
    
        with simViewer as viewer:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    
            while viewer.is_running():
                step_start = time.time()
    
                # Update target positions less frequently
                if np.random.random() < 0.02:  # 2% chance to change target each frame
                    for i, actuators in enumerate(robot_actuators):
                        target_positions[i] = self.data.ctrl[actuators] + np.random.uniform(-0.1, 0.1, len(actuators))
    
                # Move each robot towards its target position
                for i, actuators in enumerate(robot_actuators):
                    current_pos = self.data.ctrl[actuators]
                    target_pos = target_positions[i]
                    
                    # Calculate the direction to move
                    direction = target_pos - current_pos
                    
                    # Limit the movement speed
                    movement = np.clip(direction, -max_joint_vel * self.dt, max_joint_vel * self.dt)
                    
                    # Update the control signals
                    self.data.ctrl[actuators] += movement
    
                # Step the simulation with sub-steps
                for _ in range(n_substeps):
                    mujoco.mj_step(self.model, self.data)
    
                # Update the viewer
                viewer.sync()
    
                # Control the simulation speed
                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    
        
if __name__ == "__main__":
    simulation = MuJocoSimulation()
    simulation.runSimulation()
