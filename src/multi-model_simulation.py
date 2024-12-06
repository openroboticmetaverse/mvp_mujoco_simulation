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
        This function creates an xml file containing all the models that are present in the scene_id.  
        The new xml file is stored in scene/combined_models.xml.
        """
        # import models
        self.import_models(scene_id)

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
                if element.get('name') is not None and robot_name not in element.get('name'):
                    element.set('name', f"{element.get('name')}_{robot_name}")

                # adjust class names
                if element.get('class') is not None and robot_name not in element.get('class'):
                    element.set('class', f"{element.get('class')}_{robot_name}")

                # adjust texture paths
                if element.get('texture') is not None:
                    texture_file = element.get('texture')
                    element.set('texture', f"assets_{robot_name}/{texture_file}")

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
                # adjust element name
                if element.get('name') is not None and robot_name not in element.get('name'):
                    element.set('name', f"{element.get('name')}_{robot_name}_{id}")

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

                # adjust position
                if upper_level:
                    if element.get('pos') is not None:
                        base_pos = element.get('pos').split(" ")
                        pos = [str(float(p) + float(base_pos[i])) for i, p in enumerate(pos)]
                        element.set('pos', " ".join(pos))

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
                # adjust class names
                if robot_name not in element.get('class'):
                    element.set('class', f"{element.get('class')}_{robot_name}")

                # adjust joint names
                element.set('joint', f"{element.get('joint')}_{robot_name}_{id}")

                # adjust names
                element.set('name', f"{element.get('name')}_{robot_name}_{id}")

    def copy_elements(self, source_element, target_element, exclude_tags=None, pos=None):
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
            #mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
            
            # Reset the free camera.
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
            # Enable site frame visualization.
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            while viewer.is_running():
                step_start = time.time()
                continue
                
    
if __name__ == "__main__":
    simulation = MuJocoSimulation()
    # simulation.runSimulation()
