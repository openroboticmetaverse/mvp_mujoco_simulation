import os
import mujoco
import mujoco.viewer
import numpy as np
import xml.etree.ElementTree as ET
import time
import numpy as np
import requests
qpos="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.785398 0 -1.5708 0 0 0 0 0.785398 0 -1.5708 0 0 0 0 0 0 -1.57079 0 1.57079 -0.7853" 
ctrl="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.785398 0 -1.5708 0 0 0 0 0.785398 0 -1.5708 0 0 0 0 0 0 -1.57079 0 1.57079 -0.7853"
print(len(qpos.split(" ")))
print(len(ctrl.split(" ")))