import os
import mujoco
import mujoco.viewer
import numpy as np
import xml.etree.ElementTree as ET
import time
import numpy as np
import requests

response = requests.get('https://github.com/openroboticmetaverse/robot-description/blob/main/mujoco/anybotics_anymal_b/anymal_b.xml')
print(response)