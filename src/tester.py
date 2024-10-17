
from mujoco_simulation import MuJocoSimulation
import websockets, socket
ws = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

sim = MuJocoSimulation()
sim.setupRobotConfigs()


sim.serverExecutable( ws, None)

print(2)