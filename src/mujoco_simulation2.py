import mujoco
import mujoco.viewer
import numpy as np
import time
import asyncio
import websockets
import os
from helpers import create_output_string
import argparse
import motorcortex
import threading
import queue
import sys

class MuJocoSimulation:
    """
    Class for the MuJoCo Simulation.
    Integrates with Motorcortex to receive control data.
    """

    def __init__(self, for_mtx):
        # Websocket Settings
        self.host = "0.0.0.0"
        self.port = 8081

        # Simulation timestep in seconds.
        self.dt: float = 0.01

        # Define path to robot xml file
        # Update this path to point to your robot's XML model file
        self.robot_path = "/home/amine/Documents/orom/mvp_mujoco_simulation/config/ur5/scene.xml"

        # Define joint names of the robot.
        # Ensure these match your robot's joint names in the MuJoCo model
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

        # Name of initial joint configuration - as defined in your XML model
        self.name_home_pose = "home"

        # Flag to indicate if control is obtained via Motorcortex API
        self.for_mtx = for_mtx

        # Queue for inter-thread communication
        self.mtx_queue = queue.Queue()

    def runServer(self):
        """
        Start the server.
        """
        assert mujoco.__version__ >= "3.1.0", "Please upgrade to MuJoCo 3.1.0 or later."
        print(f">> Server is running and waiting for the client at {self.host}:{self.port}")

        # Initialize robot configurations
        self.setupRobotConfigs()

        # Start Motorcortex thread if required
        if self.for_mtx:
            threading.Thread(target=self.mtx_main, daemon=True).start()

        # Start the WebSocket server
        start_server = websockets.serve(self.serverExecutable, self.host, self.port)

        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

        print(">> Server was stopped")

    def setupRobotConfigs(self):
        """
        Setup robot configurations.
        """
        # Load the model and data.
        self.model = mujoco.MjModel.from_xml_path(self.robot_path)
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = self.dt

        # Get the indices of the joints in the model
        self.dof_ids = np.array([self.model.joint(name).qposadr[0] for name in self.joint_names])

        # Initial joint configuration saved as a keyframe in the XML file.
        self.key_id = self.model.key(self.name_home_pose).id
        self.q0 = self.model.key(self.name_home_pose).qpos

    async def serverExecutable(self, websocket, path):
        """
        Main behavior function for the server.
        Executed when a client connects to the server.
        """
        print(">> Server listening on Port " + str(self.port))

        num_joints = len(self.joint_names)
        print("Number of joints: ", num_joints)

        # Launch the simulation viewer
        simViewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        with simViewer as viewer:
            # Reset the simulation to the initial keyframe.
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)

            # Reset the free camera.
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
            # Optional: Enable site frame visualization.
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            while viewer.is_running():
                step_start = time.time()

                if self.for_mtx:
                    # Try to get data from the queue
                    try:
                        q_from_mtx = self.mtx_queue.get_nowait()
                        q = np.array(q_from_mtx)
                        print("Received q from Motorcortex:", q)
                    except queue.Empty:
                        # No new data; use the current joint positions
                        q = self.data.qpos[self.dof_ids].copy()
                        print("No new q from Motorcortex, using current q:", q)
                else:
                    # If not using Motorcortex, you might set default positions or exit
                    print("Not using Motorcortex, exiting.")
                    break

                # Update the joint positions in the MuJoCo model
                self.data.qpos[self.dof_ids] = q

                # Update the kinematics without advancing time or simulating dynamics
                mujoco.mj_forward(self.model, self.data)

                # Send data over WebSocket if needed
                mapped_joint_names = self.joint_names  # Or use any mapping you prefer
                q_string = create_output_string(mapped_joint_names, "", q)

                try:
                    await websocket.send(q_string)
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed.")
                    break
                except Exception as ex:
                    print(f"WebSocket error: {ex}")
                    break

                # Synchronize the viewer
                viewer.sync()

                # Maintain the loop timing
                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    await asyncio.sleep(time_until_next_step)

    def mtx_main(self):
        """
        Main function to handle Motorcortex communication.
        Runs in a separate thread.
        """
        # Initialize connection
        parameter_tree = motorcortex.ParameterTree()
        # Open request and subscribe connection
        try:
            req, sub = motorcortex.connect(
                "wss://192.168.56.101:5568:5567",
                motorcortex.MessageTypes(),
                parameter_tree,
                certificate="mcx.cert.crt",
                timeout_ms=1000,
                login="admin",
                password="vectioneer"
            )
        except RuntimeError as err:
            print(err)
            sys.exit()

        paths = ['root/AxesControl/axesPositionsActual']  # Update this path if necessary
        divider = 100  # Adjust the divider as needed

        # Subscribe and wait for the reply
        subscription = sub.subscribe(paths, 'group1', divider)
        is_subscribed = subscription.get()
        if is_subscribed and is_subscribed.status == motorcortex.OK:
            print(f"Subscription successful, layout: {subscription.layout()}")
        else:
            print(f"Subscription failed, paths: {paths}")
            sub.close()
            return

        # Set the callback function
        callback_function = self.callback_generator(self.mtx_queue)
        subscription.notify(callback_function)

        # Keep the thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sub.close()

    def callback_generator(self, mtx_queue):
        def build_q_from_mtx(parameters):
            # Assuming that the first parameter contains the joint positions
            q_from_mtx = parameters[0].value
            print("Received joint positions from Motorcortex:", q_from_mtx)
            mtx_queue.put(q_from_mtx)
        return build_q_from_mtx

def get_args():
    parser = argparse.ArgumentParser(description="Matrix option parser")
    parser.add_argument('--for-mtx', action='store_true', help='Set this flag for Motorcortex option')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    server = MuJocoSimulation(for_mtx=args.for_mtx)
    server.runServer()
