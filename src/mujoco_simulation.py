import mujoco
import mujoco.viewer
import numpy as np
import time
import asyncio
import websockets
import pickle

from helpers import create_output_string
from motion_functions import circular_motion

class MuJocoBackendServer:
    """
    Class for the Mujoco backend server.
    Currently it is only publishing demo data!

    TODO: Deactivate visualisation
    TODO: Function: Move to point
    TODO: Select robot - get data from config-file
    """
    
    def __init__(self):
        # Server data
        self.host = "localhost"
        self.port = 8081

        # Controller parameters.
        self.integration_dt: float = 1.0                            # Integration timestep (seconds).
        self.damping: float = 1e-5                                  # Damping term for the pseudoinverse.

        #self.robot_path = "universal_robots_ur5e/scene.xml"         # Robot folder path in config
        self.robot_path = "kuka_iiwa_14/scene.xml" 

        # Joints you wish to control. You find them in the xml file.
        #self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow",
        #                    "wrist_1", "wrist_2", "wrist_3"]
        self.joint_names = ["joint1", "joint2", "joint3", "joint4",
                            "joint5", "joint6", "joint7"]
        
        self.name_home_pose = "home"                                # Name of initial joint configuration - see xml


    def runServer(self):
        """
        Start server.
        Need to be executed using asyncrio: asyncio.run(server.runServer())
        """
        print("Server is runnung and waiting for the client")

        self.setupRobotConfigs()

        start_server = websockets.serve(self.serverExecutable, self.host, self.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

        print("Server was stopped")


    def setupRobotConfigs(self):
        """
        Setup robot configurations
        """

        # Load the model and data.
        self.model = mujoco.MjModel.from_xml_path("../config/" + self.robot_path)
        self.data = mujoco.MjData(self.model)

        self.num_joints = len(self.joint_names)

        # Set PD gains.
        #Kp = np.asarray([2000.0, 2000.0, 2000.0, 500.0, 500.0, 500.0])
        #Kd = np.asarray([100.0, 100.0, 100.0, 50.0, 50.0, 50.0])
        Kp = np.asarray([2000.0, 2000.0, 2000.0, 500.0, 500.0, 500.0, 500.0])
        Kd = np.asarray([100.0, 100.0, 100.0, 50.0, 50.0, 50.0,  50.0])

        #self.model.actuator_gainprm[:, 0] = Kp
        #self.model.actuator_biasprm[:, 1] = -Kp
        #self.model.actuator_biasprm[:, 2] = -Kd

        # Enable gravity compensation. Set to 0.0 to disable.
        self.model.body_gravcomp[:] = 1.0

        # Simulation and control timesteps. Feel free to change these.
        dt = 0.002
        self.model.opt.timestep = dt
        self.control_dt = 10*dt  # Must be a multiple of dt.
        self.n_steps = int(round(self.control_dt / dt))

        # End-effector site we wish to control.
        site_name = "attachment_site"
        self.site_id = self.model.site(site_name).id

        self.dof_ids = np.array([self.model.joint(name).id for name in self.joint_names])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in self.joint_names])

        # Initial joint configuration saved as a keyframe in the XML file.
        key_id = self.model.key(self.name_home_pose).id

        # Mocap body we will control with our mouse.
        mocap_name = "target"
        self.mocap_id = self.model.body(mocap_name).mocapid[0]

        # Reset the simulation to the initial joint ration.
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        
    async def serverExecutable(self, websocket, path):
        """
        Main behavior function for the server - currently only publishing demo data
        Function is executed when a client connects to the server.
        """

        print("Server listening on Port " + str(self.port))

        # Pre-allocate numpy arrays.
        jac = np.zeros((6, self.model.nv))
        diag = self.damping * np.eye(self.model.nv)
        twist = np.zeros(6)
        site_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)

        # TODO: Implement differentation when visualisation should be opened for debugging and when it 
        # should run in the backend to save computational power
        simViewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )


        with simViewer as viewer:
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
            while viewer.is_running():
                step_start = time.time()
                
                # Get motion function
                self.data.mocap_pos[self.mocap_id, 0:2] = circular_motion(self.data.time, 0.1, 0.5, 0.0, 0.5)

                # Spatial velocity (aka twist).
                twist[:3] = self.data.mocap_pos[self.mocap_id] - self.data.site(self.site_id).xpos
                mujoco.mju_mat2Quat(site_quat, self.data.site(self.site_id).xmat)
                mujoco.mju_negQuat(site_quat_conj, site_quat)
                mujoco.mju_mulQuat(error_quat, self.data.mocap_quat[self.mocap_id], site_quat_conj)

                # Only determined by testing: twist needs to be 3 elements long (at least for robots with 6 or 7 joints)
                mujoco.mju_quat2Vel(twist[-3:], error_quat, 1.0)

                # Jacobian
                # Only determined by testing: second jacobian needs to be 3 elements long (at least for robots with 6 or 7 joints)
                mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[-3:], self.site_id)

                # Solve for joint velocities: J * dq = twist using damped least squares.
                dq = np.linalg.solve(jac.T @ jac + diag, jac.T @ twist)

                # Integrate joint velocities to obtain joint positions - copying is important
                q = self.data.qpos.copy()

                # Transform array to send via websockets
                q_format = create_output_string(self.joint_names, q)
                q_string = pickle.dumps(q_format)
                await websocket.send(q_string)

                # Adds a vector in the format of qvel (scaled by dt) to a vector in the format of qpos.
                mujoco.mj_integratePos(self.model, q, dq, self.integration_dt)
                np.clip(q, *self.model.jnt_range.T, out=q)

                # Set the control signal and step the simulation.
                self.data.ctrl[self.actuator_ids] = q[self.dof_ids]
                mujoco.mj_step(self.model, self.data, self.n_steps)

                viewer.sync()
                time_until_next_step = self.control_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                
                # Debugging
                #breakpoint()



if __name__ == "__main__":
    server = MuJocoBackendServer()
    asyncio.run(server.runServer())