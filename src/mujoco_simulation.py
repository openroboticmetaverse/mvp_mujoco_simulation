"""
Differential IK with nullspace control on a 7-DoF Panda.
"""
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

        # Integration timestep in seconds. This corresponds to the amount of time the joint
        # velocities will be integrated for to obtain the desired joint positions. 
        # Source values: 6-DoF: 1.0, 7-DoF: 0.1
        self.integration_dt: float = 1.0

        # Damping term for the pseudoinverse. This is used to prevent joint velocities from
        # becoming too large when the Jacobian is close to singular.
        # Source values: 6-DoF: 1e-5, 7-DoF: 1e-4
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
        self.dt: float = 0.002

        # Maximum allowable joint velocity in rad/s.
        self.max_angvel = 0.785

        # Robot specific settings
            # UR5e - "universal_robots_ur5e/scene.xml"
            # Panda - "franka_emika_panda/scene.xml" 
        self.robot_path = "franka_emika_panda/scene.xml"

        # Joints you wish to control. You find them in the xml file.
            # UR5e - ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
            # Panda - ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        
        # Name of initial joint configuration - see xml
        self.name_home_pose = "home"



    def runServer(self):
        """
        Start server.
        Need to be executed using asyncrio: asyncio.run(server.runServer())
        """
        assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."
        print(">> Server is runnung and waiting for the client")

        # Initialize robot values
        self.setupRobotConfigs()

        start_server = websockets.serve(self.serverExecutable, self.host, self.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

        print(">> Server was stopped")



    def setupRobotConfigs(self):
        """
        Setup robot configurations
        """
        # Load the model and data.
        self.model = mujoco.MjModel.from_xml_path("../config/" + self.robot_path)
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

        

    async def serverExecutable(self, websocket, path):
        """
        Main behavior function for the server - currently only publishing demo data
        Function is executed when a client connects to the server.
        """
        print(">> Server listening on Port " + str(self.port))

        num_joints = len(self.joint_names)

        # Pre-allocate numpy arrays.
        jac = np.zeros((6, self.model.nv))
        diag = self.damping * np.eye(self.model.nv)
        eye = np.eye(self.model.nv)
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
            # Reset the simulation.
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
            
            # Reset the free camera.
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)

            # Enable site frame visualization.
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            
            while viewer.is_running():
                step_start = time.time()
                
                # Get motion function
                self.data.mocap_pos[self.mocap_id, 0:2] = circular_motion(self.data.time, 0.1, 0.5, 0.0, 0.5)

                # Spatial velocity (aka twist).
                dx = self.data.mocap_pos[self.mocap_id] - self.data.site(self.site_id).xpos
                twist[:3] = self.Kpos * dx / self.integration_dt
                mujoco.mju_mat2Quat(site_quat, self.data.site(self.site_id).xmat)
                mujoco.mju_negQuat(site_quat_conj, site_quat)
                mujoco.mju_mulQuat(error_quat, self.data.mocap_quat[self.mocap_id], site_quat_conj)
                mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
                twist[3:] *= self.Kori / self.integration_dt

                # Jacobian
                mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], self.site_id)

                # Solve for joint velocities: J * dq = twist using damped least squares.
                dq = np.linalg.solve(jac.T @ jac + diag, jac.T @ twist)

                # Nullspace control biasing joint velocities towards the home configuration.
                if num_joints == 7:
                    dq += (eye - np.linalg.pinv(jac) @ jac) @ (self.Kn * (self.q0 - self.data.qpos[self.dof_ids]))

                # Clamp maximum joint velocity.
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > self.max_angvel:
                    dq *= self.max_angvel / dq_abs_max

                # Integrate joint velocities to obtain joint positions - copying is important
                q = self.data.qpos.copy()

                # Adds a vector in the format of qvel (scaled by dt) to a vector in the format of qpos.
                mujoco.mj_integratePos(self.model, q, dq, self.integration_dt)
                np.clip(q, *self.model.jnt_range.T, out=q)

                # Set the control signal and step the simulation.
                self.data.ctrl[self.actuator_ids] = q[self.dof_ids]
                mujoco.mj_step(self.model, self.data)

                viewer.sync()
                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                try:
                    # Transform array to send via websockets
                    q_format = create_output_string(self.joint_names, q)
                    q_string = pickle.dumps(q_format)
                    await websocket.send(q_string)
                except websockets.exceptions.ConnectionClosedError:
                    print("Client closed connection without sending close frame")
                    await websocket.close()
                    break




if __name__ == "__main__":
    server = MuJocoBackendServer()
    asyncio.run(server.runServer())