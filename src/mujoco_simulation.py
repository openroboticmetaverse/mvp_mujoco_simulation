import mujoco
import mujoco.viewer
import numpy as np
import time
import asyncio
import websockets
import os
from helpers import create_output_string
from motion_functions import circular_motion, clifford_attractor
import argparse
from mtx import mtx_

class MuJocoSimulation:
    """
    Class for the Mujoco Simulation.
    Currently it is only publishing demo data!
    """
    
    def __init__(self, for_mtx):
        # Websocket Settings
        # self.host = "localhost"
        self.host = "0.0.0.0"
        self.port = 8081

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

        # Define path to robot xml file
            # UR5e - "universal_robots_ur5e/scene.xml"
            # Panda - "franka_emika_panda/scene.xml" 
        self.robot_path = "/home/amine/Documents/orom/mvp_mujoco_simulation/config/franka_emika_panda/scene.xml"

        # Define joint names of the robot. They have to match the names of the urdf-file.
            # UR5e - ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
            # Panda - ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

        # Used as prefix of the joint names in the data for the websocket.
        self.joint_name_prefix = "panda_"
        
        # Name of initial joint configuration - see xml
        self.name_home_pose = "home"

        #case if the control is obtained via motorcortex api
        self.for_mtx = for_mtx


    def runServer(self):
        """
        Start server.
        Need to be executed using asyncio: asyncio.run(server.runServer())
        """
        assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."
        print(f">> Server is runnung and waiting for the client at {self.host}:{self.port}")

        # Initialize robot values
        self.setupRobotConfigs()

        # Code is waiting here until a client connects. Then the function self.serverExecutable is executed.
        # If the client disconnects the function stops and starts again if a new client connects.
        start_server = websockets.serve(self.serverExecutable, self.host, self.port)
        
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
        
        if self.for_mtx:
            mtx_.main(self)

        print(">> Server was stopped")



    def setupRobotConfigs(self):
        """
        Setup robot configurations
        """
        # Load the model and data.
        self.model = mujoco.MjModel.from_xml_path(self.robot_path)
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

        # Pre-allocate numpy arrays.
        self.jac = np.zeros((6, self.model.nv))
        self.diag = self.damping * np.eye(self.model.nv)
        self.eye = np.eye(self.model.nv)
        self.twist = np.zeros(6)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)

    def build_q_from_mjc(self, num_joints):
        # Get demo motion function
            a = 1.5
            b = -1.8
            c = 1.6
            d = 0.9
            self.data.mocap_pos[self.mocap_id, 0:3] = clifford_attractor(self.data.time, a, b, c, d)
            #self.data.mocap_pos[self.mocap_id, 0:3] = np.array([0.514, 0.55, 0.5])
            #self.data.mocap_pos[self.mocap_id, 0:2] = circular_motion(self.data.time, 0.1, 0.5, 0.0, 0.5)

            # Spatial velocity (aka twist).
            dx = self.data.mocap_pos[self.mocap_id] - self.data.site(self.site_id).xpos
            self.twist[:3] = self.Kpos * dx / self.integration_dt
            mujoco.mju_mat2Quat(self.site_quat, self.data.site(self.site_id).xmat)
            mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
            mujoco.mju_mulQuat(self.error_quat, self.data.mocap_quat[self.mocap_id], self.site_quat_conj)
            mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
            self.twist[3:] *= self.Kori / self.integration_dt

            # Jacobian
            mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)

            # Solve for joint velocities: J * dq = twist using damped least squares.
            dq = np.linalg.solve(self.jac.T @ self.jac + self.diag, self.jac.T @ self.twist)

            # Nullspace control biasing joint velocities towards the home configuration.
            if num_joints == 7:
                dq += (self.eye - np.linalg.pinv(self.jac) @ self.jac) @ (self.Kn * (self.q0 - self.data.qpos[self.dof_ids]))

            # Clamp maximum joint velocity.
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > self.max_angvel:
                dq *= self.max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions - copying is important
            q = self.data.qpos.copy()

            # Adds a vector in the format of qvel (scaled by dt) to a vector in the format of qpos.
            mujoco.mj_integratePos(self.model, q, dq, self.integration_dt)
            np.clip(q, *self.model.jnt_range.T, out=q)
            return q
        

    async def serverExecutable(self, websocket, path):
        """
        Main behavior function for the server - currently only publishing demo data
        Function is executed when a client connects to the server.
        """
        print(">> Server listening on Port " + str(self.port))

        num_joints = 6 #len(self.joint_names)

        # TODO: Implement differentation when visualisation should be opened for debugging and when it 
        # should run in the backend to save computational power
        #simViewer = mujoco.viewer.launch_passive(
        #    model=self.model,
        #    data=self.data,
        #    show_left_ui=False,
        #    show_right_ui=False,
        #)
        #with simViewer as viewer:

        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        
        # Reset the free camera.
        #mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
        # Enable site frame visualization.
        #viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        #while viewer.is_running():

        self.websocket = websocket # quick fix to send q to frontend with mtx
        websocket_open = True
        # Simulation Loop
        while websocket_open:
            step_start = time.time()

            if self.for_mtx:
                mtx_.main(self)

            #self.handle_control_from_mtx(self,q_from_mtx)

            if not self.for_mtx:
                q = self.build_q_from_mjc(num_joints)

                # Set the control signal and step the simulation.
                self.data.ctrl[self.actuator_ids] = q[self.dof_ids]
                mujoco.mj_step(self.model, self.data)

                #viewer.sync() # used for local visualisation
                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    await asyncio.sleep(time_until_next_step)

                # Transform joint postions array to publish via websocket and catch disconnection errors
                mapped_joint_names = ["Base", "Link1", "Link2", "Link3", "Link4", "Link5", "Link6"]
                q_string = create_output_string(mapped_joint_names, "", q)

                print(q_string)
                try:
                    await self.websocket.send(q_string)
                    # print(q_string)

                except websockets.exceptions.ConnectionClosedOK:
                    print("Connection closed - OK")
                    websocket_open = False
                    await self.websocket.close()
                    break

                except websockets.exceptions.ConnectionClosedError:
                    print("Connection closed - Error")
                    websocket_open = False
                    await self.websocket.close()
                    break

                except Exception as ex:
                    websocket_open = False
                    print(f"{type(ex)} : {ex}")
                    await self.websocket.close()
                    break

    def handle_control_from_mtx(self,q_from_mtx):
        print(f"q_from_mtx: {q_from_mtx}")
        #self.data.ctrl[self.actuator_ids] = q_from_mtx[0][self.dof_ids]
        mujoco.mj_step(self.model, self.data)

        q_string = create_output_string(self.joint_names, self.joint_name_prefix, q_from_mtx[0])  
        print(q_string)  
        websocket_open = True
        try:
            self.websocket.send(q_string)
            # print(q_string)

        except websockets.exceptions.ConnectionClosedOK:
            print("Connection closed - OK")
            websocket_open = False
            self.websocket.close()

        except websockets.exceptions.ConnectionClosedError:
            print("Connection closed - Error")
            websocket_open = False
            self.websocket.close()

        except Exception as ex:
            websocket_open = False
            print(f"{type(ex)} : {ex}")
            self.websocket.close()
            
        if not websocket_open :
            pass # should raise an exception to stop the websocket 



def get_args():
    parser = argparse.ArgumentParser(description="Matrix option parser")
    parser.add_argument('--for-mtx', action='store_true', help='Set this flag for matrix option')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    server = MuJocoSimulation(for_mtx = args.for_mtx)
    server.runServer()