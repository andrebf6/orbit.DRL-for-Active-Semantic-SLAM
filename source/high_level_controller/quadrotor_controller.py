#!/usr/bin/env python
"""
| File: quadrotor_controller.py
| Author: Andrea Bravo  Forn
   NOTE: Code adapted from Marcelo Jacinto and Joao Pinto (creators of Pegasus Simulator)
"""

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from omni.isaac.dynamic_control import _dynamic_control
dc = _dynamic_control.acquire_dynamic_control_interface()

class ActionBroadcaster:
    def __init__(self):
        
        N=100

        self.actions = np.zeros((4*N, 2))
        self.actions[0:N,0] = 1
        self.actions[N:2*N,1] = -1
        self.actions[2*N:3*N,0] = -1
        self.actions[3*N:4*N,1] = 1

        self.idx = -1
    
    def get_action(self):
        self.idx+=1
        if self.idx == self.actions.shape[0]:
            self.idx=0
        return self.actions[self.idx, :]

class NonlinearController():
    """A nonlinear controller class. It implements a nonlinear controller that allows a drone to track
    aggressive trajectories. 
    """

    def __init__(self,  
        num_envs: int = 1,
        Kp=None,
        Kd=None,
        Ki=None,
        Kr=None,
        Kw=None):

        # Number of spawned environments
        self.num_envs = num_envs 

        # The current rotor references [rad/s]
        self.input_ref = torch.zeros((self.num_envs, 4))

        # The drone state in each environment expressed in the inertial frame (in ENU)
        self.p = np.zeros((self.num_envs, 3))                   # The vehicle position
        identity_quat = np.tile([0, 0, 0, 1], (self.num_envs, 1))
        self.R = Rotation.from_quat(identity_quat)              # The vehicle attitude                 
        self.w = np.zeros((self.num_envs, 3))                   # The angular velocity of the vehicle
        self.v = np.zeros((self.num_envs, 3))                   # The linear velocity of the vehicle in the inertial frame
        self.a = np.zeros((self.num_envs, 3))                   # The linear acceleration of the vehicle in the inertial frame

        # Define the control gains matrix for the outer-loop
        self.Kp = np.diag(Kp)
        self.Kd = np.diag(Kd)
        self.Ki = np.diag(Ki)
        self.Kr = np.diag(Kr)
        self.Kw = np.diag(Kw)

        self.int = np.array([0.0, 0.0, 0.0])   
   
        # Define the dynamic parameters for the vehicle
        self.m = 1.50               # Mass in Kg
        self.g = 9.81               # The gravity acceleration ms^-2
        self.max_rotor_vel = 1100
        
        # Define the alocation matrix
        cf = 8.54858e-6        # Thurst coefficient
        ct = 1e-6              # Drag coefficient
        rot_dir = np.array([-1, -1, 1, 1])
        relative_poses = np.array([[0.13759538531303406, -0.20673562586307526, 0.023], [-0.125,  0.21869462728500366, 0.0230001], [0.13830871880054474, 0.20321962237358093, 0.023], [-0.12450209259986877, -0.22199904918670654, 0.023]])
        
        rb =dc.get_rigid_body("World/envs/env0/Robot" + "/body")
        rotors = [dc.get_rigid_body("env0/Robot" + "/rotor" + str(i)) for i in range(4)]
        relative_poses_test = dc.get_relative_body_poses(rb, rotors)
        print(relative_poses_test)
        
        aloc_matrix = np.zeros((4, 4))
        aloc_matrix[0, :] = np.array(cf)
        aloc_matrix[1, :] = np.array([relative_poses[i,1] * cf for i in range(4)])
        aloc_matrix[2, :] = np.array([-relative_poses[i,0] * cf for i in range(4)])
        aloc_matrix[3, :] = np.array([ct * rot_dir[i] for i in range(4)])

        self.aloc_inv = np.linalg.pinv(aloc_matrix)
        print("allocation matrix: ", aloc_matrix)
              
        # Auxiliar variable, so that we only start sending motor commands once we get the state of the vehicle
        self.received_first_state = False

        self.RL_algorithm = ActionBroadcaster()
        self.velocity = 2

   
    def update_state(self, state: torch.Tensor):
        """
        Method that updates the current state of the drones.

        Args:
            state (torch.Tensor): [pos, quat, lin_vel, ang_vel]) in the simulation world frame. Shape is (env_num, 13).
        """
        if self.num_envs != state.size(0):
            raise ValueError(f"Unexpected state row number: rows in state {state.size(0)}, env_num: {self.num_envs}.")
        
        orientation_quat = np.zeros((self.num_envs, 4))
        for i in range(self.num_envs):
            self.p[i,:] = state[i,:3].cpu().numpy()
            orientation_quat[i,:] = state[i,3:7].cpu().numpy()
            self.v[i,:] = state[i, 7:10].cpu().numpy()
            self.w[i,:] = state[i, 10:13].cpu().numpy()
            print("Robot position: ", self.p[i,:], "linear velocity: ", self.v[i,:], "angular velocity: ", self.w[i,:])
        
        orientation_quat = orientation_quat[:, [1, 2, 3, 0]]
        self.R = Rotation.from_quat(orientation_quat)
            
        for i in range(self.num_envs):
            print("Robot rotation: ", self.R[i].as_matrix())
        
        self.received_first_state = True


    def update(self, dt: float):
        """Method that implements the nonlinear control law and updates the target angular velocities for each rotor. 
        This method will be called by the simulation on every physics step

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """
        
        if self.received_first_state == False:
            return

        # -------------------------------------------------
        # Update the references for the controller to track
        # -------------------------------------------------
        # p_ref = np.array([0,0,1])
        # v_ref = np.array([0,0,0])
        # a_ref = np.array([0,0,0])
        # j_ref = np.array([0,0,0])
        # yaw_ref = 0
        # yaw_rate_ref = 0

        action =  self.RL_algorithm.get_action()
        v_ref = np.zeros(3)
        v_ref[0:2] = action * self.velocity
        p_ref = self.p[0,:] + v_ref*dt
        print(v_ref)
        print(self.p)
        print(p_ref)
        p_ref[2] = 1
        # v_ref = np.array([0,0,0])
        a_ref = np.array([0,0,0])
        j_ref = np.array([0,0,0])
        yaw_ref = 0                 # Make drone face the direction it is going!!
        yaw_rate_ref = 0
  
        # -------------------------------------------------
        # Start the controller implementation
        # -------------------------------------------------

        for i in range(self.num_envs):

            # Compute the tracking errors
            ep = self.p[i,:] - p_ref
            ev = self.v[i,:] - v_ref
            self.int = self.int +  (ep * dt)
            ei = self.int

            # Compute F_des term
            F_des = -(self.Kp @ ep) - (self.Kd @ ev) - (self.Ki @ ei) + np.array([0.0, 0.0, self.m * self.g]) + (self.m * a_ref)

            print("prop f: ", self.Kp @ ep)
            print("der f: ", self.Kd @ ev)
            print("int f: ", self.Ki @ ei)
            print("Weight term: ", np.array([0.0, 0.0, self.m * self.g]))
            print("Acceleration term: ", self.m * a_ref)
            print("Desired force: ", F_des)

            # Get the current axis Z_B (given by the last column of the rotation matrix)
            Z_B = self.R[i].as_matrix()[:,2]
            
            # Get the desired total thrust in Z_B direction (u_1)
            u_1 = F_des @ Z_B
            print("Force on zb axis: ", u_1 )

            # Compute the desired body-frame axis Z_b
            Z_b_des = F_des / np.linalg.norm(F_des)

            # Compute X_C_des 
            X_c_des = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])

            # Compute Y_b_des
            Z_b_cross_X_c = np.cross(Z_b_des, X_c_des)
            Y_b_des = Z_b_cross_X_c / np.linalg.norm(Z_b_cross_X_c)

            # Compute X_b_des
            X_b_des = np.cross(Y_b_des, Z_b_des)

            # Compute the desired rotation R_des = [X_b_des | Y_b_des | Z_b_des]
            R_des = np.c_[X_b_des, Y_b_des, Z_b_des]
            R = self.R[i].as_matrix()

            # Compute the rotation error
            e_R = 0.5 * self.vee((R_des.T @ R) - (R.T @ R_des))

            # Compute an approximation of the current vehicle acceleration in the inertial frame (since we cannot measure it directly)
            self.a = (u_1 * Z_B) / self.m - np.array([0.0, 0.0, self.g])

            # Compute the desired angular velocity by projecting the angular velocity in the Xb-Yb plane
            # projection of angular velocity on xB − yB plane
            # see eqn (7) from [2].
            hw = (self.m / u_1) * (j_ref - np.dot(Z_b_des, j_ref) * Z_b_des) 
            
            # desired angular velocity
            w_des = np.array([-np.dot(hw, Y_b_des), 
                            np.dot(hw, X_b_des), 
                            yaw_rate_ref * Z_b_des[2]])
            print("Desired angular velocity: ", w_des)

            # Compute the angular velocity error
            e_w = self.w[i,:] - w_des

            # Compute the torques to apply on the rigid body
            tau = -(self.Kr @ e_R) - (self.Kw @ e_w)
            print("Kr tau: ", self.Kr @ e_R)
            print("Kw tau: ", self.Kw @ e_w)
            print("Torque: ", tau )

            # Convert the desired force and torque to angular velocity [rad/s] references to give to each rotor.
            self.input_ref[i,:] = self.force_and_torques_to_velocities(u_1, tau, self.aloc_inv, self.max_rotor_vel)
                   
        return self.input_ref

    @staticmethod
    def vee(S):
        """Auxiliary function that computes the 'v' map which takes elements from so(3) to R^3.

        Args:
            S (np.array): A matrix in so(3)
        """
        return np.array([-S[1,2], S[0,2], -S[0,1]])
    
    @staticmethod
    def force_and_torques_to_velocities(force: float, torque: np.ndarray, aloc_inv: np.ndarray, max_vel: float):
        """
        Auxiliar method used to get the target angular velocities for each rotor, given the total desired thrust [N] and
        torque [Nm] to be applied in the multirotor's body frame.

        Note: This method assumes a quadratic thrust curve. This method will be improved in a future update,
        and a general thrust allocation scheme will be adopted. For now, it is made to work with multirotors directly.

        Args:
            force (np.ndarray): A vector of the force to be applied in the body frame of the vehicle [N]
            torque (np.ndarray): A vector of the torque to be applied in the body frame of the vehicle [Nm]

        Returns:
            list: A list of angular velocities [rad/s] to apply in reach rotor to accomplish suchs forces and torques
        """

        # Compute the target angular velocities (squared)
        squared_ang_vel = aloc_inv @ np.array([force, torque[0], torque[1], torque[2]])
        print("squared rotor velocities : ", squared_ang_vel)
        
        # Making sure that there is no negative value on the target squared angular velocities
        squared_ang_vel[squared_ang_vel < 0] = 0.0

        # ------------------------------------------------------------------------------------------------
        # Saturate the inputs while preserving their relation to each other, by performing a normalization
        # ------------------------------------------------------------------------------------------------
        max_thrust_vel_squared = np.power(max_vel, 2)
        max_val = np.max(squared_ang_vel)

        if max_val >= max_thrust_vel_squared:
            normalize = np.maximum(max_val / max_thrust_vel_squared, 1.0)

            squared_ang_vel = squared_ang_vel / normalize

        # Compute the angular velocities for each rotor in [rad/s]
        ang_vel = np.sqrt(squared_ang_vel)
        print("rotor velocities : ", ang_vel)

        return torch.tensor(ang_vel)