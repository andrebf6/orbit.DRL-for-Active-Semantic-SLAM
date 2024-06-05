#!/usr/bin/env python
"""
| File: quadrotor_controller.py
| Author: Andrea Bravo  Forn
   NOTE: Code adapted from Marcelo Jacinto and Joao Pinto (creators of Pegasus Simulator)
"""

import numpy as np
from scipy.spatial.transform import Rotation
import torch

# Instantiation example:
# [NonlinearController(
#             env_num = self.num_envs,
#             trajectory_file=self.curr_dir + "/trajectories/pitch_relay_90_deg_2.csv",
#             results_file=self.curr_dir + "/results/single_statistics.npz",
#             Ki=[0.5, 0.5, 0.5],
#             Kr=[2.0, 2.0, 2.0]
#         )]

class NonlinearController():
    """A nonlinear controller class. It implements a nonlinear controller that allows a drone to track
    aggressive trajectories. 
    """

    def __init__(self,  
        num_envs: int = 1,
        # setpoint: np.zeros((4,)),         # Action passed by the RL algorithm
        trajectory_file: str = None,        # Remove afterwards
        results_file: str = None,           # Remove afterwards (or create a switch to analyse mode)
        Kp=[10.0, 10.0, 10.0],
        Kd=[8.5, 8.5, 8.5],
        Ki=[1.50, 1.50, 1.50],
        Kr=[3.5, 3.5, 3.5],
        Kw=[0.5, 0.5, 0.5]):

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
        self.m = 0.0282             # Mass in Kg
        self.g = 9.81               # The gravity acceleration ms^-2

        # Define the alocation matrix
        self.cf = 5.84e-6           # Thurst coefficient
        self.ct = 1e-6              # Drag coefficient
        self.L = 0.043841           # Arm length
        
        aloc_matrix = np.zeros((4, 4))
        aloc_matrix[0, :] = np.array(self.cf)
        row_2 = np.array([0, self.cf*self.L, 0, -self.cf*self.L])                                           
        aloc_matrix[1, :] = row_2 
        row_3 = np.array([-self.cf*self.L, 0, self.cf*self.L, 0])
        aloc_matrix[2, :] = row_3
        row_4 = np.array([self.ct, -self.ct, self.ct, -self.ct])
        aloc_matrix[3, :] =  row_4

        self.aloc_inv = np.linalg.pinv(aloc_matrix)
        # print("allocation matrix inverted : ", self.aloc_inv)

        self.max_rotor_vel = 2000
              
        # Read the target trajectory from a CSV file inside the trajectories directory
        # Remove afterwards
        self.trajectory = self.read_trajectory_from_csv(trajectory_file)
        self.index = 0
        self.max_index, _ = self.trajectory.shape
        self.total_time = 0.0

        # Auxiliar variable, so that we only start sending motor commands once we get the state of the vehicle
        self.received_first_state = False

        # Lists used for analysing performance statistics
        # Remove afterwards (or create a switch to analyse mode)
        self.results_files = results_file
        self.time_vector = []
        self.desired_position_over_time = []
        self.position_over_time = []
        self.position_error_over_time = []
        self.velocity_error_over_time = []
        self.atittude_error_over_time = []
        self.attitude_rate_error_over_time = []

    def read_trajectory_from_csv(self, file_name: str):
        """Auxiliar method used to read the desired trajectory from a CSV file

        Args:
            file_name (str): A string with the name of the trajectory inside the trajectories directory

        Returns:
            np.ndarray: A numpy matrix with the trajectory desired states over time
        """
        # Remove afterwards
        # Read the trajectory to a pandas frame
        return np.flip(np.genfromtxt(file_name, delimiter=','), axis=0)


    def update_state(self, state: torch.Tensor):
        """
        Method that updates the current state of the drones.

        Args:
            state (torch.Tensor): [pos, quat, lin_vel, ang_vel]`` in the simulation world frame. Shape is (env_num, 13).
        """
        if self.num_envs != state.size(0):
            raise ValueError(f"Unexpected state row number: rows in state {state.size(0)}, env_num: {self.num_envs}.")
        
        orientation_quat = np.zeros((self.num_envs, 4))
        for i in range(self.num_envs):
            self.p[i,:] = state[i,:3].cpu().numpy()
            orientation_quat[i,:] = state[i,3:7].cpu().numpy()
            self.v[i,:] = state[i, 7:10].cpu().numpy()
            self.w[i,:] = state[i, 10:13].cpu().numpy()
            # print("Robot position: ", self.p[i,:], "linear velocity: ", self.v[i,:], "angular velocity: ", self.w[i,:])
        
        orientation_quat[:, [0, 3]] = orientation_quat[:, [3, 0]]
        self.R = Rotation.from_quat(orientation_quat)
            
        # for i in range(self.num_envs):
        #     print("Robot rotation: ", self.R[i].as_matrix())
        
        self.received_first_state = True

    def input_reference(self):
        """
        Method that is used to return the latest target angular velocities to be applied to the vehicle

        Returns:
            A list with the target angular velocities for each individual rotor of the vehicle
        """
        return self.input_ref

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
        self.total_time += dt
        
        # Check if we need to update to the next trajectory index
        # Remove afterwards
        if self.index < self.max_index - 1 and self.total_time >= self.trajectory[self.index + 1, 0]:
            self.index += 1

        # Update using an external trajectory
        # Remove afterwards
        # the target positions [m], velocity [m/s], accelerations [m/s^2], jerk [m/s^3], yaw-angle [rad], yaw-rate [rad/s]
        p_ref = np.array([self.trajectory[self.index, 1], self.trajectory[self.index, 2], self.trajectory[self.index, 3]])
        v_ref = np.array([self.trajectory[self.index, 4], self.trajectory[self.index, 5], self.trajectory[self.index, 6]])
        a_ref = np.array([self.trajectory[self.index, 7], self.trajectory[self.index, 8], self.trajectory[self.index, 9]])
        j_ref = np.array([self.trajectory[self.index, 10], self.trajectory[self.index, 11], self.trajectory[self.index, 12]])
        yaw_ref = self.trajectory[self.index, 13]
        yaw_rate_ref = self.trajectory[self.index, 14]

        # print("References updated as: position ", p_ref, "Linear velocity",v_ref , "Jerk", j_ref , "Acceleration", a_ref ,"Yaw", yaw_ref, "Yaw_ref", yaw_rate_ref)
            
        # -------------------------------------------------
        # Start the controller implementation
        # -------------------------------------------------

        for i in range(self.num_envs):

            # Compute the tracking errors
            ep = self.p[i,:] - p_ref
            ev = self.v[i,:] - v_ref
            self.int = self.int +  (ep * dt)
            ei = self.int

            print("Robot position in world frame:", self.p[i,:])
            print("Desired position in world frame:", p_ref)

            # print("Postion error:", ep)
            # print("Velocity error:", ev)
            # print("Integral error:", ei)

            # Compute F_des term
            F_des = -(self.Kp @ ep) - (self.Kd @ ev) - (self.Ki @ ei) + np.array([0.0, 0.0, self.m * self.g]) + (self.m * a_ref)


            # Get the current axis Z_B (given by the last column of the rotation matrix)
            Z_B = self.R[i].as_matrix()[:,2]
            
            # Get the desired total thrust in Z_B direction (u_1)
            u_1 = F_des @ Z_B
            print("Force on zb axis", u_1 )

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

            # Compute the angular velocity error
            e_w = self.w[i,:] - w_des

            # Compute the torques to apply on the rigid body
            tau = -(self.Kr @ e_R) - (self.Kw @ e_w)

            # Convert the desired force and torque to angular velocity [rad/s] references to give to each rotor.
            self.input_ref[i,:] = self.force_and_torques_to_velocities(u_1, tau, self.aloc_inv, self.max_rotor_vel)
                   

        # ----------------------------
        # Statistics to save for later
        # ----------------------------
        self.time_vector.append(self.total_time)
        self.position_over_time.append(self.p)
        self.desired_position_over_time.append(p_ref)
        self.position_error_over_time.append(ep)
        self.velocity_error_over_time.append(ev)
        self.atittude_error_over_time.append(e_R)
        self.attitude_rate_error_over_time.append(e_w)

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

        print("rotor velocities : ", np.sqrt(squared_ang_vel))

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

        return torch.tensor(ang_vel)