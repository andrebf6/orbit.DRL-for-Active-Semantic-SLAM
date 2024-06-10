from __future__ import annotations

import torch
import numpy as np
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING

import omni.isaac.orbit.utils.string as string_utils
from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.managers.action_manager import ActionTerm
from omni.isaac.orbit.envs import BaseEnv

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv
    from . import body_actions_cfg


class BodyWrenchAction(ActionTerm):
    
    cfg: body_actions_cfg.BodyWrenchActionCfg

    def __init__(self, cfg: body_actions_cfg.BodyWrenchActionCfg, env: BaseEnv):
        # initialize the action term
        super().__init__(cfg, env)

        print('Asset: ', self._asset)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._asset.num_bodies, self.action_dim, device=self.device)

        # TEST ROTATION        
        # self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device) 
        # self._processed_actions = torch.zeros(self.num_envs, self._asset.num_bodies, 6, device=self.device)
        
        # TEST SPLIT FORCES 
        # self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        # self._processed_actions = torch.zeros(self.num_envs, self._asset.num_bodies, self.action_dim, device=self.device)
        # self._processed_actions_force = torch.zeros(self.num_envs, 4, self.action_dim, device=self.device)
        # self._processed_actions_torque = torch.zeros(self.num_envs, 1, self.action_dim, device=self.device)
        
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        # return 10 # TEST ROTATION
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    """
    Operations.
    """
    def process_actions(self, actions: torch.Tensor):

        self._raw_actions[:] = actions      # DIMENSION: (#env, 6)
        body_num = 1
        self._processed_actions = self._raw_actions.unsqueeze(1).repeat(1, body_num, 1)   # DIMENSION: (#env, #body_parts_where_force_torque_is_applied, 6)

        # TEST ROTATION    
        ## Add before step: action = torch.cat((wrench.to(torch.device('cuda:0')), robot.data.root_state_w[:, 3:7]), dim=1)           
        # orientation_quat = actions[:,6:].cpu().numpy()
        # orientation_quat = orientation_quat[:, [1, 2, 3, 0]]
        # R = Rotation.from_quat(orientation_quat)
        # for i in range(self.num_envs):
        #     rotated_force = actions[i,0:3].cpu().numpy()@ R[i].as_matrix()
        #     rotated_torque = actions[i,3:].cpu().numpy()@ R[i].as_matrix()

        #     self._raw_actions[i,0:3] = torch.from_numpy(rotated_force).to(torch.device('cuda:0'))
        #     self._raw_actions[i,3:] = torch.from_numpy(rotated_torque).to(torch.device('cuda:0'))
        # body_num = 1
        # self._processed_actions = self._raw_actions.unsqueeze(1).repeat(1, body_num, 1)   # DIMENSION: (#env, #body_parts_where_force_torque_is_applied, 6)

        # TEST SPLIT FORCES 
        # self._raw_actions[:] = actions      
        # self._processed_actions_force = self._raw_actions.unsqueeze(1).repeat(1, 4, 1)    # DIMENSION: (#env, #body_parts_where_force, 6)
        # self._processed_actions_torque = self._raw_actions.unsqueeze(1).repeat(1, 1, 1)   # DIMENSION: (#env, #body_parts_where_torque_is_applied, 6)
        
        
        # NOTE: TO DO
        # Parse in the BodyWrenchActionCfg the drone body parts where the torque/ force will be applied (one, the drone main body) - Options are the prims spawned with the usd. 
        # Go from name to: body_ids, total number of body parts where we apply the force (Q)
        # List of body parts: print(self._asset.body_names)
        # Look for body part id: string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)
        # self._processed_actions = self._raw_actions.unsqueeze(1).repeat(1, Q, 1)   # DIMENSION: (#env, #body_parts_where_force_torque_is_applied, 6)
        # Total number of body parts: self._asset.num_bodies

    def apply_actions(self):
        # set external wrench
        self._asset.set_external_force_and_torque(forces=self._processed_actions[:,:,0:3], torques= self._processed_actions[:,:, 3:6], body_ids=0)

        # TEST SPLIT FORCES 
        # self._asset.set_external_force_and_torque(forces = self._processed_actions_force[:,:,0:3]/4, torques= torch.zeros((self.num_envs, 4, 3)).to(torch.device('cuda:0')), body_ids=[1,2,3,4])       
        # self._asset.set_external_force_and_torque(forces =  torch.zeros(self.num_envs, 1, 3).to(torch.device('cuda:0')), torques=self._processed_actions_torque[:,:,3:6], body_ids=0)

class RotorVelocitiesAction(ActionTerm):
    
    cfg: body_actions_cfg.RotorVelocitiesActionCfg

    def __init__(self, cfg: body_actions_cfg.RotorVelocitiesActionCfg, env: BaseEnv):
        # initialize the action term
        super().__init__(cfg, env)

        print('Asset: ', self._asset)

        self.min_rotor_vel = 0
        self.max_rotor_vel = 1100
        self.c_f = 8.54858e-6
        self.c_t = 1e-6
        self.rot_dir = np.array([-1, -1, 1, 1])
        self.c_drag = np.diag([0.50, 0.30, 0.0])

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions_rotor_force = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self._processed_actions_body_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._processed_actions_body_drag_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    """
    Operations.
    """
    def process_actions(self, actions: torch.Tensor):

        self._raw_actions[:] = actions      # DIMENSION: (#env, 4)
        
        for j in range(self.num_envs):
            rolling_moment_z = 0.0
            angular_vel = [0.0 for i in range(4)]
            force_z = [0.0 for i in range(4)]

            for i in range(4):
                angular_vel[i] = np.maximum(self.min_rotor_vel, np.minimum(self._raw_actions[j,i].cpu().numpy(), self.max_rotor_vel))
                force_z [i] = self.c_f * np.power(angular_vel[i], 2)
                rolling_moment_z += self.c_t * np.power(angular_vel[i], 2.0) * self.rot_dir[i]

                print('Angular velocity', angular_vel[i],'Force in z', force_z[i], 'Torque in z', rolling_moment_z)
                
                # NOTE: Check rotor order!!!!!
                # Forces and torques are applied in the rigid body frame!
                print("Body part names", self._asset.body_names)
                print("Drone linear velocity", self._asset.data.root_vel_b [j,0:3].cpu().numpy())

                self._processed_actions_rotor_force [j,i,:] = torch.tensor([0.0, 0.0, force_z [i]])
                    
            self._processed_actions_body_torque [j,0,:] = torch.tensor([0.0, 0.0,  rolling_moment_z])
            
            drag_force = -np.dot(self.c_drag, self._asset.data.root_vel_b [j,0:3].cpu().numpy())
            print("Drag force", drag_force)

            self._processed_actions_body_drag_force [j,0,:] = torch.tensor(drag_force)

            print('Rotor forces', self._processed_actions_rotor_force)
            print('Body torque ', self._processed_actions_body_torque)
            print('Body drag', self._processed_actions_body_drag_force)
        
    def apply_actions(self):
        self._asset.set_external_force_and_torque(forces = self._processed_actions_rotor_force, torques= torch.zeros((self.num_envs, 4, 3)).to(torch.device('cuda:0')), body_ids=[1,2,3,4])       
        self._asset.set_external_force_and_torque(forces = self._processed_actions_body_drag_force, torques= torch.zeros((self.num_envs, 1, 3)).to(torch.device('cuda:0')), body_ids=0)       
        self._asset.set_external_force_and_torque(forces =  torch.zeros(self.num_envs, 1, 3).to(torch.device('cuda:0')), torques=self._processed_actions_body_torque, body_ids=0)
