from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.orbit.utils.string as string_utils
from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.managers.action_manager import ActionTerm
from omni.isaac.orbit.envs import BaseEnv

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv
    from . import body_actions_cfg


class BodyWrenchAction(ActionTerm):
    """Joint action term that applies the processed actions to the articulation's joints as velocity commands."""

    cfg: body_actions_cfg.BodyWrenchActionCfg

    def __init__(self, cfg: body_actions_cfg.BodyWrenchActionCfg, env: BaseEnv):
        # initialize the action term
        super().__init__(cfg, env)

        print('Asset: ', self._asset)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._asset.num_bodies, self.action_dim, device=self.device)
        
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
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
        
        self._raw_actions[:] = actions                                                    # DIMENSION: (#env, 6)
        body_num = 1
        self._processed_actions = self._raw_actions.unsqueeze(1).repeat(1, body_num, 1)   # DIMENSION: (#env, #body_parts_where_force_torque_is_applied, 6)
        
        # NOTE: TO DO
        # Parse in the BodyWrenchActionCfg the drone body parts where the torque/ force will be applied (one, the drone main body) - Options are the prims spawned with the usd. 
        # Go from name to: body_ids, total number of body parts where we apply the force (Q)
        # List of body parts: print(self._asset.body_names)
        # Look for body part id: string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)
        # self._processed_actions = self._raw_actions.unsqueeze(1).repeat(1, Q, 1)   # DIMENSION: (#env, #body_parts_where_force_torque_is_applied, 6)
        # Total number of body parts: self._asset.num_bodies

    def apply_actions(self):
        # set external wrench
        self._asset.set_external_force_and_torque(forces=self.processed_actions[:,:,0:3], torques= self.processed_actions[:,:, 3:6], body_ids=0)