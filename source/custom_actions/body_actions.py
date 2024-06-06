import torch
from typing import TYPE_CHECKING

import omni.isaac.orbit.utils.string as string_utils
from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.managers.action_manager import ActionTerm
from omni.isaac.orbit.envs import BaseEnv
from . import body_actions_cfg


class BodyWrenchAction(ActionTerm):
    """Joint action term that applies the processed actions to the articulation's joints as velocity commands."""

    def __init__(self, cfg: body_actions_cfg.BodyWrenchActionCfg, env: BaseEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # create tensors for raw and processed actions (???)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        
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
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions
        print("Post-processed Term action:", self._processed_actions)

        # NOTE: We don't apply scale/offset before applying the actions 

    def apply_actions(self):
        # set external wrench
        self._asset.set_external_force_and_torque(forces=self.processed_actions[:,0:3], torques=torch.self.processed_actions[:,3:6])
        print('External Wrench target: ', self.processed_actions )