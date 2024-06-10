from omni.isaac.orbit.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.orbit.utils import configclass

from . import body_actions

@configclass
class BodyWrenchActionCfg(ActionTermCfg):
    """Configuration for the body wrench action term.

    See :class:`BodyWrenchAction` for more details.
    """

    class_type: type[ActionTerm] = body_actions.BodyWrenchAction

@configclass
class RotorVelocitiesActionCfg(ActionTermCfg):
    """Configuration for the body wrench action term.

    See :class:`BodyWrenchAction` for more details.
    """

    class_type: type[ActionTerm] = body_actions.RotorVelocitiesAction

