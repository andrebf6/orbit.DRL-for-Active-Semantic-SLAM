import argparse
from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Launch a quadrotor base environment.")
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import other libararies

import math
import torch

import omni.isaac.orbit.envs.mdp as mdp
from omni.isaac.orbit.envs import BaseEnv, BaseEnvCfg
from omni.isaac.orbit.managers import EventTermCfg as EventTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.scene import InteractiveSceneCfg

# Depends on UAV
from source.drone_models.crazyflie import get_crazyflie_config 
# from source.drone_models.quadrotor import get_quadrotor_config

@configclass
class QuadrotorSceneCfg(InteractiveSceneCfg):
    """Configuration for a drone scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # drone
    robot: ArticulationCfg =  get_crazyflie_config().replace(prim_path="{ENV_REGEX_NS}/Robot")
    # robot: ArticulationCfg = get_quadrotor_config().replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )



@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    # joint_names depend on UAV
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["m1_joint","m2_joint","m3_joint","m4_joint"], scale={"m1_joint":-1.0,"m2_joint":1.0,"m3_joint":-1.0, "m4_joint":1.0})


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class DroneEnvCfg(BaseEnvCfg):
    """Configuration for the drone environment."""

    # Scene settings
    scene = QuadrotorSceneCfg(num_envs=10, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [2.0, 0.0, 2.5]
        self.viewer.lookat = [-0.5, 0.0, 0.5]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


def main():
    """Main function."""
   
    # Set environment configuration
    env_cfg = DroneEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # setup base environment
    env = BaseEnv(cfg=env_cfg)
    print(env.scene.articulations["robot"].joint_names)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 500 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            # sample random actions
            effort_target = torch.zeros_like(env.action_manager.action)
            # NOTE: action dimension = # envs x sum action terms dimensions -> (action_term_dim: # joints affected by the action term)
            # EX: (3, 4 ) - 1 action term that affects 4 joints)
            
            mod = 9.81*0.025/4
            effort_target[0, :] = mod 
            
            mod = 9.81*0.05/4
            effort_target[1, :] = mod 

            mod = 9.81*0.1/4
            effort_target[2, :] = mod  

            if count % 500 == 0:
                print('Effort commands: ',effort_target) 


            # step the environment
            obs, _ = env.step(effort_target)

            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
