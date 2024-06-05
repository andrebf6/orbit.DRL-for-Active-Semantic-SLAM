import argparse
from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Launch a quadrotor base environment.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
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
# from omni.isaac.orbit.managers import EventTermCfg as EventTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.scene import InteractiveSceneCfg

from source.drone_models.crazyflie import get_crazyflie_config # Depends on UAV
from source.high_level_controller.quadrotor_controller import NonlinearController

@configclass
class QuadrotorSceneCfg(InteractiveSceneCfg):
    """Configuration for a drone scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # drone
    robot: ArticulationCfg = get_crazyflie_config().replace(prim_path="{ENV_REGEX_NS}/Robot")
    
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
    m1_joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["m1_joint"], scale=1)
    m2_joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["m2_joint"], scale=-1)
    m3_joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["m3_joint"], scale=1)
    m4_joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["m4_joint"], scale=-1)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_positions = ObsTerm(func=mdp.joint_pos)
        joint_velocities = ObsTerm(func=mdp.joint_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class DroneEnvCfg(BaseEnvCfg):
    """Configuration for the drone environment."""

    # Scene settings
    scene = QuadrotorSceneCfg(num_envs=5, env_spacing=2.5)
    
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


def main():
    """Main function."""
   
    # Set environment configuration
    env_cfg = DroneEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # setup base environment
    env = BaseEnv(cfg=env_cfg)

    # Setup Hig-level controller
    controller = NonlinearController(
            num_envs = env_cfg.scene.num_envs,
            trajectory_file="/orbit.DRL-for-Active-Semantic-SLAM/trajectories/pitch_relay_90_deg_2.csv",
            results_file="/orbit.DRL-for-Active-Semantic-SLAM/results/single_statistics.npz",
            Ki=[0.5, 0.5, 0.5],
            Kr=[2.0, 2.0, 2.0]
        )
    print('Controller set up!')
    
    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1000 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            # High-level controller
            # NOTE: Now the setpoint is given from a csv file. Later on the setpoint will be send by the RL algorithm.
            
            # Access robot state
            robot = env.scene["robot"]
            print("Robot root state: ", robot.data.root_state_w)

            # Apply high-level controller
            controller.update_state(state=robot.data.root_state_w)
            rotor_vel = controller.update(dt=env_cfg.sim.dt)
            print("Actions: ", rotor_vel)
            # NOTE: Output action dimension = # envs x (sum action terms dimensions)
                      
            # step the environment
            obs, _ = env.step(rotor_vel.to(torch.device('cuda:0')))

            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()