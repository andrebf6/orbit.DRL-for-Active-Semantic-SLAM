import argparse
from omni.isaac.orbit.app import AppLauncher

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Launch a drone in an environment using InteractiveScene")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import other libararies
import torch

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.utils import configclass

from scripts.quadrotor import get_quadrotor_config


@configclass
class QuadrotorSceneCfg(InteractiveSceneCfg):
    """Configuration for a drone scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    quadrotor: ArticulationCfg = get_quadrotor_config().replace(prim_path="{ENV_REGEX_NS}/Robot")




def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["quadrotor"]
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Simulation loop
    while simulation_app.is_running():
        
        # Reset
        if count % 500 == 0:
            count = 0

            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)

            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            # joint_pos += torch.rand_like(joint_pos) * 0.001
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

        # Apply action
        # efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        efforts = torch.zeros_like(robot.data.joint_pos)
        efforts[0, 0] = 2.0
        efforts[0, 1] = 5.0
        efforts[0, 2] = 2.0
        efforts[0, 3] = 5.0
        efforts[0, 4] = 2.0
        efforts[0, 5] = 5.0
        efforts[0, 6] = 2.0
        efforts[0, 7] = 5.0

        efforts[1, 0] = 10.0
        efforts[1, 1] = 5.0
        efforts[1, 2] = 10.0
        efforts[1, 3] = 5.0
        efforts[1, 4] = 10.0
        efforts[1, 5] = 5.0
        efforts[1, 6] = 10.0
        efforts[1, 7] = 5.0
        
        if count % 200 == 0:
            print("Effot:", efforts)
            print("joint_vel:", robot.data.joint_vel)

        robot.set_joint_effort_target(efforts)
        scene.write_data_to_sim()

        # Perform step
        sim.step()

        count += 1
        robot.update(sim_dt)

def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False)
    sim = SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    print("[INFO]: Simulation settings done")

    # Design scene
    scene_cfg = QuadrotorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    print("[INFO]: Scene set up")

    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()