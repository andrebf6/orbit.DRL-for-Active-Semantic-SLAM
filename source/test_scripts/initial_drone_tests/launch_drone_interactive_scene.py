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

# Depends on UAV
from source.drone_models.crazyflie import get_crazyflie_config 
# from source.drone_models.quadrotor import get_quadrotor_config 


@configclass
class QuadrotorSceneCfg(InteractiveSceneCfg):
    """Configuration for a drone scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation (depends on UAV)
    drone: ArticulationCfg = get_crazyflie_config().replace(prim_path="{ENV_REGEX_NS}/Robot")
    # drone: ArticulationCfg = get_quadrotor_config().replace(prim_path="{ENV_REGEX_NS}/Robot")



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["drone"]
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Simulation loop
    while simulation_app.is_running():
        
        # Reset
        if count % 500 == 0:
            count = 0

            # ROOT STATE
            print("Drone root state:", robot.data.root_state_w)
            # """Default root state ``[pos, quat(w,x,y,z), lin_vel, ang_vel]`` in local environment frame. Shape is (num_instances, 13).
            
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            
            # JOINT STATE
            print("Drone joint position state:", robot.data.joint_pos)
            # """Default joint positions of all joints. Shape is (num_instances, num_joints).""" 
               
            print("Drone joint velocity state:", robot.data.joint_vel)
            # """Default joint velocities of all joints. Shape is (num_instances, num_joints)."""

            # set joint positions (with some noise)
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            # joint_pos += torch.rand_like(joint_pos) * 0.001
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

        # Apply action

        # APPLY EFFORT
        if count % 400 == 0:
            efforts = torch.zeros_like(robot.data.joint_pos)
            mod = 9.81*0.025/4  # mod = 3 
            efforts[0, 0] = mod
            efforts[0, 1] = -mod
            efforts[0, 2] = mod
            efforts[0, 3] = -mod
            # efforts[0, 1] = mod
            # efforts[0, 3] = -mod
            # efforts[0, 5] = mod
            # efforts[0, 7] = -mod
            
            mod = 9.81*0.025/4 #5
            efforts[1, 0] = mod
            efforts[1, 1] = mod
            efforts[1, 2] = mod
            efforts[1, 3] = mod
            # efforts[1, 1] = mod
            # efforts[1, 3] = -mod
            # efforts[1, 5] = mod
            # efforts[1, 7] = -mod
        
            print("Effort commands:", efforts)
            print("joint_vel:", robot.data.joint_vel)

            robot.set_joint_effort_target(efforts)
            scene.write_data_to_sim()

        # APPLY VELOCITY
        # vel = torch.zeros_like(robot.data.joint_vel)
        # mod = 100
        # vel[0, 0] = mod
        # vel[0, 1] = -mod
        # vel[0, 2] = mod
        # vel[0, 3] = -mod

        # mod = 1000
        # vel[1, 0] = mod
        # vel[1, 1] = -mod
        # vel[1, 2] = mod
        # vel[1, 3] = -mod

        # if count % 200 == 0:
        #     print("Velocity commands:", vel)
        #     print("joint_vel:", robot.data.joint_vel)

        # robot.set_joint_velocity_target(vel)
        # scene.write_data_to_sim()

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