import argparse
from omni.isaac.orbit.app import AppLauncher

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Launch a drone in an environment using an Articulation Asset")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import oher libararies
import torch
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.sim import SimulationContext

from scripts.quadrotor import get_quadrotor_config


def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    
    # Ground
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robot
    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    quadrotor_cfg = get_quadrotor_config().copy()
    quadrotor_cfg.prim_path = "/World/Origin.*/Robot"
    quadrotor = Articulation(cfg=quadrotor_cfg)

    scene_entities = {"quadrotor": quadrotor}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = entities["quadrotor"]
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Simulation loop
    while simulation_app.is_running():
        
        # Reset
        if count % 500 == 0:
            count = 0

            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_state_to_sim(root_state)

            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            # joint_pos += torch.rand_like(joint_pos) * 0.001
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # clear internal buffers
            robot.reset()
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
        robot.write_data_to_sim()

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
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    print("[INFO]: Scene set up")

    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()