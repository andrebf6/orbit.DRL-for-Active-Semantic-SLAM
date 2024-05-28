import argparse
from omni.isaac.orbit.app import AppLauncher

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Launch a drone in an environment")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import oher libararies
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))
    
    # spawn a usd file of a drone into the scene
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Quadcopter/quadcopter.usd")
    cfg.func("/World/Robot", cfg, translation=(0.0, 0.0, 1.05))


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, substeps=1)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    print("[INFO]: Simulation settings done")

    # Design scene
    design_scene()
    print("[INFO]: Scene set up")

    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    main()

    simulation_app.close()
