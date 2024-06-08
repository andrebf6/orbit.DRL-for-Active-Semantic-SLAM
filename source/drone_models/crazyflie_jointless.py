
"""Configuration for a simple Crazyfly robot."""


import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg

def get_crazyflie_config():
    CRAZIFLIE_CONFIG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="/orbit.DRL-for-Active-Semantic-SLAM/source/drone_models/crazyflie_jointless.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                enabled_self_collisions=True,
                solver_position_iteration_count=6,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), 
        ),
        actuators={
        },
    )
    return CRAZIFLIE_CONFIG
