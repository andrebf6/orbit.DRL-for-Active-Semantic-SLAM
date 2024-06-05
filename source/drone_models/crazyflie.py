
"""Configuration for a simple Crazyfly robot."""


import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.assets import ArticulationCfg

from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

def get_crazyflie_config():
    CRAZIFLIE_CONFIG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}//Robots/Crazyflie/cf2x.usd",
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
            pos=(0.0, 0.0, 0.0), joint_pos={"m1_joint": 0.0, "m2_joint": 0.0, "m3_joint": 0.0, "m4_joint": 0.0}
        ),
        actuators={
            "body_prop_1": ImplicitActuatorCfg(
                joint_names_expr=["m1_joint"],
                effort_limit=2000.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "body_prop_2": ImplicitActuatorCfg(
                joint_names_expr=["m2_joint"],
                effort_limit=2000.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "body_prop_3": ImplicitActuatorCfg(
                joint_names_expr=["m3_joint"],
                effort_limit=2000.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "body_prop_4": ImplicitActuatorCfg(
                joint_names_expr=["m4_joint"],
                effort_limit=2000.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    return CRAZIFLIE_CONFIG
