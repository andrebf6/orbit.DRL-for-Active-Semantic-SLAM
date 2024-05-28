
"""Configuration for a simple Quadrotor robot."""


import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.assets import ArticulationCfg

from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

def get_quadrotor_config():
    QUADROTOR_CONFIG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Quadcopter/quadcopter.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        # init_state=ArticulationCfg.InitialStateCfg(
        #     pos=(0.0, 0.0, 2.0), joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0}
        # ),
        actuators={
            "body_arm_0": ImplicitActuatorCfg(
                joint_names_expr=["rotor_arm0"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
            "arm_rotor_0": ImplicitActuatorCfg(
                joint_names_expr=["rotor0"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
            "body_arm_1": ImplicitActuatorCfg(
                joint_names_expr=["rotor_arm1"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
            "arm_rotor_1": ImplicitActuatorCfg(
                joint_names_expr=["rotor1"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
            "body_arm_2": ImplicitActuatorCfg(
                joint_names_expr=["rotor_arm2"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
            "arm_rotor_2": ImplicitActuatorCfg(
                joint_names_expr=["rotor2"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
            "body_arm_3": ImplicitActuatorCfg(
                joint_names_expr=["rotor_arm3"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
            "arm_rotor_3": ImplicitActuatorCfg(
                joint_names_expr=["rotor3"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
        },
    )
    return QUADROTOR_CONFIG
