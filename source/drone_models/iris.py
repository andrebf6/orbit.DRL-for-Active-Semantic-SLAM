
"""Configuration for a simple Iris robot."""


import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.assets import ArticulationCfg

def get_iris_config():
    IRIS_CONFIG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/orbit.DRL-for-Active-Semantic-SLAM/source/drone_models/iris.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
                enable_gyroscopic_forces=True,
                disable_gravity =False
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                enabled_self_collisions=True,
                solver_position_iteration_count=6,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
                fix_root_link=False
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), joint_pos={"joint0": 0.0, "joint1": 0.0, "joint2": 0.0, "joint3": 0.0}
        ),
        actuators={
            "body_prop_1": ImplicitActuatorCfg(
                joint_names_expr=["joint0"],
                effort_limit=2000.0,
                velocity_limit=1000.0,
                stiffness=0.0,
                damping=100.0,
            ),
            "body_prop_2": ImplicitActuatorCfg(
                joint_names_expr=["joint1"],
                effort_limit=2000.0,
                velocity_limit=1000.0,
                stiffness=0.0,
                damping=100.0,
            ),
            "body_prop_3": ImplicitActuatorCfg(
                joint_names_expr=["joint2"],
                effort_limit=2000.0,
                velocity_limit=1000.0,
                stiffness=0.0,
                damping=100.0,
            ),
            "body_prop_4": ImplicitActuatorCfg(
                joint_names_expr=["joint3"],
                effort_limit=2000.0,
                velocity_limit=1000.0,
                stiffness=0.0,
                damping=100.0,
            ),
        },
    )
    return IRIS_CONFIG
