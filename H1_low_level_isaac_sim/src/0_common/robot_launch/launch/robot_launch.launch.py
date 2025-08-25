import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction, ExecuteProcess, RegisterEventHandler
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # --- Robot Discription ---
    robot_dir = get_package_share_directory('h1_description')
    launch_dir = os.path.join(os.getcwd(), 'src', '0_common', 'robot_launch')
    urdf_file = os.path.join(robot_dir, 'urdf', 'h1_sim.urdf')
    foxglove_params = os.path.join(launch_dir, 'config', 'foxglove_bridge_params.yaml')

    with open(urdf_file, 'r') as file:
        robot_description = {'robot_description': file.read()}

    robot_controllers_yaml = os.path.join(launch_dir, 'config', 'h1_controllers.yaml')


    # -- Debuging ---
    debug_logger = LaunchConfiguration("debug_pkg")
    declare_debug_logger_arg = DeclareLaunchArgument(
        "debug_pkg",
        default_value=["none"],
        description="Debug Logging level in the format: '<package1> <package2> ...'. Example: 'mpc wbc'",
    )
    debug_log_packages_expr = PythonExpression(["'", debug_logger, "'.split()"])

    use_foxglove = LaunchConfiguration('use_foxglove')
    declare_use_foxglove_arg = DeclareLaunchArgument(
        'use_foxglove',
        default_value='false',
        description='Use Foxglove: "true" or "false"'
    )
    log_use_foxglove = LogInfo(msg=["Using Foxglove: ", use_foxglove])


    # --- ROS2 Controller Manager ---
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_controllers_yaml, robot_description],
        output="screen",
    )

    # --- Control Nodes ---
    spawn_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
            arguments=[
                "joint_state_broadcaster",
                "--param-file", robot_controllers_yaml
            ],
        )

    spawn_imu_sensor_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["imu_sensor_broadcaster",
                "--param-file", robot_controllers_yaml
            ],
    )

    spawn_wbc_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["wbc",
                "--param-file", robot_controllers_yaml
            ],
    )

    # --- Standard Nodes ---
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    sim_odom_publisher_node = Node(
        package='sim_odom_publisher',
        executable='sim_odom_publisher_node',
        output='screen',
        parameters=[]
    )
        
    imu_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter_node',
        output='screen',
        parameters=[{
            'use_mag': False,
            'publish_tf': True,
            'world_frame': 'enu'
        }],
        remappings=[
            ('imu/data_raw', '/imu_sensor_broadcaster/imu'),
        ]
    )

    # State Estimator Knoten
    state_estimator_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[robot_controllers_yaml],
        arguments=[
            '--ros-args', '--log-level', 
            PythonExpression([
            "'debug' if 'ekf' in ", debug_log_packages_expr, " else 'info'"
        ])],
        # Remappen Sie den Output auf das Topic, das Ihr WBC erwartet
        remappings=[('odometry/filtered', '/state_estimator/odometry')]
    )
    
    ref_gen_node = Node(
        package='reference_generator',
        executable='reference_generator_node',
        parameters=[],
        arguments=[
            '--ros-args', '--log-level', 
            PythonExpression([
            "'debug' if 'reference_generator' in ", debug_log_packages_expr, " else 'info'"
        ])]
    )

    mpc_node = Node(
        package='mpc',
        executable='mpc_node',
        parameters=[robot_controllers_yaml],
        arguments=[
            '--ros-args', '--log-level', 
            PythonExpression([
            "'debug' if 'mpc' in ", debug_log_packages_expr, " else 'info'"
        ])]
    )

    foxglove_bridge_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        output='log',
        condition=IfCondition(PythonExpression([
            "'", use_foxglove, "'.lower()", " == 'true'"
        ])),
        parameters=[foxglove_params]
    )


    # --- Event Handlers ---
    



    return LaunchDescription([
        # Declare Arguments
        declare_debug_logger_arg,
        declare_use_foxglove_arg,

        # Log Arguments
        log_use_foxglove,

        # Add Nodes
        robot_state_publisher_node,
        ros2_control_node,
        spawn_joint_state_broadcaster,
        spawn_imu_sensor_broadcaster,
        spawn_wbc_controller,
        sim_odom_publisher_node,
        # imu_filter_node,
        # state_estimator_node,
        # ref_gen_node,
        # mpc_node,
        foxglove_bridge_node
    ])