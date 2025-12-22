# Chapter 3: Navigation and Path Planning with Nav2

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the Navigation2 (Nav2) architecture and its components
- Configure and launch Nav2 for autonomous navigation
- Implement global and local path planning algorithms
- Integrate perception data from Isaac ROS into Nav2
- Set up costmaps for obstacle avoidance and dynamic navigation
- Configure behavior trees for navigation orchestration
- Implement recovery behaviors for challenging navigation scenarios

## Introduction to Navigation2 (Nav2)

Navigation2 (Nav2) is the next-generation autonomous navigation system for mobile robots in ROS 2. It represents a complete rewrite of the original ROS Navigation stack, designed from the ground up to leverage modern ROS 2 concepts such as composition, lifecycle management, and improved performance. Nav2 provides a flexible, modular framework for autonomous navigation that can be adapted to various robot platforms and environments.

### Key Features of Nav2

1. **Modular Architecture**: Components are designed as reusable plugins that can be swapped out based on specific requirements.
2. **Behavior Trees**: Navigation behaviors are orchestrated using behavior trees for complex decision-making.
3. **Lifecycle Management**: Proper state management for reliable system startup and shutdown.
4. **Advanced Path Planning**: Support for multiple global and local planners with dynamic reconfiguration.
5. **Costmap Integration**: Sophisticated costmap management for obstacle avoidance and terrain awareness.
6. **Recovery Behaviors**: Built-in mechanisms to handle navigation failures and recover from difficult situations.

## Nav2 Architecture and Components

The Nav2 system consists of several key components that work together to achieve autonomous navigation:

### Core Components

1. **Navigation Server**: Central coordination node that manages navigation requests and orchestrates other components.
2. **Planners**: Global and local path planning algorithms (Global Planner, Local Planner).
3. **Controller**: Local trajectory controller that follows planned paths.
4. **Costmap Server**: Manages obstacle and cost information for navigation.
5. **Recovery Server**: Handles navigation recovery behaviors when stuck or failed.
6. **Behavior Tree Server**: Executes navigation behavior trees.

### Navigation Pipeline

```
Goal Pose -> Global Planner -> Local Planner -> Controller -> Robot
              ↑                  ↑
        Costmap Updates    Costmap Updates
```

## Installing and Setting Up Nav2

### Prerequisites

```bash
# Install Nav2 packages
sudo apt update
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup

# Install additional dependencies
sudo apt install ros-humble-nav2-rviz-plugins ros-humble-nav2-msgs
sudo apt install ros-humble-robot-localization ros-humble-slam-toolbox
```

### Nav2 Workspace Setup

```bash
# Create workspace
mkdir -p ~/nav2_ws/src
cd ~/nav2_ws/src

# Clone Nav2 repositories
git clone -b humble https://github.com/ros-planning/navigation2.git
git clone -b humble https://github.com/ros-planning/navigation_msgs.git
git clone -b humble https://github.com/ros-planning/navigation2_tutorials.git

# Build the workspace
cd ~/nav2_ws
colcon build --symlink-install --packages-select nav2_bringup
source install/setup.bash
```

## Configuring Nav2 for Your Robot

### Robot Configuration Package

Create a configuration package for your robot's navigation:

```bash
# Create navigation package
mkdir -p ~/your_robot_ws/src/your_robot_nav2/config
mkdir -p ~/your_robot_ws/src/your_robot_nav2/launch
mkdir -p ~/your_robot_ws/src/your_robot_nav2/maps
```

### Main Navigation Parameters (`nav2_params.yaml`)

```yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: map
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: likelihood_field
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: odom
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: nav2_amcl::DifferentialMotionModel
    save_pose_delay: 0.5
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    scan_topic: scan

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    # Specify the JSON file containing the behavior tree for navigation
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: True

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Progress checker parameters
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    # Goal checker parameters
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True

    # DWB parameters
    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      debug_trajectory_details: False
      min_vel_x: 0.0
      min_vel_y: 0.0
      max_vel_x: 0.5
      max_vel_y: 0.0
      max_vel_theta: 1.0
      min_speed_xy: 0.0
      max_speed_xy: 0.5
      min_speed_theta: 0.0
      acc_lim_x: 2.5
      acc_lim_y: 0.0
      acc_lim_theta: 3.2
      decel_lim_x: -2.5
      decel_lim_y: 0.0
      decel_lim_theta: -3.2
      vx_samples: 20
      vy_samples: 5
      vtheta_samples: 20
      sim_time: 1.7
      linear_granularity: 0.05
      angular_granularity: 0.025
      transform_tolerance: 0.2
      xy_goal_tolerance: 0.25
      trans_stopped_velocity: 0.25
      short_circuit_trajectory_evaluation: True
      stateful: True
      critics: ["RotateToGoal", "Oscillation", "BaseObstacle", "GoalAlign", "PathAlign", "PathDist", "GoalDist"]
      BaseObstacle.scale: 0.02
      PathAlign.scale: 32.0
      PathAlign.forward_point_distance: 0.1
      GoalAlign.scale: 24.0
      GoalAlign.forward_point_distance: 0.1
      PathDist.scale: 32.0
      GoalDist.scale: 24.0
      RotateToGoal.scale: 32.0
      RotateToGoal.slowing_factor: 5.0
      RotateToGoal.lookahead_time: -1.0

controller_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.22
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.05
        z_voxels: 16
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        map_subscribe_transient_local: True
      always_send_full_costmap: True
  local_costmap_client:
    ros__parameters:
      use_sim_time: True
  local_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.22
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: True
  global_costmap_client:
    ros__parameters:
      use_sim_time: True
  global_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

planner_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

recoveries_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_recoveries::Spin"
    backup:
      plugin: "nav2_recoveries::BackUp"
    wait:
      plugin: "nav2_recoveries::Wait"
    use_sim_time: True

robot_state_publisher:
  ros__parameters:
    use_sim_time: True

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: true
      waypoint_pause_duration: 200
```

### Launch File (`navigation_launch.py`)

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # Get the launch directory
    bringup_dir = os.getenv('NAV2_BRINGUP_DIR')

    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    default_bt_xml_filename = LaunchConfiguration('default_bt_xml_filename')
    map_subscribe_transient_local = LaunchConfiguration('map_subscribe_transient_local')

    lifecycle_nodes = ['controller_server',
                       'planner_server',
                       'recoveries_server',
                       'bt_navigator',
                       'waypoint_follower']

    # Map fully qualified names to relative ones so the node's namespace can be prepended.
    # In case of the transforms (tf), currently, there doesn't seem to be a better alternative
    # https://github.com/ros/geometry2/issues/32
    # https://github.com/ros/robot_state_publisher/pull/30
    remappings = [('/tf', 'tf'),
                  ('/tf_static', 'tf_static')]

    return LaunchDescription([
        # Set env var to print messages to stdout immediately
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),

        DeclareLaunchArgument(
            'namespace', default_value='',
            description='Top-level namespace'),

        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        DeclareLaunchArgument(
            'autostart', default_value='true',
            description='Automatically startup the nav2 stack'),

        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(bringup_dir, 'params', 'nav2_params.yaml'),
            description='Full path to the ROS2 parameters file to use'),

        DeclareLaunchArgument(
            'default_bt_xml_filename',
            default_value=os.path.join(
                bringup_dir, 'behavior_trees', 'navigate_w_replanning_and_recovery.xml'),
            description='Full path to the behavior tree xml file to use'),

        DeclareLaunchArgument(
            'map_subscribe_transient_local', default_value='false',
            description='Whether to set the map subscriber to transient local'),

        Node(
            package='nav2_controller',
            executable='controller_server',
            output='screen',
            parameters=[params_file],
            remappings=remappings),

        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[params_file],
            remappings=remappings),

        Node(
            package='nav2_recoveries',
            executable='recoveries_server',
            name='recoveries_server',
            output='screen',
            parameters=[params_file],
            remappings=remappings),

        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[params_file],
            remappings=remappings),

        Node(
            package='nav2_waypoint_follower',
            executable='waypoint_follower',
            name='waypoint_follower',
            output='screen',
            parameters=[params_file],
            remappings=remappings),

        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time},
                        {'autostart': autostart},
                        {'node_names': lifecycle_nodes}]),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(bringup_dir, 'rviz', 'nav2_default_view.rviz')],
            parameters=[{'use_sim_time': use_sim_time}],
            remappings=[
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static'),
                ('/goal_pose', 'goal_pose'),
                ('/clicked_point', 'clicked_point'),
                ('/initialpose', 'initial_pose')])
    ])
```

## Global Path Planning

Global path planning determines the optimal route from the robot's current position to the goal position. Nav2 supports multiple global planners that can be configured based on the robot's requirements and environment characteristics.

### Available Global Planners

1. **NavFn**: A grid-based planner using Dijkstra's algorithm
2. **Global Planner**: An implementation of A* algorithm
3. **Carrot Planner**: A planner that finds a valid point near the goal
4. **Theta*: A planner that creates any-angle paths

### NavFn Planner Configuration

The NavFn planner is based on Dijkstra's algorithm and is effective for finding paths in grid-based maps. Here's how to configure it:

```yaml
planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5          # Tolerance for goal position
      use_astar: false        # Use A* instead of Dijkstra
      allow_unknown: true     # Allow planning through unknown space
```

### Advanced Global Planners

For more sophisticated path planning, you can use planners like:

```yaml
# Using Global Planner (A*)
planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"  # or nav2_global_planner/GlobalPlanner
      tolerance: 0.5
      use_astar: true
      allow_unknown: false
      planner_frequency: 1.0
```

## Local Path Planning and Trajectory Control

Local path planning focuses on short-term path following and obstacle avoidance around the robot. The local planner works in conjunction with the trajectory controller to ensure smooth, safe navigation.

### Dynamic Window Approach (DWA)

The DWA local planner evaluates possible velocities within the robot's kinematic constraints:

```yaml
controller_server:
  ros__parameters:
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001

    FollowPath:
      plugin: "nav2_dwb_controller::DWBLocalPlanner"
      debug_trajectory_details: False
      min_vel_x: 0.0
      min_vel_y: 0.0
      max_vel_x: 0.5
      max_vel_y: 0.0
      max_vel_theta: 1.0
      acc_lim_x: 2.5
      acc_lim_y: 0.0
      acc_lim_theta: 3.2
      decel_lim_x: -2.5
      decel_lim_y: 0.0
      decel_lim_theta: -3.2
      vx_samples: 20
      vy_samples: 5
      vtheta_samples: 20
      sim_time: 1.7
      linear_granularity: 0.05
      angular_granularity: 0.025
```

### Trajectory Controller Implementation

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math

class LocalTrajectoryController(Node):
    def __init__(self):
        super().__init__('local_trajectory_controller')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Robot state
        self.current_pose = None
        self.current_twist = None
        self.laser_data = None

        # Control parameters
        self.linear_vel = 0.2
        self.angular_vel = 0.5
        self.safe_distance = 0.5

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

    def odom_callback(self, msg):
        """Update robot pose and twist from odometry"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def scan_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = msg

    def check_obstacles(self):
        """Check for obstacles in front of the robot"""
        if self.laser_data is None:
            return False

        # Check distances in front of the robot (within 30 degrees)
        front_ranges = []
        center_idx = len(self.laser_data.ranges) // 2

        # Sample ranges in front of the robot
        for i in range(center_idx - 15, center_idx + 15):
            if 0 <= i < len(self.laser_data.ranges):
                if not math.isnan(self.laser_data.ranges[i]):
                    front_ranges.append(self.laser_data.ranges[i])

        if front_ranges:
            min_distance = min(front_ranges)
            return min_distance < self.safe_distance

        return False

    def control_loop(self):
        """Main control loop"""
        if self.current_pose is None or self.laser_data is None:
            return

        cmd_vel = Twist()

        # Check for obstacles
        if self.check_obstacles():
            # Stop robot if obstacle detected
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
        else:
            # Move forward
            cmd_vel.linear.x = self.linear_vel
            cmd_vel.angular.z = 0.0  # No rotation for now

        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    controller = LocalTrajectoryController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Costmap Integration

Costmaps are essential for navigation as they represent the environment with obstacles, free space, and navigation costs. Nav2 uses layered costmaps that combine multiple sources of information.

### Static Layer

The static layer represents the known map of the environment:

```yaml
static_layer:
  plugin: "nav2_costmap_2d::StaticLayer"
  map_subscribe_transient_local: True
  transform_tolerance: 0.2
```

### Obstacle Layer

The obstacle layer processes sensor data to detect dynamic obstacles:

```yaml
obstacle_layer:
  plugin: "nav2_costmap_2d::ObstacleLayer"
  enabled: True
  observation_sources: scan
  scan:
    topic: /scan
    max_obstacle_height: 2.0
    clearing: True
    marking: True
    data_type: "LaserScan"
    raytrace_max_range: 3.0
    raytrace_min_range: 0.0
    obstacle_max_range: 2.5
    obstacle_min_range: 0.0
```

### Inflation Layer

The inflation layer adds safety margins around obstacles:

```yaml
inflation_layer:
  plugin: "nav2_costmap_2d::InflationLayer"
  cost_scaling_factor: 3.0
  inflation_radius: 0.55
  inflate_unknown: false
```

## Behavior Trees for Navigation

Behavior trees provide a flexible way to define complex navigation behaviors. Nav2 uses behavior trees to orchestrate navigation tasks, recovery behaviors, and decision-making processes.

### Behavior Tree Structure

A typical navigation behavior tree includes:
- **Root**: The entry point of the tree
- **Sequences**: Execute children in order until one fails
- **Fallbacks**: Try children until one succeeds
- **Conditions**: Check if certain conditions are met
- **Actions**: Perform specific navigation tasks

### Example Behavior Tree (XML)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <Sequence name="NavigateWithReplanning">
            <PipelineSequence name="ComputePathThroughPoses">
                <IsPathValid/>
                <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
            </PipelineSequence>
            <PipelineSequence name="FollowPath">
                <IsGoalReached goal="{goal}" tolerance="0.25"/>
                <FollowPath path="{path}" controller_id="FollowPath"/>
            </PipelineSequence>
        </Sequence>
    </BehaviorTree>
</root>
```

### Custom Behavior Tree Nodes

You can create custom behavior tree nodes for specific navigation requirements:

```cpp
#include "behaviortree_cpp_v3/action_node.h"
#include "geometry_msgs/msg/pose_stamped.hpp"

class CheckSensorData : public BT::ActionNodeBase
{
public:
    CheckSensorData(const std::string& name, const BT::NodeConfiguration& config)
        : BT::ActionNodeBase(name, config) {}

    BT::NodeStatus tick() override
    {
        // Get sensor data from blackboard
        auto sensor_data = getInput<std::vector<double>>("sensor_data");

        // Check if sensor data indicates safe navigation
        bool is_safe = checkSafety(sensor_data.value());

        if (is_safe) {
            return BT::NodeStatus::SUCCESS;
        } else {
            return BT::NodeStatus::FAILURE;
        }
    }

    static BT::PortsList providedPorts() {
        return { BT::InputPort<std::vector<double>>("sensor_data") };
    }

private:
    bool checkSafety(const std::vector<double>& data) {
        // Implement safety check logic
        // Return true if navigation is safe
        return true;
    }
};
```

## Integrating Isaac ROS Perception Data

Integrating Isaac ROS perception data into Nav2 enhances navigation capabilities by providing rich environmental understanding.

### Depth Camera Integration

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2

class DepthCameraProcessor(Node):
    def __init__(self):
        super().__init__('depth_camera_processor')

        self.bridge = CvBridge()

        # Subscribe to Isaac ROS depth camera topics
        self.depth_sub = self.create_subscription(
            Image, '/depth_camera/image_rect_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/depth_camera/camera_info', self.camera_info_callback, 10)

        # Publish processed depth information to Nav2
        self.obstacle_pub = self.create_publisher(Image, '/processed_depth', 10)

        self.camera_info = None

    def camera_info_callback(self, msg):
        """Store camera intrinsic parameters"""
        self.camera_info = msg

    def depth_callback(self, msg):
        """Process depth image and extract obstacle information"""
        try:
            # Convert ROS image to OpenCV format
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Process depth image to identify obstacles
            obstacle_mask = self.process_depth_for_obstacles(depth_image)

            # Publish processed obstacle information
            obstacle_msg = self.bridge.cv2_to_imgmsg(obstacle_mask, encoding='mono8')
            obstacle_msg.header = msg.header
            self.obstacle_pub.publish(obstacle_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def process_depth_for_obstacles(self, depth_image):
        """Process depth image to identify obstacles"""
        # Define minimum distance threshold for obstacles
        min_distance = 1.0  # meters

        # Create binary mask where obstacles are detected
        obstacle_mask = (depth_image > 0) & (depth_image < min_distance)

        # Convert to uint8 for publishing
        obstacle_mask = obstacle_mask.astype(np.uint8) * 255

        return obstacle_mask

def main(args=None):
    rclpy.init(args=args)
    processor = DepthCameraProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### LiDAR Integration

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Point
import numpy as np

class IsaacROSLidarIntegration(Node):
    def __init__(self):
        super().__init__('isaac_ros_lidar_integration')

        # Subscribe to Isaac ROS LiDAR topics
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/lidar/points', self.lidar_callback, 10)

        # Publish to Nav2 costmap
        self.costmap_pub = self.create_publisher(PointCloud2, '/local_costmap/obstacles', 10)

        # Parameters
        self.min_range = 0.3  # Minimum detection range
        self.max_range = 10.0 # Maximum detection range
        self.obstacle_height = 2.0  # Height threshold for obstacles

    def lidar_callback(self, msg):
        """Process LiDAR point cloud data"""
        try:
            # Extract points from point cloud
            points = list(point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True))

            # Filter points based on range and height
            filtered_points = self.filter_points(points)

            # Create obstacle point cloud for Nav2
            obstacle_cloud = self.create_obstacle_cloud(filtered_points, msg.header)

            # Publish to Nav2
            self.costmap_pub.publish(obstacle_cloud)

        except Exception as e:
            self.get_logger().error(f'Error processing LiDAR data: {e}')

    def filter_points(self, points):
        """Filter points based on range and height"""
        filtered = []
        for x, y, z in points:
            distance = np.sqrt(x*x + y*y)
            if self.min_range <= distance <= self.max_range and z <= self.obstacle_height:
                filtered.append((x, y, z))
        return filtered

    def create_obstacle_cloud(self, points, header):
        """Create obstacle point cloud message"""
        # Create PointCloud2 message with filtered points
        fields = [
            PointCloud2.Field(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointCloud2.Field(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointCloud2.Field(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        obstacle_cloud = point_cloud2.create_cloud(header, fields, points)
        return obstacle_cloud

def main(args=None):
    rclpy.init(args=args)
    integration = IsaacROSLidarIntegration()
    rclpy.spin(integration)
    integration.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Recovery Behaviors

Recovery behaviors are crucial for handling navigation failures and getting the robot unstuck. Nav2 provides several built-in recovery behaviors and allows for custom implementations.

### Built-in Recovery Behaviors

1. **Spin**: Rotate the robot in place to clear local minima
2. **Backup**: Move the robot backward to escape tight spaces
3. **Wait**: Pause navigation temporarily

### Custom Recovery Behavior

```python
#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class CustomRecovery(Node):
    def __init__(self):
        super().__init__('custom_recovery')

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for recovery trigger
        self.recovery_sub = self.create_subscription(
            Bool, '/trigger_recovery', self.recovery_callback, 10)

        # Recovery parameters
        self.recovery_active = False
        self.recovery_timer = None
        self.recovery_step = 0

    def recovery_callback(self, msg):
        """Trigger custom recovery behavior"""
        if msg.data and not self.recovery_active:
            self.start_recovery()

    def start_recovery(self):
        """Start custom recovery sequence"""
        self.get_logger().info('Starting custom recovery behavior')
        self.recovery_active = True
        self.recovery_step = 0

        # Start recovery timer
        self.recovery_timer = self.create_timer(0.1, self.recovery_step_callback)

    def recovery_step_callback(self):
        """Execute recovery steps"""
        if not self.recovery_active:
            return

        cmd_vel = Twist()

        if self.recovery_step < 50:  # Step 1: Move backward
            cmd_vel.linear.x = -0.2
            cmd_vel.angular.z = 0.0
        elif self.recovery_step < 100:  # Step 2: Rotate slightly
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.3
        elif self.recovery_step < 150:  # Step 3: Move forward
            cmd_vel.linear.x = 0.2
            cmd_vel.angular.z = 0.0
        else:  # End recovery
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.end_recovery()

        self.cmd_vel_pub.publish(cmd_vel)
        self.recovery_step += 1

    def end_recovery(self):
        """End recovery behavior"""
        self.get_logger().info('Custom recovery behavior completed')
        self.recovery_active = False
        if self.recovery_timer:
            self.recovery_timer.cancel()
            self.recovery_timer = None

def main(args=None):
    rclpy.init(args=args)
    recovery = CustomRecovery()
    rclpy.spin(recovery)
    recovery.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Practical Exercise: Autonomous Navigation with Nav2

### Exercise Overview

In this exercise, you'll implement a complete navigation system using Nav2 that integrates perception data from Isaac ROS to enable autonomous navigation in a simulated environment.

### Requirements

1. Set up Nav2 with proper parameter configuration
2. Integrate Isaac ROS perception data into the navigation pipeline
3. Implement custom recovery behaviors
4. Test navigation in various scenarios

### Implementation Steps

1. **Configure Nav2 Parameters**: Set up the `nav2_params.yaml` file with appropriate parameters for your robot.

2. **Launch Navigation System**:
```bash
# Terminal 1: Launch the robot simulation
ros2 launch your_robot_gazebo your_robot_world.launch.py

# Terminal 2: Launch Nav2
ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=true \
  params_file:=~/your_robot_ws/src/your_robot_nav2/config/nav2_params.yaml

# Terminal 3: Launch RViz for visualization
ros2 run rviz2 rviz2 -d ~/your_robot_ws/src/your_robot_nav2/rviz/nav2_config.rviz
```

3. **Test Navigation**: Use RViz to send navigation goals and observe the robot's behavior.

4. **Monitor Performance**: Use ROS 2 tools to monitor navigation performance:
```bash
# Monitor navigation topics
ros2 topic echo /local_costmap/costmap_updates
ros2 topic echo /global_costmap/costmap_updates
ros2 topic echo /plan
ros2 topic echo /cmd_vel
```

## Best Practices and Optimization

### Performance Optimization

1. **Parameter Tuning**: Adjust planner frequencies, costmap resolutions, and controller parameters for optimal performance.

2. **Costmap Optimization**: Use appropriate resolution and update rates for your application.

3. **Sensor Fusion**: Combine multiple sensor sources for robust navigation.

### Safety Considerations

1. **Emergency Stop**: Implement emergency stop mechanisms.
2. **Safe Velocities**: Set appropriate velocity limits based on robot capabilities.
3. **Obstacle Detection**: Ensure reliable obstacle detection and avoidance.

### Troubleshooting Common Issues

1. **Path Not Found**: Check map quality, robot footprint, and costmap parameters.
2. **Oscillation**: Adjust controller parameters and critic weights.
3. **Stuck Behavior**: Verify recovery behaviors and costmap inflation settings.

## Summary

This chapter covered the comprehensive implementation of navigation and path planning using Navigation2 (Nav2). We explored the Nav2 architecture, configured parameters for navigation, implemented both global and local path planning, integrated perception data from Isaac ROS, and developed custom recovery behaviors.

Key takeaways include:
- Understanding the modular architecture of Nav2 and its core components
- Configuring proper parameters for navigation performance
- Integrating perception data to enhance navigation capabilities
- Implementing custom behaviors for specific requirements
- Following best practices for safe and efficient navigation

The combination of Nav2 with Isaac ROS perception provides a powerful foundation for autonomous navigation in complex environments, enabling robots to navigate safely while leveraging advanced perception capabilities.

## Exercises

1. **Basic Navigation**: Set up Nav2 for a simple differential drive robot and test basic navigation in Gazebo.

2. **Perception Integration**: Integrate Isaac ROS stereo vision data into Nav2 costmaps and evaluate navigation performance improvement.

3. **Custom Behavior**: Implement a custom behavior tree node for specific navigation requirements (e.g., door passage, narrow corridor navigation).

4. **Recovery Enhancement**: Develop and test custom recovery behaviors for specific challenging scenarios in your environment.

5. **Performance Analysis**: Analyze navigation performance metrics and tune parameters for optimal efficiency and safety.