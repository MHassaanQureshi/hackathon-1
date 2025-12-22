---
sidebar_label: 'Chapter 1: Physics Simulation in Gazebo'
sidebar_position: 6
---

# Chapter 1: Physics Simulation in Gazebo

## Overview

In this chapter, you'll learn how to create realistic physics simulations using Gazebo, a powerful 3D simulation environment for robotics. Gazebo provides accurate physics simulation, high-quality graphics, and convenient programmatic interfaces that make it ideal for testing robotic systems before deploying them in the real world.

## Learning Objectives

By the end of this chapter, you will be able to:

- Install and configure Gazebo for robotics simulation
- Load and simulate URDF robot models in Gazebo
- Configure physics properties and environmental parameters
- Implement gazebo plugins for sensor integration and control
- Create custom simulation worlds with obstacles and environments
- Interface Gazebo with ROS 2 for robot control and sensor data processing

## Introduction to Gazebo

Gazebo is a 3D dynamic simulator with the ability to accurately and efficiently simulate populations of robots in complex indoor and outdoor environments. It provides:

- **Physics simulation**: Accurate simulation of rigid body dynamics using ODE, Bullet, Simbody, or DART physics engines
- **Sensor simulation**: Support for various sensors including cameras, LiDAR, IMU, GPS, and more
- **Rendering**: High-quality graphics using OGRE3D engine
- **ROS integration**: Seamless integration with ROS/ROS 2 through Gazebo ROS packages

### Key Components of Gazebo

1. **Gazebo Server**: Core simulation engine that handles physics, rendering, and sensor updates
2. **Gazebo Client**: Visualization interface that connects to the server
3. **Gazebo Plugins**: Extensions that provide custom functionality (sensors, controllers, etc.)
4. **World Files**: SDF (Simulation Description Format) files that define environments

## Installing Gazebo

Gazebo comes in different versions, with Gazebo Garden (Fortress) being the latest stable version. For ROS 2 Humble Hawksbill, you'll typically use Gazebo Garden:

```bash
# Update package list
sudo apt update

# Install Gazebo Garden
sudo apt install ros-humble-gazebo-*

# Install additional gazebo ROS packages
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros-control
```

## Basic Gazebo Concepts

### SDF (Simulation Description Format)

SDF is an XML-based format used to describe simulation environments in Gazebo. It's similar to URDF but designed for complete simulation worlds:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- Include a model -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Define a light source -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Define a ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Loading URDF Models into Gazebo

To load your URDF robot models into Gazebo, you need to convert them to SDF format or use the robot_state_publisher and gazebo_ros spawn plugins.

### Method 1: Using xacro and spawn_model

First, convert your URDF to SDF:

```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf

# Or use xacro if your robot is defined in xacro
xacro robot.xacro | gz sdf -p /dev/stdin > robot.sdf
```

### Method 2: Using ROS 2 launch files

Create a launch file to spawn your robot in Gazebo:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch Gazebo server and client
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={'verbose': 'false'}.items()
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_robot'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        spawn_entity,
    ])
```

## Gazebo Plugins

Gazebo plugins extend the functionality of the simulation. Common plugin types include:

### 1. Sensor Plugins

```xml
<sensor name="camera" type="camera">
  <pose>0.1 0 0.1 0 0 0</pose>
  <camera name="head">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### 2. Control Plugins

For ROS 2 integration, use the joint state broadcaster and controller manager:

```xml
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/my_robot</robotNamespace>
  </plugin>
</gazebo>
```

### 3. IMU Sensor Plugin

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <visualize>false</visualize>
  <topic>__default_topic__</topic>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

## Creating Custom Worlds

Create custom world files to define your simulation environments:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_room">
    <!-- Physics engine configuration -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Ambient and background settings -->
    <scene>
      <ambient>0.4 0.4 0.4</ambient>
      <background>0.7 0.7 0.7</background>
      <shadows>true</shadows>
    </scene>

    <!-- Sun light -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Simple room with walls -->
    <model name="room_wall_1">
      <pose>-5 0 2.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.5 0.5 0.5 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name="room_wall_2">
      <pose>5 0 2.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.5 0.5 0.5 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name="room_wall_3">
      <pose>0 -5 2.5 0 0 1.5707</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.5 0.5 0.5 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name="room_wall_4">
      <pose>0 5 2.5 0 0 1.5707</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.5 0.5 0.5 1</specular>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Integrating with ROS 2

To integrate Gazebo with ROS 2, you need several components:

### 1. Robot State Publisher

Publishes joint states to tf:

```xml
<node name="robot_state_publisher" pkg="robot_state_publisher" exec="robot_state_publisher">
  <param name="use_sim_time" value="true"/>
</node>
```

### 2. Joint State Publisher (for non-simulated joints)

```xml
<node name="joint_state_publisher" pkg="joint_state_publisher" exec="joint_state_publisher">
  <param name="use_sim_time" value="true"/>
</node>
```

### 3. Example Launch File for Robot in Gazebo

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get the package share directory
    pkg_share = FindPackageShare('my_robot_description').find('my_robot_description')

    # Launch Gazebo with custom world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([pkg_share, 'worlds', 'simple_room.sdf']),
            'verbose': 'false'
        }.items()
    )

    # Publish robot state
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': open(PathJoinSubstitution([pkg_share, 'urdf', 'robot.urdf']).perform(None)).read()
        }]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
    ])
```

## Physics Configuration

Gazebo offers extensive physics configuration options:

### ODE Physics Parameters

```xml
<physics name="ode" type="ode">
  <!-- Time step -->
  <max_step_size>0.001</max_step_size>

  <!-- Real-time update rate -->
  <real_time_update_rate>1000</real_time_update_rate>

  <!-- Gravity -->
  <gravity>0 0 -9.8</gravity>

  <!-- ODE-specific parameters -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Practical Exercise: Simulating a Simple Mobile Robot

Let's create a simple differential drive robot simulation:

### 1. Robot URDF with Gazebo plugins

```xml
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.08"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.08"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Wheels -->
  <link name="wheel_left">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <link name="wheel_right">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="wheel_left_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_left"/>
    <origin xyz="0 0.1 -0.02" rpy="1.570796 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="wheel_right_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_right"/>
    <origin xyz="0 -0.1 -0.02" rpy="1.570796 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Gazebo plugins -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="wheel_left">
    <mu1>10</mu1>
    <mu2>10</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="wheel_right">
    <mu1>10</mu1>
    <mu2>10</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
    <material>Gazebo/Black</material>
  </gazebo>

  <!-- Differential drive plugin -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <update_rate>30</update_rate>
      <left_joint>wheel_left_joint</left_joint>
      <right_joint>wheel_right_joint</right_joint>
      <wheel_separation>0.2</wheel_separation>
      <wheel_diameter>0.1</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <publish_odom>true</publish_odom>
      <publish_wheel_tf>false</publish_wheel_tf>
      <publish_odom_tf>true</publish_odom_tf>
      <ros>
        <namespace>/simple_robot</namespace>
      </ros>
    </plugin>
  </gazebo>

</robot>
```

## Best Practices for Gazebo Simulation

### 1. Performance Optimization

- Use appropriate time step sizes (typically 0.001s for accurate simulation)
- Balance physics accuracy with real-time performance
- Simplify collision meshes where possible
- Use static models for unchanging environment elements

### 2. Stability Tips

- Set appropriate solver parameters (iterations, SOR parameter)
- Configure joint limits and dynamics properly
- Use appropriate friction and damping coefficients
- Ensure proper mass distribution in inertial properties

### 3. Integration Best Practices

- Use ROS 2 control interfaces for complex robots
- Implement proper sensor noise models
- Validate simulation results against real-world data
- Test both open-loop and closed-loop control in simulation

## Troubleshooting Common Issues

### 1. Robot Falls Through Ground

This is usually due to incorrect collision geometry or missing collision elements:

```xml
<!-- Make sure you have collision elements for all visual elements -->
<collision>
  <geometry>
    <cylinder radius="0.1" length="0.08"/>
  </geometry>
</collision>
```

### 2. Robot Jittering

This often occurs due to physics parameters or mass distribution issues:

```xml
<!-- Adjust physics parameters -->
<physics name="ode" type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_update_rate>1000</real_time_update_rate>
  <ode>
    <solver>
      <iters>20</iters>  <!-- Increase iterations -->
      <sor>1.0</sor>
    </solver>
  </ode>
</physics>
```

### 3. Robot Not Responding to Commands

Check that plugins are properly configured and topics are connected:

```bash
# Check topic connections
ros2 topic list
ros2 topic echo /simple_robot/cmd_vel
```

## Summary

In this chapter, you've learned how to create realistic physics simulations using Gazebo. You've understood the core concepts of SDF world files, URDF integration, physics configuration, and ROS 2 integration. You've also learned how to create custom environments and implement best practices for stable and efficient simulations.

These skills will be essential for testing the humanoid robots you created in Module 1, allowing you to validate their behavior in various simulated environments before moving to real-world implementation.

## Exercises

1. **Simple Robot Simulation**: Create a simulation of a simple robot (e.g., a cube with wheels) and implement basic movement control through ROS 2 topics.

2. **Custom Environment**: Design a custom world file with obstacles and test your robot's navigation capabilities in this environment.

3. **Sensor Integration**: Add a camera or LiDAR sensor to your robot and verify that sensor data is correctly published to ROS 2 topics.

## Next Steps

In the next chapter, you'll learn about high-fidelity environments in Unity, which complement the physics simulation capabilities of Gazebo with advanced visualization and rendering features.