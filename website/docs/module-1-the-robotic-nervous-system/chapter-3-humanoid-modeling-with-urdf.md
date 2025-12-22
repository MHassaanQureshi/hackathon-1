---
sidebar_label: 'Chapter 3: Humanoid Modeling with URDF'
sidebar_position: 4
---

# Chapter 3: Humanoid Modeling with URDF

## Overview

In this final chapter of Module 1, you'll learn about URDF (Unified Robot Description Format), which is the standard for representing robot models in ROS. You'll specifically focus on creating humanoid robot models that can be used in simulation environments. This knowledge is essential for the subsequent modules where you'll simulate and control these humanoid robots.

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand the structure and components of URDF files
- Create basic robot models with links and joints
- Define visual and collision properties for robot parts
- Model a simple humanoid robot with multiple degrees of freedom
- Use Xacro to create parameterized and reusable robot descriptions
- Validate and visualize URDF models in simulation

## Introduction to URDF

URDF (Unified Robot Description Format) is an XML-based format used to describe robot models in ROS. It defines the physical and visual properties of a robot, including its links (rigid parts), joints (connections between links), and other properties like inertia and materials.

URDF is fundamental to robotics simulation as it allows simulation environments like Gazebo to understand the robot's physical structure and properties.

### Basic URDF Structure

A basic URDF file has the following structure:

```xml
<?xml version="1.0"?>
<robot name="robot_name">
  <!-- Links define rigid parts of the robot -->
  <link name="link_name">
    <!-- Visual properties for rendering -->
    <visual>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
      <material name="color">
        <color rgba="0.8 0.2 0.2 1.0"/>
      </material>
    </visual>

    <!-- Collision properties for physics simulation -->
    <collision>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </collision>

    <!-- Inertial properties for physics simulation -->
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="joint_name" type="revolute">
    <parent link="parent_link"/>
    <child link="child_link"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>
</robot>
```

## Links

Links represent the rigid parts of a robot. Each link can have visual, collision, and inertial properties.

### Visual Properties

Visual properties define how a link appears in simulation and visualization tools:

```xml
<link name="link_name">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Choose one geometry type -->
      <box size="1 1 1"/>
      <!-- <cylinder radius="0.5" length="1"/> -->
      <!-- <sphere radius="0.5"/> -->
      <!-- <mesh filename="package://my_robot/meshes/link_name.stl"/> -->
    </geometry>
    <material name="red">
      <color rgba="1 0 0 1"/>
    </material>
  </visual>
</link>
```

### Collision Properties

Collision properties define how a link interacts with the physics engine:

```xml
<link name="link_name">
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="1 1 1"/>
    </geometry>
  </collision>
</link>
```

### Inertial Properties

Inertial properties define the mass and moments of inertia for physics simulation:

```xml
<link name="link_name">
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
  </inertial>
</link>
```

## Joints

Joints connect links and define how they can move relative to each other. URDF supports several joint types:

### Joint Types

1. **Fixed**: No movement allowed (0 DOF)
2. **Revolute**: Rotational movement around an axis (1 DOF)
3. **Continuous**: Like revolute but unlimited rotation (1 DOF)
4. **Prismatic**: Linear movement along an axis (1 DOF)
5. **Floating**: 6 DOF movement (3 translation + 3 rotation)
6. **Planar**: Movement in a plane (3 DOF)

### Joint Definition Example

```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

## Creating a Simple Humanoid Model

Let's create a basic humanoid robot model with a torso, head, arms, and legs:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Torso -->
  <link name="torso">
    <visual>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 1.0"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 1.0"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="skin">
        <color rgba="0.8 0.6 0.4 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 1.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0.1 0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="1"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.008" ixy="0.0" ixz="0.0" iyy="0.008" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="0" effort="15" velocity="1"/>
  </joint>

  <!-- Right Arm -->
  <link name="right_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.15 -0.1 0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="1"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.008" ixy="0.0" ixz="0.0" iyy="0.008" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="0" effort="15" velocity="1"/>
  </joint>

  <!-- Left Leg -->
  <link name="left_upper_leg">
    <visual>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.5"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_leg"/>
    <origin xyz="-0.1 0.1 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="0.78" effort="50" velocity="1"/>
  </joint>

  <link name="left_lower_leg">
    <visual>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.5"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="1.57" effort="40" velocity="1"/>
  </joint>

  <link name="left_foot">
    <visual>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.5" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="20" velocity="1"/>
  </joint>

  <!-- Right Leg -->
  <link name="right_upper_leg">
    <visual>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.5"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="right_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_leg"/>
    <origin xyz="-0.1 -0.1 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="0.78" effort="50" velocity="1"/>
  </joint>

  <link name="right_lower_leg">
    <visual>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.5"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="1.57" effort="40" velocity="1"/>
  </joint>

  <link name="right_foot">
    <visual>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_lower_leg"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.5" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="20" velocity="1"/>
  </joint>
</robot>
```

## Using Xacro for Complex Models

Xacro (XML Macros) is a macro language that extends URDF, allowing you to create more complex and reusable robot descriptions using variables, properties, and includes.

### Basic Xacro Example

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_example">

  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="base_radius" value="0.2" />
  <xacro:property name="base_height" value="0.1" />

  <!-- Define a macro for a wheel -->
  <xacro:macro name="wheel" params="prefix parent *origin">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <xacro:insert_block name="origin" />
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <axis xyz="0 1 0"/>
    </joint>

    <link name="${prefix}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="0 ${M_PI/2} 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.08"/>
        </geometry>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="0 ${M_PI/2} 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.08"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.2"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="${base_radius}" length="${base_height}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="${base_radius}" length="${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Use the wheel macro -->
  <xacro:wheel prefix="front_left" parent="base_link">
    <origin xyz="${base_radius} 0.1 0" rpy="0 0 0"/>
  </xacro:wheel>

  <xacro:wheel prefix="front_right" parent="base_link">
    <origin xyz="${base_radius} -0.1 0" rpy="0 0 0"/>
  </xacro:wheel>

</robot>
```

### Advanced Xacro for Humanoid Model

Here's a more sophisticated approach to modeling our humanoid using Xacro:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="advanced_humanoid">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_mass" value="10.0" />
  <xacro:property name="head_mass" value="2.0" />
  <xacro:property name="arm_mass" value="1.0" />
  <xacro:property name="leg_mass" value="2.0" />

  <!-- Inertial macro -->
  <xacro:macro name="default_inertial" params="mass">
    <inertial>
      <mass value="${mass}" />
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
    </inertial>
  </xacro:macro>

  <!-- Link macro -->
  <xacro:macro name="simple_link" params="name mass geometry_type *geometry material_name color_rgba">
    <link name="${name}">
      <visual>
        <geometry>
          <xacro:insert_block name="geometry" />
        </geometry>
        <material name="${material_name}">
          <color rgba="${color_rgba}" />
        </material>
      </visual>
      <collision>
        <geometry>
          <xacro:insert_block name="geometry" />
        </geometry>
      </collision>
      <xacro:default_inertial mass="${mass}" />
    </link>
  </xacro:macro>

  <!-- Torso -->
  <xacro:simple_link name="torso" mass="${torso_mass}" material_name="gray" color_rgba="0.5 0.5 0.5 1.0">
    <geometry>
      <box size="0.3 0.2 1.0" />
    </geometry>
  </xacro:simple_link>

  <!-- Head -->
  <xacro:simple_link name="head" mass="${head_mass}" material_name="skin" color_rgba="0.8 0.6 0.4 1.0">
    <geometry>
      <sphere radius="0.15" />
    </geometry>
  </xacro:simple_link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso" />
    <child link="head" />
    <origin xyz="0 0 1.0" rpy="0 0 0" />
    <axis xyz="0 0 1" />
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="10" velocity="1" />
  </joint>

</robot>
```

## Validating URDF Models

Before using your URDF model in simulation, it's important to validate it:

### Using check_urdf Command

```bash
# Check if the URDF is valid
check_urdf /path/to/robot.urdf

# Or with xacro
xacro robot.xacro | check_urdf /dev/stdin
```

### Using urdf_to_graphiz

```bash
# Generate a visual representation of the robot structure
urdf_to_graphiz /path/to/robot.urdf
```

## Visualization and Testing

### Using RViz

You can visualize your URDF model in RViz:

```bash
# Launch RViz with robot state publisher
ros2 run rviz2 rviz2
```

Then add a RobotModel display and set the robot description parameter to your URDF.

### Using Gazebo

For physics simulation:

```bash
# Launch Gazebo with your robot
gazebo --verbose -u world_file.world
```

## Best Practices for Humanoid Modeling

### 1. Realistic Proportions

When modeling humanoid robots, use realistic proportions based on human anatomy:

- Height: Typically 1.5-1.8m for adult-sized robots
- Limb ratios: Upper arm ~60% of total arm length, lower arm ~40%
- Head size: Approximately 1/8 of total height
- Torso: ~2/5 of total height

### 2. Proper Inertial Properties

Realistic inertial properties are crucial for stable simulation:

- Use simplified shapes for collision models (boxes, cylinders, spheres)
- Ensure center of mass is correctly positioned
- Use realistic mass values based on materials and size

### 3. Appropriate Joint Limits

Set realistic joint limits based on human range of motion:

```xml
<!-- Shoulder joint with realistic limits -->
<joint name="shoulder_joint" type="revolute">
  <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="50" velocity="1"/>
</joint>
```

### 4. Collision Avoidance

Consider potential self-collision scenarios and add appropriate collision meshes:

- Add collision elements for parts that might contact each other
- Use simpler collision geometries for performance
- Test various poses to ensure no unexpected collisions

## Practical Exercise: Complete Humanoid Model

Let's create a complete humanoid model using Xacro with all the components:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="complete_humanoid">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_height" value="0.8" />
  <xacro:property name="torso_width" value="0.3" />
  <xacro:property name="torso_depth" value="0.2" />
  <xacro:property name="head_radius" value="0.15" />
  <xacro:property name="upper_arm_length" value="0.3" />
  <xacro:property name="lower_arm_length" value="0.25" />
  <xacro:property name="upper_arm_radius" value="0.05" />
  <xacro:property name="lower_arm_radius" value="0.04" />
  <xacro:property name="upper_leg_length" value="0.4" />
  <xacro:property name="lower_leg_length" value="0.4" />
  <xacro:property name="leg_radius" value="0.06" />

  <!-- Materials -->
  <material name="red">
    <color rgba="0.8 0.2 0.2 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.2 0.2 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.2 0.8 0.2 1.0"/>
  </material>
  <material name="gray">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>
  <material name="skin">
    <color rgba="0.8 0.6 0.4 1.0"/>
  </material>

  <!-- Inertial macro -->
  <xacro:macro name="inertial_sphere" params="mass radius">
    <inertial>
      <mass value="${mass}" />
      <inertia
        ixx="${0.4 * mass * radius * radius}"
        ixy="0.0"
        ixz="0.0"
        iyy="${0.4 * mass * radius * radius}"
        iyz="0.0"
        izz="${0.4 * mass * radius * radius}" />
    </inertial>
  </xacro:macro>

  <xacro:macro name="inertial_box" params="mass x y z">
    <inertial>
      <mass value="${mass}" />
      <inertia
        ixx="${mass * (y*y + z*z) / 12.0}"
        ixy="0.0"
        ixz="0.0"
        iyy="${mass * (x*x + z*z) / 12.0}"
        iyz="0.0"
        izz="${mass * (x*x + y*y) / 12.0}" />
    </inertial>
  </xacro:macro>

  <xacro:macro name="inertial_cylinder" params="mass radius length axis">
    <xacro:if value="${axis == 'z'}">
      <inertial>
        <mass value="${mass}" />
        <inertia
          ixx="${mass * (3*radius*radius + length*length) / 12.0}"
          ixy="0.0"
          ixz="0.0"
          iyy="${mass * (3*radius*radius + length*length) / 12.0}"
          iyz="0.0"
          izz="${mass * radius * radius / 2.0}" />
      </inertial>
    </xacro:if>
    <xacro:if value="${axis == 'y'}">
      <inertial>
        <mass value="${mass}" />
        <inertia
          ixx="${mass * (3*radius*radius + length*length) / 12.0}"
          ixy="0.0"
          ixz="0.0"
          iyy="${mass * radius * radius / 2.0}"
          iyz="0.0"
          izz="${mass * (3*radius*radius + length*length) / 12.0}" />
      </inertial>
    </xacro:if>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link" />

  <!-- Torso -->
  <link name="torso">
    <visual>
      <origin xyz="0 0 ${torso_height/2}" rpy="0 0 0"/>
      <geometry>
        <box size="${torso_width} ${torso_depth} ${torso_height}"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <origin xyz="0 0 ${torso_height/2}" rpy="0 0 0"/>
      <geometry>
        <box size="${torso_width} ${torso_depth} ${torso_height}"/>
      </geometry>
    </collision>
    <xacro:inertial_box mass="10.0" x="${torso_width}" y="${torso_depth}" z="${torso_height}"/>
  </link>

  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="${head_radius}"/>
      </geometry>
      <material name="skin"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="${head_radius}"/>
      </geometry>
    </collision>
    <xacro:inertial_sphere mass="2.0" radius="${head_radius}"/>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 ${torso_height}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="10" velocity="1"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <origin xyz="0 0 ${-upper_arm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${upper_arm_radius}" length="${upper_arm_length}"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 ${-upper_arm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${upper_arm_radius}" length="${upper_arm_length}"/>
      </geometry>
    </collision>
    <xacro:inertial_cylinder mass="1.0" radius="${upper_arm_radius}" length="${upper_arm_length}" axis="y"/>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="${torso_width/2} ${torso_depth/2} ${torso_height*0.6}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="20" velocity="1"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <origin xyz="0 0 ${-lower_arm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${lower_arm_radius}" length="${lower_arm_length}"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 ${-lower_arm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${lower_arm_radius}" length="${lower_arm_length}"/>
      </geometry>
    </collision>
    <xacro:inertial_cylinder mass="0.8" radius="${lower_arm_radius}" length="${lower_arm_length}" axis="y"/>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 ${-upper_arm_length}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="0" effort="15" velocity="1"/>
  </joint>

  <!-- Right Arm -->
  <link name="right_upper_arm">
    <visual>
      <origin xyz="0 0 ${-upper_arm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${upper_arm_radius}" length="${upper_arm_length}"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 ${-upper_arm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${upper_arm_radius}" length="${upper_arm_length}"/>
      </geometry>
    </collision>
    <xacro:inertial_cylinder mass="1.0" radius="${upper_arm_radius}" length="${upper_arm_length}" axis="y"/>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="${torso_width/2} ${-torso_depth/2} ${torso_height*0.6}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="20" velocity="1"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <origin xyz="0 0 ${-lower_arm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${lower_arm_radius}" length="${lower_arm_length}"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 ${-lower_arm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${lower_arm_radius}" length="${lower_arm_length}"/>
      </geometry>
    </collision>
    <xacro:inertial_cylinder mass="0.8" radius="${lower_arm_radius}" length="${lower_arm_length}" axis="y"/>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 ${-upper_arm_length}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="0" effort="15" velocity="1"/>
  </joint>

  <!-- Left Leg -->
  <link name="left_upper_leg">
    <visual>
      <origin xyz="0 0 ${-upper_leg_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${leg_radius}" length="${upper_leg_length}"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 ${-upper_leg_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${leg_radius}" length="${upper_leg_length}"/>
      </geometry>
    </collision>
    <xacro:inertial_cylinder mass="2.0" radius="${leg_radius}" length="${upper_leg_length}" axis="y"/>
  </link>

  <joint name="left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_leg"/>
    <origin xyz="${-torso_width/2} ${torso_depth/2} 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/4}" effort="50" velocity="1"/>
  </joint>

  <link name="left_lower_leg">
    <visual>
      <origin xyz="0 0 ${-lower_leg_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${leg_radius}" length="${lower_leg_length}"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 ${-lower_leg_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${leg_radius}" length="${lower_leg_length}"/>
      </geometry>
    </collision>
    <xacro:inertial_cylinder mass="1.5" radius="${leg_radius}" length="${lower_leg_length}" axis="y"/>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 ${-upper_leg_length}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="${M_PI/2}" effort="40" velocity="1"/>
  </joint>

  <link name="left_foot">
    <visual>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.1 0.1"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.1 0.1"/>
      </geometry>
    </collision>
    <xacro:inertial_box mass="0.5" x="0.15" y="0.1" z="0.1"/>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0 0 ${-lower_leg_length}" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/6}" upper="${M_PI/6}" effort="20" velocity="1"/>
  </joint>

  <!-- Right Leg -->
  <link name="right_upper_leg">
    <visual>
      <origin xyz="0 0 ${-upper_leg_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${leg_radius}" length="${upper_leg_length}"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 ${-upper_leg_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${leg_radius}" length="${upper_leg_length}"/>
      </geometry>
    </collision>
    <xacro:inertial_cylinder mass="2.0" radius="${leg_radius}" length="${upper_leg_length}" axis="y"/>
  </link>

  <joint name="right_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_leg"/>
    <origin xyz="${-torso_width/2} ${-torso_depth/2} 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/4}" effort="50" velocity="1"/>
  </joint>

  <link name="right_lower_leg">
    <visual>
      <origin xyz="0 0 ${-lower_leg_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${leg_radius}" length="${lower_leg_length}"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 ${-lower_leg_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="${leg_radius}" length="${lower_leg_length}"/>
      </geometry>
    </collision>
    <xacro:inertial_cylinder mass="1.5" radius="${leg_radius}" length="${lower_leg_length}" axis="y"/>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 ${-upper_leg_length}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="${M_PI/2}" effort="40" velocity="1"/>
  </joint>

  <link name="right_foot">
    <visual>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.1 0.1"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.1 0.1"/>
      </geometry>
    </collision>
    <xacro:inertial_box mass="0.5" x="0.15" y="0.1" z="0.1"/>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_lower_leg"/>
    <child link="right_foot"/>
    <origin xyz="0 0 ${-lower_leg_length}" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/6}" upper="${M_PI/6}" effort="20" velocity="1"/>
  </joint>

</robot>
```

## Summary

In this chapter, you've learned how to create humanoid robot models using URDF and Xacro. You've understood the structure of URDF files, including links, joints, and their properties. You've also learned best practices for creating realistic humanoid models that can be used in simulation environments.

These skills are crucial for the upcoming modules where you'll simulate these humanoid robots in physics environments and implement control systems using ROS 2 nodes.

## Exercises

1. **Extended Humanoid Model**: Enhance the humanoid model by adding fingers to the hands with appropriate joints and limits.

2. **Custom Robot**: Create a completely new robot model (not humanoid) using the concepts learned in this chapter, such as a mobile manipulator or a quadruped robot.

3. **Xacro Practice**: Convert the simple URDF model at the beginning of this chapter to use Xacro macros and properties for better reusability.

## Next Steps

With the completion of Module 1, you now have a solid foundation in ROS 2 fundamentals, Python agents with rclpy, and humanoid modeling with URDF. In Module 2, you'll learn about physics simulation in Gazebo and high-fidelity environments in Unity, where you'll use the robot models you've created in this module.