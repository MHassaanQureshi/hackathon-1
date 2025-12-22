---
sidebar_label: 'Chapter 1: ROS 2 Fundamentals'
sidebar_position: 2
---

# Chapter 1: ROS 2 Fundamentals

## Overview

Welcome to the first chapter of Module 1. In this chapter, you'll learn the foundational concepts of Robot Operating System 2 (ROS 2), which serves as the communication backbone for robotic applications. Understanding these fundamentals is crucial as they form the basis for all subsequent modules in this book.

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain the core architecture of ROS 2
- Describe the differences between ROS 1 and ROS 2
- Create and run ROS 2 nodes
- Implement topics for message passing between nodes
- Create and use services for request-response communication
- Understand the role of ROS 2 in robotic systems

## What is ROS 2?

ROS 2 (Robot Operating System 2) is not an actual operating system, but rather a flexible framework for writing robotic software. It provides services designed for a heterogeneous computer cluster such as hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more.

ROS 2 is the successor to ROS 1, addressing many of the limitations of the original system, particularly around security, real-time performance, and deployment in production environments.

### Key Improvements in ROS 2

- **Security**: Built-in security features including authentication, authorization, and encryption
- **Real-time support**: Better support for real-time systems
- **Multiple DDS implementations**: Pluggable middleware for communication
- **Official support for Windows and macOS**: No longer Linux-only
- **Quality of Service (QoS) settings**: Configurable reliability and performance options
- **Lifecycle management**: Better management of node states and transitions

## Core Concepts

### Nodes

A node is a process that performs computation. In ROS 2, nodes are the fundamental building blocks of a robotic application. Each node typically handles a specific task or function.

In ROS 2, nodes are implemented as objects that inherit from the `rclpy.Node` class (in Python) or `rclcpp::Node` class (in C++). Each node should have a unique name within the ROS graph.

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        # Node initialization code here
```

### Topics and Messages

Topics enable asynchronous message passing between nodes. Multiple nodes can publish messages to a topic, and multiple nodes can subscribe to a topic to receive messages.

Messages are the data structures that are passed between nodes. They have a specific type and structure defined in `.msg` files.

```python
# Publisher example
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
```

### Services

Services provide synchronous request-response communication between nodes. A service client sends a request to a service server, which processes the request and returns a response.

Services are defined in `.srv` files that specify the request and response message types.

```python
# Service server example
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response
```

### Actions

Actions are a more sophisticated form of communication that support long-running tasks with feedback. They consist of a goal, feedback, and result.

Actions are defined in `.action` files and are useful for tasks like navigation where you need to track progress.

## The ROS 2 Ecosystem

### Packages

A package is the basic building unit of ROS 2. It contains nodes, libraries, and other resources needed for a specific functionality. Each package has a `package.xml` file that describes dependencies and metadata.

### Workspaces

A workspace is a directory that contains one or more packages. The typical structure includes:
- `src/` - Source code for packages
- `build/` - Build artifacts
- `install/` - Installation directory
- `log/` - Log files

### Launch Files

Launch files allow you to start multiple nodes at once with a single command. They can be written in Python or XML and provide a way to configure and manage complex robotic systems.

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='demo_nodes_py',
            executable='listener',
            name='minimal_listener',
        ),
    ])
```

## Quality of Service (QoS)

QoS settings allow you to configure the reliability and performance characteristics of communication between nodes. This is particularly important for real-time and safety-critical applications.

Key QoS policies include:
- **Reliability**: Reliable vs. best-effort delivery
- **Durability**: Volatile vs. transient-local durability
- **History**: Keep-all vs. keep-last history policy
- **Deadline**: Time constraints for message delivery

## Practical Exercise: Creating Your First ROS 2 Package

Let's create a simple ROS 2 package to understand the basics:

1. **Create a workspace directory:**
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   ```

2. **Create a new package:**
   ```bash
   ros2 pkg create --build-type ament_python my_first_package
   ```

3. **Navigate to the package:**
   ```bash
   cd my_first_package
   ```

4. **Create a simple publisher node:**
   Create a file `my_first_package/my_first_package/simple_publisher.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String

   class SimplePublisher(Node):
       def __init__(self):
           super().__init__('simple_publisher')
           self.publisher_ = self.create_publisher(String, 'chatter', 10)
           timer_period = 0.5  # seconds
           self.timer = self.create_timer(timer_period, self.timer_callback)
           self.i = 0

       def timer_callback(self):
           msg = String()
           msg.data = 'Hello World: %d' % self.i
           self.publisher_.publish(msg)
           self.get_logger().info('Publishing: "%s"' % msg.data)
           self.i += 1

   def main(args=None):
       rclpy.init(args=args)
       simple_publisher = SimplePublisher()
       rclpy.spin(simple_publisher)
       simple_publisher.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

5. **Make the file executable:**
   ```bash
   chmod +x my_first_package/simple_publisher.py
   ```

6. **Build the package:**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_first_package
   ```

7. **Source the workspace:**
   ```bash
   source install/setup.bash
   ```

8. **Run the publisher:**
   ```bash
   ros2 run my_first_package simple_publisher
   ```

## Summary

In this chapter, you've learned the fundamental concepts of ROS 2, including nodes, topics, services, and the overall ecosystem. These concepts form the foundation for all robotic applications built with ROS 2. Understanding these fundamentals is crucial for the next chapters where you'll implement Python agents and create robot models.

## Exercises

1. **Conceptual Understanding**: Explain the difference between topics and services in ROS 2. When would you use one over the other?

2. **Practical Implementation**: Create a simple subscriber node that subscribes to the 'chatter' topic and logs the received messages.

3. **QoS Exploration**: Research and explain when you would use reliable vs. best-effort QoS settings in a robotic application.

## Next Steps

In the next chapter, you'll dive deeper into implementing Python agents using the `rclpy` client library, building on the foundational concepts you've learned here.