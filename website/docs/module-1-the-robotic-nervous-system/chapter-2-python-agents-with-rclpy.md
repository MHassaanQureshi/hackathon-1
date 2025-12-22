---
sidebar_label: 'Chapter 2: Python Agents with rclpy'
sidebar_position: 3
---

# Chapter 2: Python Agents with rclpy

## Overview

In this chapter, you'll learn how to implement ROS 2 nodes using Python and the `rclpy` client library. Building on the fundamentals from Chapter 1, you'll create sophisticated Python agents that can interact with the ROS 2 ecosystem, handle complex data processing, and coordinate with other nodes in a robotic system.

## Learning Objectives

By the end of this chapter, you will be able to:

- Use the `rclpy` library to create ROS 2 nodes in Python
- Implement publishers and subscribers for message passing
- Create service clients and servers for request-response communication
- Design action clients and servers for long-running tasks
- Handle parameters and configuration in ROS 2 nodes
- Implement error handling and logging in Python agents

## Introduction to rclpy

`rclpy` is the Python client library for ROS 2. It provides Python bindings for the ROS 2 client library (rcl), allowing you to write ROS 2 nodes in Python. The library follows the same patterns and concepts as the C++ client library (`rclcpp`), ensuring consistency across different programming languages.

### Key Components of rclpy

1. **Node**: The fundamental building block that provides execution context
2. **Publisher**: For sending messages on topics
3. **Subscriber**: For receiving messages from topics
4. **Service Server**: For providing services
5. **Service Client**: For calling services
6. **Action Server**: For providing actions
7. **Action Client**: For calling actions

## Creating a Basic Node

Let's start by creating a basic ROS 2 node using rclpy:

```python
import rclpy
from rclpy.node import Node

class BasicNode(Node):
    def __init__(self):
        super().__init__('basic_node')
        self.get_logger().info('Basic node initialized')

def main(args=None):
    rclpy.init(args=args)
    node = BasicNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Publishers and Subscribers

### Publisher Implementation

A publisher sends messages to a topic. Here's an example of creating a publisher:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    publisher_node = PublisherNode()

    try:
        rclpy.spin(publisher_node)
    except KeyboardInterrupt:
        pass
    finally:
        publisher_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Implementation

A subscriber receives messages from a topic. Here's how to implement one:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    subscriber_node = SubscriberNode()

    try:
        rclpy.spin(subscriber_node)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services

### Service Server

A service server provides a synchronous request-response interface:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceServerNode(Node):
    def __init__(self):
        super().__init__('service_server_node')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request\na: {request.a} b: {request.b}')
        return response

def main(args=None):
    rclpy.init(args=args)
    service_server_node = ServiceServerNode()

    try:
        rclpy.spin(service_server_node)
    except KeyboardInterrupt:
        pass
    finally:
        service_server_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client

A service client calls a service and waits for a response:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceClientNode(Node):
    def __init__(self):
        super().__init__('service_client_node')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    service_client_node = ServiceClientNode()

    response = service_client_node.send_request(1, 2)
    service_client_node.get_logger().info(f'Result: {response.sum}')

    service_client_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Actions

Actions are used for long-running tasks that provide feedback during execution:

### Action Server

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info('Returning result: {0}'.format(result.sequence))

        return result

def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()

    try:
        rclpy.spin(fibonacci_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        fibonacci_action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import time
    main()
```

### Action Client

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = FibonacciActionClient()

    action_client.send_goal(10)

    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
```

## Parameters

Parameters allow nodes to be configured at runtime:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('my_parameter', 'default_value')
        self.declare_parameter('integer_param', 42)
        self.declare_parameter('float_param', 3.14)
        self.declare_parameter('bool_param', True)

        # Get parameter values
        my_param = self.get_parameter('my_parameter').value
        int_param = self.get_parameter('integer_param').value

        self.get_logger().info(f'My parameter: {my_param}')
        self.get_logger().info(f'Integer parameter: {int_param}')

        # Set a callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            self.get_logger().info(f'Parameter {param.name} changed to {param.value}')
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    parameter_node = ParameterNode()

    try:
        rclpy.spin(parameter_node)
    except KeyboardInterrupt:
        pass
    finally:
        parameter_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    from rcl_interfaces.msg import SetParametersResult
    main()
```

## Advanced Patterns

### Timer-based Execution

Timers allow you to execute code at regular intervals:

```python
class TimedNode(Node):
    def __init__(self):
        super().__init__('timed_node')
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        self.get_logger().info(f'Timer callback executed: {self.counter}')
        self.counter += 1
```

### Multi-threaded Execution

For nodes that need to perform multiple operations simultaneously:

```python
import threading
from rclpy.qos import QoSProfile

class MultiThreadedNode(Node):
    def __init__(self):
        super().__init__('multi_threaded_node')

        # Use a different QoS profile for this example
        qos_profile = QoSProfile(depth=10)
        self.publisher = self.create_publisher(String, 'multi_thread_topic', qos_profile)

        # Start a background thread
        self.background_thread = threading.Thread(target=self.background_work)
        self.background_thread.start()

    def background_work(self):
        # Background work that can run while ROS 2 is spinning
        import time
        counter = 0
        while rclpy.ok():
            msg = String()
            msg.data = f'Background message {counter}'
            self.publisher.publish(msg)
            counter += 1
            time.sleep(2)
```

## Best Practices for Python Agents

### 1. Proper Resource Management

Always properly clean up resources in the node's destructor:

```python
class WellDesignedNode(Node):
    def __init__(self):
        super().__init__('well_designed_node')
        # Initialize resources here

    def destroy_node(self):
        # Clean up resources before destroying the node
        # Close file handles, stop threads, etc.
        super().destroy_node()
```

### 2. Error Handling

Implement proper error handling to make your nodes robust:

```python
def safe_callback(self, msg):
    try:
        # Process the message
        result = self.process_message(msg)
        self.publish_result(result)
    except Exception as e:
        self.get_logger().error(f'Error processing message: {e}')
        # Handle the error appropriately
```

### 3. Logging

Use appropriate logging levels to help with debugging:

```python
# Use different logging levels appropriately
self.get_logger().debug('Detailed debugging information')
self.get_logger().info('General information')
self.get_logger().warn('Warning message')
self.get_logger().error('Error message')
self.get_logger().fatal('Fatal error message')
```

## Practical Exercise: Creating a Robot Controller Agent

Let's create a more complex Python agent that simulates controlling a simple robot:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Create a QoS profile for reliable communication
        qos_profile = QoSProfile(depth=10)

        # Publishers for robot commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist, '/cmd_vel', qos_profile)

        # Subscribers for sensor data
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile)

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.obstacle_distance = float('inf')
        self.robot_state = 'SEARCHING'  # SEARCHING, AVOIDING, STOPPED

        self.get_logger().info('Robot controller initialized')

    def scan_callback(self, msg):
        # Process laser scan to detect obstacles
        if len(msg.ranges) > 0:
            # Get the minimum distance in the front 90 degrees
            front_ranges = msg.ranges[:len(msg.ranges)//8] + msg.ranges[-len(msg.ranges)//8:]
            self.obstacle_distance = min([r for r in front_ranges if r > 0.0 and r < float('inf')], default=float('inf'))

    def control_loop(self):
        cmd_msg = Twist()

        if self.obstacle_distance < 1.0:  # Obstacle within 1 meter
            self.robot_state = 'AVOIDING'
            # Turn to avoid obstacle
            cmd_msg.angular.z = 0.5
            cmd_msg.linear.x = 0.0
        else:
            self.robot_state = 'SEARCHING'
            # Move forward
            cmd_msg.linear.x = 0.5
            cmd_msg.angular.z = 0.0

        self.cmd_vel_publisher.publish(cmd_msg)
        self.get_logger().info(f'Robot state: {self.robot_state}, Obstacle distance: {self.obstacle_distance:.2f}')

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        pass
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

In this chapter, you've learned how to create sophisticated Python agents using the `rclpy` library. You've implemented publishers, subscribers, services, and actions, and learned best practices for creating robust and maintainable ROS 2 nodes. These skills are essential for building complex robotic systems that can interact with the ROS 2 ecosystem effectively.

## Exercises

1. **Enhanced Publisher/Subscriber**: Create a publisher that sends sensor data (e.g., temperature readings) and a subscriber that processes this data, calculating and publishing statistics like average, min, and max values.

2. **Parameter Configuration**: Implement a node that uses parameters to configure its behavior (e.g., movement speed, sensor thresholds) and can respond to parameter changes at runtime.

3. **Robot Navigation Agent**: Extend the robot controller example to implement a more sophisticated navigation behavior, such as wall following or goal-based navigation.

## Next Steps

In the next chapter, you'll learn about URDF (Unified Robot Description Format) and how to create humanoid robot models that can be used in simulation environments.