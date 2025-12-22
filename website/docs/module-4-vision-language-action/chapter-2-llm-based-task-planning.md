# Chapter 2: LLM-Based Task Planning

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the architecture and capabilities of Large Language Models for robotics
- Design LLM-based task planning systems for humanoid robots
- Integrate LLMs with robot control systems for intelligent task execution
- Implement multimodal reasoning combining vision and language
- Create context-aware task planning systems
- Design error recovery and validation mechanisms for LLM-generated plans

## Introduction to LLM-Based Task Planning

Large Language Models (LLMs) have revolutionized artificial intelligence by demonstrating remarkable capabilities in understanding, reasoning, and generating human-like text. In robotics, LLMs can serve as high-level cognitive controllers that interpret natural language commands, reason about the environment, and generate executable task plans for robots.

### Why LLMs for Robotics?

LLMs offer several advantages for robotics applications:

1. **Natural Language Understanding**: LLMs can interpret complex natural language commands without requiring structured input
2. **Common Sense Reasoning**: LLMs possess world knowledge that can be leveraged for task planning
3. **Generalization**: LLMs can handle novel situations and commands they haven't explicitly seen before
4. **Multi-step Planning**: LLMs can decompose complex tasks into sequences of subtasks
5. **Context Awareness**: LLMs can maintain context across multiple interactions

### Challenges in LLM-Robotics Integration

Despite their capabilities, LLMs present several challenges for robotics:

1. **Precision**: LLMs may generate plans that are conceptually correct but not executable by robots
2. **Grounding**: Abstract LLM concepts need to be grounded in the physical world
3. **Real-time Constraints**: LLM inference can be slow for real-time robotics applications
4. **Safety**: LLM-generated plans must be validated for safety before execution
5. **State Tracking**: LLMs need to be aware of the robot's current state and environment

## Setting Up LLM Integration

### Prerequisites

```bash
# Install required dependencies
pip install openai
pip install anthropic
pip install transformers
pip install torch
pip install tokenizers
pip install tiktoken  # For OpenAI token counting
```

### Basic LLM Interface

```python
import openai
import os
from typing import List, Dict, Any, Optional
import json

class LLMInterface:
    """Generic interface for interacting with LLMs"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key

    def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using the LLM"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return ""

    def generate_structured_output(self, prompt: str, output_format: str) -> Dict[str, Any]:
        """Generate structured output in a specific format"""
        formatted_prompt = f"{prompt}\n\nPlease respond in the following format: {output_format}"
        response = self.generate_text(formatted_prompt)

        try:
            # Try to parse as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # If JSON parsing fails, return as text
            return {"response": response}
```

## Task Planning Architecture

### Planning System Components

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """Represents a single task in a plan"""
    id: str
    description: str
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    status: TaskStatus = TaskStatus.PENDING

@dataclass
class TaskPlan:
    """Represents a complete task plan"""
    id: str
    description: str
    tasks: List[Task]
    context: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING

class TaskPlanner:
    """Main task planning system using LLMs"""

    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.current_plan: Optional[TaskPlan] = None
        self.task_history: List[TaskPlan] = []

    def create_plan(self,
                   goal: str,
                   context: Dict[str, Any],
                   robot_capabilities: List[str]) -> Optional[TaskPlan]:
        """Create a task plan for the given goal using LLM"""

        prompt = self._create_planning_prompt(goal, context, robot_capabilities)
        response = self.llm.generate_structured_output(
            prompt,
            "JSON with 'tasks' array containing objects with 'id', 'description', 'action', 'parameters', and 'dependencies'"
        )

        if "tasks" in response:
            tasks = []
            for task_data in response["tasks"]:
                task = Task(
                    id=task_data.get("id", f"task_{len(tasks)}"),
                    description=task_data.get("description", ""),
                    action=task_data.get("action", ""),
                    parameters=task_data.get("parameters", {}),
                    dependencies=task_data.get("dependencies", [])
                )
                tasks.append(task)

            plan = TaskPlan(
                id=f"plan_{len(self.task_history)}",
                description=goal,
                tasks=tasks,
                context=context
            )

            self.current_plan = plan
            self.task_history.append(plan)
            return plan

        return None

    def _create_planning_prompt(self, goal: str, context: Dict[str, Any], capabilities: List[str]) -> str:
        """Create a prompt for task planning"""
        return f"""
        You are a task planner for a robot. Create a detailed plan to accomplish the following goal:
        {goal}

        Robot capabilities: {', '.join(capabilities)}

        Current context: {json.dumps(context, indent=2)}

        Break down the goal into specific, executable tasks. Each task should be:
        1. Specific and actionable
        2. Have clear parameters
        3. Account for dependencies between tasks
        4. Be executable by a robot

        Return the plan as a JSON object with a 'tasks' array containing task objects.
        Each task should have: id, description, action, parameters, and dependencies.
        """
```

### Robot State and Context Management

```python
from dataclasses import dataclass
from typing import Dict, Any, List
import time

@dataclass
class RobotState:
    """Represents the current state of the robot"""
    position: Dict[str, float]  # x, y, theta
    battery_level: float
    current_task: str
    objects_in_environment: List[Dict[str, Any]]
    last_action_time: float
    capabilities: List[str]

class ContextManager:
    """Manages robot state and environmental context"""

    def __init__(self):
        self.robot_state = RobotState(
            position={"x": 0.0, "y": 0.0, "theta": 0.0},
            battery_level=100.0,
            current_task="idle",
            objects_in_environment=[],
            last_action_time=time.time(),
            capabilities=["move_to", "pick_up", "place", "navigate", "detect_objects"]
        )
        self.environment_map = {}
        self.object_database = {}

    def update_robot_position(self, x: float, y: float, theta: float):
        """Update robot position"""
        self.robot_state.position = {"x": x, "y": y, "theta": theta}

    def update_battery_level(self, level: float):
        """Update battery level"""
        self.robot_state.battery_level = max(0.0, min(100.0, level))

    def add_detected_object(self, obj: Dict[str, Any]):
        """Add a detected object to the environment"""
        self.robot_state.objects_in_environment.append(obj)

    def get_context_for_planning(self) -> Dict[str, Any]:
        """Get current context for task planning"""
        return {
            "robot_state": {
                "position": self.robot_state.position,
                "battery_level": self.robot_state.battery_level,
                "current_task": self.robot_state.current_task,
                "capabilities": self.robot_state.capabilities
            },
            "environment": {
                "objects": self.robot_state.objects_in_environment,
                "map": self.environment_map
            },
            "time": time.time()
        }
```

## LLM-Based Task Planning Implementation

### Advanced Planning System

```python
import asyncio
from typing import Callable, Optional
import re

class AdvancedTaskPlanner:
    """Advanced task planning system with multimodal capabilities"""

    def __init__(self, llm_interface: LLMInterface, context_manager: ContextManager):
        self.llm = llm_interface
        self.context_manager = context_manager
        self.current_plan: Optional[TaskPlan] = None
        self.active_tasks: List[Task] = []
        self.task_execution_callback: Optional[Callable] = None

    def create_multimodal_plan(self,
                             goal: str,
                             image_context: Optional[str] = None,
                             text_context: Optional[str] = None) -> Optional[TaskPlan]:
        """Create a plan considering both visual and textual context"""

        context = self.context_manager.get_context_for_planning()

        prompt = self._create_multimodal_prompt(goal, context, image_context, text_context)
        response = self.llm.generate_structured_output(
            prompt,
            "JSON with 'tasks' array and 'reasoning' explaining the plan"
        )

        if "tasks" in response:
            tasks = self._parse_tasks_from_response(response["tasks"])

            plan = TaskPlan(
                id=f"multimodal_plan_{int(time.time())}",
                description=goal,
                tasks=tasks,
                context=context
            )

            self.current_plan = plan
            return plan

        return None

    def _create_multimodal_prompt(self, goal: str, context: Dict[str, Any],
                                image_context: Optional[str], text_context: Optional[str]) -> str:
        """Create a prompt that considers multimodal context"""
        prompt = f"""
        You are an advanced task planner for a robot. Create a detailed plan to accomplish:
        {goal}

        Robot Context:
        {json.dumps(context, indent=2)}

        """

        if image_context:
            prompt += f"\nVisual Context: {image_context}\n"

        if text_context:
            prompt += f"\nAdditional Context: {text_context}\n"

        prompt += """
        Consider the following when creating the plan:
        1. The robot's current position and battery level
        2. Objects detected in the environment
        3. The robot's capabilities
        4. Safety constraints
        5. Efficiency of the plan

        Create a step-by-step plan with specific actions the robot can execute.
        Each task should have clear parameters and account for dependencies.

        Return as JSON with 'tasks' array and 'reasoning' field explaining your plan.
        """

        return prompt

    def _parse_tasks_from_response(self, task_list: List[Dict]) -> List[Task]:
        """Parse tasks from LLM response"""
        tasks = []
        for i, task_data in enumerate(task_list):
            task = Task(
                id=task_data.get("id", f"task_{i}"),
                description=task_data.get("description", ""),
                action=task_data.get("action", ""),
                parameters=task_data.get("parameters", {}),
                dependencies=task_data.get("dependencies", [])
            )
            tasks.append(task)
        return tasks

    def validate_plan(self, plan: TaskPlan) -> List[str]:
        """Validate the plan for safety and executability"""
        errors = []

        # Check if all actions are supported by the robot
        for task in plan.tasks:
            if task.action not in self.context_manager.robot_state.capabilities:
                errors.append(f"Task action '{task.action}' not supported by robot")

        # Check for circular dependencies
        if self._has_circular_dependencies(plan.tasks):
            errors.append("Plan contains circular dependencies")

        # Check battery constraints
        if self.context_manager.robot_state.battery_level < 20.0:
            errors.append("Battery level too low for complex task execution")

        return errors

    def _has_circular_dependencies(self, tasks: List[Task]) -> bool:
        """Check if tasks have circular dependencies"""
        # Simple cycle detection using adjacency list
        dependency_graph = {}
        for task in tasks:
            dependency_graph[task.id] = task.dependencies

        visited = set()
        rec_stack = set()

        def has_cycle(task_id):
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False

            visited.add(task_id)
            rec_stack.add(task_id)

            for dep in dependency_graph.get(task_id, []):
                if has_cycle(dep):
                    return True

            rec_stack.remove(task_id)
            return False

        for task in tasks:
            if has_cycle(task.id):
                return True

        return False
```

## ROS 2 Integration for LLM Planning

### Planning Node

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
import json
import threading
from typing import Optional, Dict, Any

class LLMPlanningNode(Node):
    """ROS 2 node for LLM-based task planning"""

    def __init__(self):
        super().__init__('llm_planning_node')

        # Publishers
        self.plan_pub = self.create_publisher(String, '/task_plan', 10)
        self.status_pub = self.create_publisher(String, '/planning_status', 10)

        # Subscribers
        self.goal_sub = self.create_subscription(
            String, '/high_level_goal', self.goal_callback, 10)
        self.robot_state_sub = self.create_subscription(
            String, '/robot_state', self.robot_state_callback, 10)

        # Parameters
        self.declare_parameter('llm_model', 'gpt-3.5-turbo')
        self.declare_parameter('api_key', '')

        self.llm_model = self.get_parameter('llm_model').value
        self.api_key = self.get_parameter('api_key').value

        # Initialize components
        if not self.api_key:
            self.get_logger().error('API key not provided')
            return

        self.llm_interface = LLMInterface(self.api_key, self.llm_model)
        self.context_manager = ContextManager()
        self.planner = AdvancedTaskPlanner(self.llm_interface, self.context_manager)

        self.get_logger().info('LLM Planning Node initialized')

    def goal_callback(self, msg):
        """Handle high-level goals from other nodes or users"""
        goal = msg.data
        self.get_logger().info(f'Received goal: {goal}')

        # Create and validate plan
        plan = self.planner.create_multimodal_plan(goal)

        if plan:
            validation_errors = self.planner.validate_plan(plan)

            if validation_errors:
                self.get_logger().error(f'Plan validation errors: {validation_errors}')
                self.publish_status(f'Plan validation failed: {", ".join(validation_errors)}')
                return

            # Publish the plan
            plan_msg = String()
            plan_msg.data = json.dumps({
                'id': plan.id,
                'description': plan.description,
                'tasks': [
                    {
                        'id': task.id,
                        'description': task.description,
                        'action': task.action,
                        'parameters': task.parameters,
                        'dependencies': task.dependencies
                    } for task in plan.tasks
                ]
            })
            self.plan_pub.publish(plan_msg)

            self.get_logger().info(f'Published task plan with {len(plan.tasks)} tasks')
            self.publish_status(f'Plan created with {len(plan.tasks)} tasks')
        else:
            self.get_logger().error('Failed to create plan')
            self.publish_status('Failed to create plan')

    def robot_state_callback(self, msg):
        """Update robot state from ROS messages"""
        try:
            state_data = json.loads(msg.data)

            # Update context manager with robot state
            if 'position' in state_data:
                pos = state_data['position']
                self.context_manager.update_robot_position(
                    pos.get('x', 0.0),
                    pos.get('y', 0.0),
                    pos.get('theta', 0.0)
                )

            if 'battery_level' in state_data:
                self.context_manager.update_battery_level(state_data['battery_level'])

        except json.JSONDecodeError:
            self.get_logger().error('Invalid robot state message format')

    def publish_status(self, status: str):
        """Publish planning status"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LLMPlanningNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Plan Execution Node

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import json
from typing import Dict, Any, List
import time

class PlanExecutionNode(Node):
    """Executes task plans generated by the LLM planner"""

    def __init__(self):
        super().__init__('plan_execution_node')

        # Subscribers
        self.plan_sub = self.create_subscription(
            String, '/task_plan', self.plan_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/execution_status', 10)
        self.feedback_pub = self.create_publisher(String, '/execution_feedback', 10)

        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Current plan and execution state
        self.current_plan = None
        self.executing_task_index = 0
        self.task_statuses = {}
        self.plan_active = False

        self.get_logger().info('Plan Execution Node initialized')

    def plan_callback(self, msg):
        """Receive and start executing a task plan"""
        try:
            plan_data = json.loads(msg.data)

            self.get_logger().info(f'Received plan with {len(plan_data["tasks"])} tasks')

            # Store the plan
            self.current_plan = plan_data
            self.executing_task_index = 0
            self.task_statuses = {task['id']: 'pending' for task in plan_data['tasks']}
            self.plan_active = True

            # Start execution
            self.execute_plan()

        except json.JSONDecodeError:
            self.get_logger().error('Invalid plan message format')

    def execute_plan(self):
        """Execute the current plan step by step"""
        if not self.current_plan or not self.plan_active:
            return

        tasks = self.current_plan['tasks']

        while self.executing_task_index < len(tasks) and self.plan_active:
            current_task = tasks[self.executing_task_index]

            self.get_logger().info(f'Executing task: {current_task["description"]}')
            self.publish_status(f'Executing: {current_task["description"]}')

            # Execute the task
            success = self.execute_task(current_task)

            if success:
                self.task_statuses[current_task['id']] = 'completed'
                self.executing_task_index += 1
            else:
                self.task_statuses[current_task['id']] = 'failed'
                self.get_logger().error(f'Task failed: {current_task["description"]}')
                self.publish_status(f'Task failed: {current_task["description"]}')
                break

        if self.executing_task_index >= len(tasks):
            self.get_logger().info('Plan completed successfully')
            self.publish_status('Plan completed')
        else:
            self.get_logger().info('Plan execution stopped')
            self.publish_status('Plan execution stopped')

        self.plan_active = False

    def execute_task(self, task: Dict[str, Any]) -> bool:
        """Execute a single task based on its action type"""
        action = task['action']
        parameters = task['parameters']

        if action == 'move_to':
            return self.execute_move_to(parameters)
        elif action == 'pick_up':
            return self.execute_pick_up(parameters)
        elif action == 'place':
            return self.execute_place(parameters)
        elif action == 'navigate':
            return self.execute_navigate(parameters)
        elif action == 'detect_objects':
            return self.execute_detect_objects(parameters)
        else:
            self.get_logger().error(f'Unknown action: {action}')
            return False

    def execute_move_to(self, params: Dict[str, Any]) -> bool:
        """Execute move to position task"""
        x = params.get('x', 0.0)
        y = params.get('y', 0.0)
        theta = params.get('theta', 0.0)

        return self.navigate_to_pose(x, y, theta)

    def execute_navigate(self, params: Dict[str, Any]) -> bool:
        """Execute navigation task"""
        target = params.get('target', '')

        # Define known locations (in a real system, this would come from a map)
        known_locations = {
            'kitchen': (2.0, 1.0, 0.0),
            'living room': (0.0, 0.0, 0.0),
            'bedroom': (-2.0, 1.0, 0.0),
            'office': (1.0, -2.0, 0.0)
        }

        if target.lower() in known_locations:
            x, y, theta = known_locations[target.lower()]
            return self.navigate_to_pose(x, y, theta)
        else:
            self.get_logger().error(f'Unknown target location: {target}')
            return False

    def navigate_to_pose(self, x: float, y: float, theta: float) -> bool:
        """Navigate to a specific pose using Nav2"""
        try:
            # Wait for the action server to be available
            if not self.nav_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error('Navigation action server not available')
                return False

            # Create goal
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = 'map'
            goal_msg.pose.pose.position.x = x
            goal_msg.pose.pose.position.y = y
            goal_msg.pose.pose.position.z = 0.0

            # Convert theta to quaternion
            from math import sin, cos
            goal_msg.pose.pose.orientation.z = sin(theta / 2.0)
            goal_msg.pose.pose.orientation.w = cos(theta / 2.0)

            # Send goal and wait for result
            future = self.nav_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, future)

            goal_result = future.result()
            if goal_result.accepted:
                self.get_logger().info('Navigation goal accepted')
                return True
            else:
                self.get_logger().info('Navigation goal rejected')
                return False

        except Exception as e:
            self.get_logger().error(f'Navigation error: {e}')
            return False

    def execute_pick_up(self, params: Dict[str, Any]) -> bool:
        """Execute pick up object task"""
        object_name = params.get('object', '')

        # In a real system, this would interface with manipulation stack
        self.get_logger().info(f'Picking up object: {object_name}')

        # Simulate pick up action
        time.sleep(2.0)  # Simulate time for pick up

        return True

    def execute_place(self, params: Dict[str, Any]) -> bool:
        """Execute place object task"""
        location = params.get('location', '')
        object_name = params.get('object', '')

        # In a real system, this would interface with manipulation stack
        self.get_logger().info(f'Placing {object_name} at {location}')

        # Simulate place action
        time.sleep(2.0)  # Simulate time for placing

        return True

    def execute_detect_objects(self, params: Dict[str, Any]) -> bool:
        """Execute object detection task"""
        area = params.get('area', 'surroundings')

        self.get_logger().info(f'Detecting objects in {area}')

        # In a real system, this would interface with perception stack
        # For now, we'll simulate detection

        return True

    def publish_status(self, status: str):
        """Publish execution status"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PlanExecutionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Multimodal Planning with Vision Integration

### Vision-Language Integration

```python
import base64
from io import BytesIO
from PIL import Image
import requests
from typing import Optional

class VisionLanguagePlanner:
    """Integrates visual information with LLM-based planning"""

    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface

    def analyze_scene(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze a scene using vision-language models"""
        # In a real implementation, this would use models like GPT-4V, Claude with vision, etc.
        # For now, we'll simulate the analysis

        # Convert image to base64 for potential API use
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        prompt = f"""
        Analyze this image and describe the scene, including:
        1. Objects present and their locations
        2. Spatial relationships between objects
        3. Potential navigation paths
        4. Safety considerations
        5. Action opportunities

        Describe the scene in detail for a robot planning system.
        """

        analysis = self.llm.generate_text(prompt)

        return {
            'objects': self._extract_objects_from_analysis(analysis),
            'spatial_description': self._extract_spatial_info(analysis),
            'navigation_info': self._extract_navigation_info(analysis),
            'full_analysis': analysis
        }

    def _extract_objects_from_analysis(self, analysis: str) -> List[Dict[str, Any]]:
        """Extract object information from scene analysis"""
        # This would use more sophisticated parsing in a real implementation
        # For simulation, we'll return some common objects
        return [
            {'name': 'table', 'type': 'furniture', 'location': 'center', 'properties': {'height': '0.75m'}},
            {'name': 'chair', 'type': 'furniture', 'location': 'left', 'properties': {'height': '0.5m'}},
            {'name': 'cup', 'type': 'object', 'location': 'on_table', 'properties': {'color': 'blue'}}
        ]

    def _extract_spatial_info(self, analysis: str) -> str:
        """Extract spatial information from analysis"""
        return "The scene contains a table in the center with a blue cup on it, and a chair to the left."

    def _extract_navigation_info(self, analysis: str) -> Dict[str, Any]:
        """Extract navigation information from analysis"""
        return {
            'clear_paths': ['forward', 'left', 'right'],
            'obstacles': ['table'],
            'safe_zones': ['open_area']
        }

    def create_vision_guided_plan(self,
                                goal: str,
                                image_data: bytes,
                                context: Dict[str, Any]) -> Optional[TaskPlan]:
        """Create a plan guided by visual information"""

        # Analyze the scene
        scene_analysis = self.analyze_scene(image_data)

        # Create a comprehensive prompt combining vision and goal
        prompt = f"""
        You are a robot task planner. The robot has the following goal:
        {goal}

        Current robot context:
        {json.dumps(context, indent=2)}

        Scene analysis from camera:
        {scene_analysis['full_analysis']}

        Objects detected: {json.dumps(scene_analysis['objects'], indent=2)}

        Based on the visual information and the goal, create a detailed task plan that:
        1. Considers the objects and layout in the scene
        2. Accounts for navigation around obstacles
        3. Exploits available objects and surfaces
        4. Ensures safe and efficient execution

        Return the plan as a JSON object with a 'tasks' array.
        """

        response = self.llm.generate_structured_output(
            prompt,
            "JSON with 'tasks' array containing task objects with id, description, action, parameters, and dependencies"
        )

        if "tasks" in response:
            tasks = self._parse_vision_tasks(response["tasks"], scene_analysis)

            return TaskPlan(
                id=f"vision_guided_plan_{int(time.time())}",
                description=goal,
                tasks=tasks,
                context={**context, **scene_analysis}
            )

        return None

    def _parse_vision_tasks(self, task_list: List[Dict], scene_analysis: Dict[str, Any]) -> List[Task]:
        """Parse tasks considering visual information"""
        tasks = []
        for i, task_data in enumerate(task_list):
            # Enhance task with visual context
            enhanced_parameters = self._enhance_parameters_with_vision(
                task_data.get("parameters", {}),
                scene_analysis
            )

            task = Task(
                id=task_data.get("id", f"vision_task_{i}"),
                description=task_data.get("description", ""),
                action=task_data.get("action", ""),
                parameters=enhanced_parameters,
                dependencies=task_data.get("dependencies", [])
            )
            tasks.append(task)
        return tasks

    def _enhance_parameters_with_vision(self, params: Dict[str, Any], scene_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance task parameters with visual information"""
        enhanced = params.copy()

        # Example: If the task involves navigation, add obstacle information
        if params.get('action_type') == 'navigate':
            obstacles = [obj['name'] for obj in scene_analysis.get('objects', [])]
            enhanced['avoid_objects'] = obstacles
            enhanced['safe_paths'] = scene_analysis.get('navigation_info', {}).get('clear_paths', [])

        return enhanced
```

## Practical Exercise: LLM-Based Robot Task Planning

### Exercise Overview

In this exercise, you'll implement a complete LLM-based task planning system that can interpret natural language commands, create executable plans, and coordinate with robot execution systems.

### Requirements

1. Set up LLM integration for task planning
2. Implement multimodal planning with vision integration
3. Create ROS 2 nodes for planning and execution
4. Test the system with various natural language commands

### Implementation Steps

1. **Set up API Access**:
```bash
# Export your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

2. **Create Planning Node Launch File** (`llm_planning_launch.py`):
```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Set environment variables
        SetEnvironmentVariable(name='OPENAI_API_KEY', value=os.environ.get('OPENAI_API_KEY', '')),

        # LLM Planning Node
        Node(
            package='your_robot_planning',
            executable='llm_planning_node',
            name='llm_planning_node',
            output='screen',
            parameters=[
                {'llm_model': 'gpt-3.5-turbo'},
                {'api_key': os.environ.get('OPENAI_API_KEY', '')}
            ]
        ),

        # Plan Execution Node
        Node(
            package='your_robot_planning',
            executable='plan_execution_node',
            name='plan_execution_node',
            output='screen'
        )
    ])
```

3. **Test Commands**:
   - "Go to the kitchen and bring me a cup"
   - "Navigate to the living room and avoid the table"
   - "Find the red ball and place it on the shelf"

## Safety and Validation Mechanisms

### Plan Validation System

```python
class PlanValidator:
    """Validates LLM-generated plans for safety and executability"""

    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager

    def validate_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        """Validate a plan and return validation results"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'safety_issues': []
        }

        # Check robot capabilities
        for task in plan.tasks:
            if task.action not in self.context_manager.robot_state.capabilities:
                results['errors'].append(f"Task '{task.description}' requires capability '{task.action}' which robot doesn't have")

        # Check battery constraints
        if self.context_manager.robot_state.battery_level < 15.0:
            results['warnings'].append("Low battery - consider charging before executing complex plans")

        # Check for safety issues
        safety_issues = self._check_safety_issues(plan)
        results['safety_issues'].extend(safety_issues)

        # Check for logical consistency
        if self._has_conflicting_tasks(plan):
            results['errors'].append("Plan contains conflicting tasks")

        # Check dependencies
        dependency_errors = self._validate_dependencies(plan)
        results['errors'].extend(dependency_errors)

        results['is_valid'] = len(results['errors']) == 0

        return results

    def _check_safety_issues(self, plan: TaskPlan) -> List[str]:
        """Check for potential safety issues in the plan"""
        issues = []

        for task in plan.tasks:
            if task.action == 'navigate' and 'target' in task.parameters:
                target = task.parameters['target']
                # Check if target location is safe based on context
                if target in ['staircase', 'cliff', 'dangerous_area']:
                    issues.append(f"Navigation to {target} may be unsafe")

        return issues

    def _has_conflicting_tasks(self, plan: TaskPlan) -> bool:
        """Check if plan has conflicting tasks"""
        # Simple check: if robot is holding an object, it shouldn't be told to pick up another
        holding_object = False
        for task in plan.tasks:
            if task.action == 'pick_up':
                if holding_object:
                    return True  # Already holding an object
                holding_object = True
            elif task.action == 'place':
                holding_object = False

        return False

    def _validate_dependencies(self, plan: TaskPlan) -> List[str]:
        """Validate task dependencies"""
        errors = []
        task_ids = {task.id for task in plan.tasks}

        for task in plan.tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    errors.append(f"Task '{task.id}' depends on non-existent task '{dep_id}'")

        return errors
```

## Best Practices and Optimization

### Performance Optimization

1. **Caching**: Cache frequently used plans and responses
2. **Prompt Engineering**: Optimize prompts for better and faster responses
3. **Parallel Processing**: Execute independent tasks in parallel when possible
4. **Local Models**: Consider using local LLMs for faster response times

### Safety Considerations

1. **Plan Validation**: Always validate plans before execution
2. **Human Oversight**: Implement human-in-the-loop for critical tasks
3. **Safety Constraints**: Define clear safety boundaries for robot actions
4. **Error Recovery**: Implement robust error handling and recovery mechanisms

### Integration Tips

1. **Modular Design**: Keep planning and execution components loosely coupled
2. **State Synchronization**: Ensure planning system has up-to-date robot state
3. **Feedback Loops**: Implement mechanisms for plan adjustment based on execution feedback
4. **Logging**: Maintain detailed logs for debugging and improvement

## Troubleshooting Common Issues

1. **LLM Hallucinations**: Validate all LLM outputs before execution
2. **Context Switching**: Maintain consistent context across planning sessions
3. **Performance**: Monitor and optimize planning time for real-time applications
4. **Safety**: Implement multiple layers of safety checks

## Summary

This chapter covered the implementation of LLM-based task planning for robotics. We explored how to integrate Large Language Models with robot systems to create intelligent task planning capabilities. The key components include natural language understanding, multimodal reasoning, plan validation, and safe execution.

Key takeaways include:
- Understanding how to structure LLM prompts for effective task planning
- Implementing multimodal planning that combines vision and language
- Creating safety and validation mechanisms for LLM-generated plans
- Integrating planning systems with ROS 2 for real-world robotics applications

LLM-based planning enables robots to understand complex natural language commands and execute sophisticated tasks by breaking them down into executable actions.

## Exercises

1. **Basic Planning**: Implement a simple LLM-based planner that can handle basic navigation commands.

2. **Multimodal Integration**: Add vision input to your planning system and create plans based on visual information.

3. **Safety Validation**: Implement additional safety checks and validation mechanisms for your plans.

4. **Context Awareness**: Add memory and context awareness to your planning system.

5. **Performance Optimization**: Optimize your planning system for faster response times and better accuracy.