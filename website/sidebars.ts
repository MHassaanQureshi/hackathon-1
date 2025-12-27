import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar configuration for the AI-Native Book
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro', 'ai-assistant-integration'],
    },
    {
      type: 'category',
      label: 'Module 1: Robotic Nervous System (ROS 2)',
      items: [
        'module-1-the-robotic-nervous-system/index',
        'module-1-the-robotic-nervous-system/chapter-1-ros2-fundamentals',
        'module-1-the-robotic-nervous-system/chapter-2-python-agents-with-rclpy',
        'module-1-the-robotic-nervous-system/chapter-3-humanoid-modeling-with-urdf',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-the-digital-twin/index',
        'module-2-the-digital-twin/chapter-1-physics-simulation-in-gazebo',
        'module-2-the-digital-twin/chapter-2-high-fidelity-environments-in-unity',
        'module-2-the-digital-twin/chapter-3-sensor-simulation',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-the-ai-robot-brain/index',
        'module-3-the-ai-robot-brain/chapter-1-isaac-sim-and-synthetic-data',
        'module-3-the-ai-robot-brain/chapter-2-isaac-ros-and-vslam',
        'module-3-the-ai-robot-brain/chapter-3-navigation-and-path-planning-with-nav2',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4-vision-language-action/index',
        'module-4-vision-language-action/chapter-1-voice-commands-with-openai-whisper',
        'module-4-vision-language-action/chapter-2-llm-based-task-planning',
        'module-4-vision-language-action/chapter-3-capstone-autonomous-humanoid-robot',
      ],
    },
  ],
};

export default sidebars;
