# Chapter 1: Voice Commands with OpenAI Whisper

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the architecture and capabilities of OpenAI Whisper for speech recognition
- Set up and configure Whisper for real-time voice command processing
- Integrate Whisper with ROS 2 for robot voice command recognition
- Implement voice activity detection and command parsing
- Create context-aware voice command systems for robotics
- Design error handling and feedback mechanisms for voice commands

## Introduction to OpenAI Whisper

OpenAI Whisper is a state-of-the-art automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data collected from the web. It demonstrates robust speech recognition performance, even in challenging conditions such as background noise, accents, and technical language. In robotics applications, Whisper enables natural human-robot interaction through voice commands.

### Key Features of Whisper

1. **Multilingual Support**: Supports 99 languages including English, Spanish, French, German, Chinese, and many others
2. **Robust Recognition**: Handles various accents, background noise, and audio quality
3. **Timestamps**: Provides precise timing information for spoken words
4. **Punctuation**: Automatically adds punctuation to transcribed text
5. **Speaker Diarization**: Can identify different speakers in multi-person conversations
6. **Real-time Processing**: Can process audio streams in real-time with minimal latency

### Whisper Models

Whisper comes in five model sizes, each optimized for different use cases:

- **Tiny**: 39M parameters, fastest inference
- **Base**: 74M parameters, good balance of speed and accuracy
- **Small**: 244M parameters, higher accuracy
- **Medium**: 769M parameters, high accuracy
- **Large**: 1550M parameters, best accuracy

## Installing and Setting Up Whisper

### Prerequisites

```bash
# Install required dependencies
pip install openai-whisper
pip install pyaudio  # For audio capture
pip install speech-recognition  # For additional audio processing
pip install transformers  # For advanced NLP processing
pip install torch torchaudio  # For PyTorch-based processing
```

### Alternative Installation with Conda

```bash
# Create a new environment for Whisper
conda create -n whisper-robotics python=3.9
conda activate whisper-robotics

# Install Whisper with GPU support (if available)
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install openai-whisper
pip install pyaudio speech-recognition
```

### Basic Whisper Usage

```python
import whisper

# Load model (this downloads the model if not already present)
model = whisper.load_model("base")

# Transcribe audio file
result = model.transcribe("audio_file.wav")
print(result["text"])
```

## Real-time Voice Command Processing

For robotics applications, we need to process voice commands in real-time. Here's a complete implementation:

### Audio Capture and Processing

```python
import pyaudio
import numpy as np
import threading
import queue
import time
import whisper
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class AudioConfig:
    """Configuration for audio capture"""
    rate: int = 16000  # Sample rate
    chunk: int = 1024  # Audio chunk size
    channels: int = 1  # Mono audio
    format: int = pyaudio.paInt16  # 16-bit audio
    device_index: Optional[int] = None  # Audio device index

class AudioCapture:
    """Handles real-time audio capture from microphone"""

    def __init__(self, config: AudioConfig = AudioConfig()):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()

    def start_recording(self):
        """Start audio recording"""
        self.stream = self.audio.open(
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.rate,
            input=True,
            frames_per_buffer=self.config.chunk,
            input_device_index=self.config.device_index
        )
        self.is_recording = True

        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()

    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def _record_audio(self):
        """Internal method to record audio in a separate thread"""
        while self.is_recording:
            try:
                data = self.stream.read(self.config.chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                self.audio_queue.put(audio_data)
            except Exception as e:
                print(f"Error recording audio: {e}")

    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get the next audio chunk from the queue"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
```

### Whisper Transcription Service

```python
import whisper
import numpy as np
import threading
import queue
from typing import Optional, Callable
import time

class WhisperTranscriber:
    """Handles Whisper transcription with buffering for real-time processing"""

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model = whisper.load_model(model_size, device=device)
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_size = 16000 * 5  # 5 seconds of audio at 16kHz
        self.transcription_queue = queue.Queue()
        self.transcription_callback: Optional[Callable] = None

    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """Add an audio chunk to the buffer for transcription"""
        # Convert int16 to float32
        if audio_chunk.dtype == np.int16:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0

        # Append to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

        # Limit buffer size to prevent memory issues
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]

    def transcribe_buffer(self, language: Optional[str] = "en") -> Optional[str]:
        """Transcribe the current audio buffer"""
        if len(self.audio_buffer) < 16000:  # Need at least 1 second of audio
            return None

        # Transcribe the audio
        result = self.model.transcribe(
            self.audio_buffer,
            language=language,
            task="transcribe"
        )

        transcription = result["text"].strip()

        # Clear the buffer after transcription
        self.audio_buffer = np.array([], dtype=np.float32)

        return transcription if transcription else None

    def set_transcription_callback(self, callback: Callable[[str], None]):
        """Set callback function for transcriptions"""
        self.transcription_callback = callback
```

### Voice Activity Detection

```python
import numpy as np
from typing import Tuple

class VoiceActivityDetector:
    """Simple voice activity detection to identify when someone is speaking"""

    def __init__(self,
                 energy_threshold: float = 0.01,
                 silence_duration: float = 1.0,
                 min_speech_duration: float = 0.5):
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration

        self.silence_samples = 0
        self.speech_samples = 0
        self.is_speaking = False

    def detect_voice_activity(self, audio_chunk: np.ndarray) -> Tuple[bool, bool]:
        """Detect if voice activity is present in the audio chunk

        Returns:
            (is_speech_detected, is_silence_detected)
        """
        # Calculate energy of the audio chunk
        energy = np.mean(audio_chunk ** 2)

        # Check if above threshold
        if energy > self.energy_threshold:
            self.speech_samples += len(audio_chunk)
            self.silence_samples = 0  # Reset silence counter

            # If we were not speaking before, we just started
            was_speaking = self.is_speaking
            self.is_speaking = True

            # Return True only if we've been speaking long enough
            if self.speech_samples > self.min_speech_duration * 16000 and not was_speaking:
                return True, False
        else:
            self.silence_samples += len(audio_chunk)
            self.speech_samples = 0  # Reset speech counter

            # If we were speaking and now have enough silence
            was_speaking = self.is_speaking
            if self.silence_samples > self.silence_duration * 16000:
                self.is_speaking = False
                if was_speaking:
                    return False, True  # End of speech detected

        return False, False
```

## Integrating Whisper with ROS 2

Now let's create a ROS 2 node that integrates Whisper for voice command processing:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
import threading
import numpy as np
import whisper
import pyaudio
import queue
import time
from typing import Optional

class WhisperVoiceNode(Node):
    """ROS 2 node for voice command recognition using OpenAI Whisper"""

    def __init__(self):
        super().__init__('whisper_voice_node')

        # Publishers
        self.voice_command_pub = self.create_publisher(String, '/voice_command', 10)
        self.voice_status_pub = self.create_publisher(String, '/voice_status', 10)

        # Parameters
        self.declare_parameter('model_size', 'base')
        self.declare_parameter('language', 'en')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('sensitivity', 0.01)

        self.model_size = self.get_parameter('model_size').value
        self.language = self.get_parameter('language').value
        self.device = self.get_parameter('device').value
        self.sensitivity = self.get_parameter('sensitivity').value

        # Initialize Whisper model
        self.get_logger().info(f'Loading Whisper model: {self.model_size}')
        self.model = whisper.load_model(self.model_size, device=self.device)

        # Audio configuration
        self.rate = 16000
        self.chunk = 1024
        self.channels = 1
        self.format = pyaudio.paInt16

        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_duration = 5.0  # 5 seconds buffer

        # Voice activity detection
        self.energy_threshold = self.sensitivity
        self.silence_duration = 1.0
        self.silence_counter = 0

        # Start audio capture thread
        self.audio_thread = threading.Thread(target=self._audio_capture_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()

        # Timer for processing audio
        self.process_timer = self.create_timer(0.1, self._process_audio)

        self.get_logger().info('Whisper Voice Node initialized')

    def start_listening(self):
        """Start listening for voice commands"""
        if not self.is_listening:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            self.is_listening = True
            self.get_logger().info('Started listening for voice commands')

            status_msg = String()
            status_msg.data = 'started_listening'
            self.voice_status_pub.publish(status_msg)

    def stop_listening(self):
        """Stop listening for voice commands"""
        if self.is_listening and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.is_listening = False
            self.get_logger().info('Stopped listening for voice commands')

            status_msg = String()
            status_msg.data = 'stopped_listening'
            self.voice_status_pub.publish(status_msg)

    def _audio_capture_loop(self):
        """Capture audio in a separate thread"""
        while rclpy.ok():
            if self.is_listening and self.stream:
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    self.audio_queue.put(audio_data)
                except Exception as e:
                    self.get_logger().error(f'Error capturing audio: {e}')
            time.sleep(0.01)  # Small delay to prevent busy waiting

    def _process_audio(self):
        """Process audio chunks and detect voice activity"""
        # Get all available audio chunks
        while not self.audio_queue.empty():
            try:
                audio_chunk = self.audio_queue.get_nowait()

                # Add to buffer
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

                # Limit buffer size
                max_buffer_size = int(self.rate * self.buffer_duration)
                if len(self.audio_buffer) > max_buffer_size:
                    self.audio_buffer = self.audio_buffer[-max_buffer_size:]

                # Calculate energy for voice activity detection
                energy = np.mean(audio_chunk ** 2)

                if energy > self.energy_threshold:
                    # Voice activity detected
                    self.silence_counter = 0
                else:
                    # No voice activity
                    self.silence_counter += len(audio_chunk) / self.rate

                    # If silence duration exceeded, transcribe the buffer
                    if self.silence_counter >= self.silence_duration and len(self.audio_buffer) > self.rate:
                        self._transcribe_buffer()

            except queue.Empty:
                break

    def _transcribe_buffer(self):
        """Transcribe the current audio buffer using Whisper"""
        if len(self.audio_buffer) < self.rate:  # Need at least 1 second of audio
            return

        try:
            # Convert audio buffer to the right format
            audio_data = self.audio_buffer.copy()

            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_data,
                language=self.language,
                task="transcribe"
            )

            transcription = result["text"].strip()

            if transcription:  # Only publish if we have a meaningful transcription
                self.get_logger().info(f'Voice command: {transcription}')

                # Publish the voice command
                command_msg = String()
                command_msg.data = transcription
                self.voice_command_pub.publish(command_msg)

                # Clear the buffer after transcription
                self.audio_buffer = np.array([], dtype=np.float32)
                self.silence_counter = 0

        except Exception as e:
            self.get_logger().error(f'Error transcribing audio: {e}')

    def destroy_node(self):
        """Clean up resources when node is destroyed"""
        self.stop_listening()
        self.audio.terminate()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WhisperVoiceNode()
    node.start_listening()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.stop_listening()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Command Parsing and Intent Recognition

To make voice commands useful for robotics, we need to parse them and extract intent:

```python
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class VoiceCommand:
    """Represents a parsed voice command"""
    intent: str
    entities: Dict[str, str]
    confidence: float
    original_text: str

class CommandParser:
    """Parses voice commands and extracts intent and entities"""

    def __init__(self):
        # Define command patterns
        self.patterns = {
            'move_to': [
                r'move to (.+)',
                r'go to (.+)',
                r'go to the (.+)',
                r'navigate to (.+)',
                r'go to location (.+)'
            ],
            'move_forward': [
                r'move forward (.+)',
                r'go forward (.+)',
                r'move (.+) forward',
                r'go (.+) forward'
            ],
            'turn': [
                r'turn (.+)',
                r'rotate (.+)',
                r'pivot (.+)'
            ],
            'stop': [
                r'stop',
                r'hold on',
                r'wait',
                r'pause'
            ],
            'pick_up': [
                r'pick up (.+)',
                r'grasp (.+)',
                r'get (.+)',
                r'grab (.+)'
            ],
            'place': [
                r'place (.+) at (.+)',
                r'put (.+) at (.+)',
                r'drop (.+) at (.+)'
            ],
            'find': [
                r'find (.+)',
                r'look for (.+)',
                r'locate (.+)'
            ],
            'follow': [
                r'follow (.+)',
                r'follow me',
                r'come with me'
            ]
        }

        # Location entities
        self.locations = {
            'kitchen': ['kitchen', 'cooking area', 'cooking room'],
            'living room': ['living room', 'living', 'lounge'],
            'bedroom': ['bedroom', 'bed room', 'sleeping room'],
            'bathroom': ['bathroom', 'bath room', 'toilet', 'restroom'],
            'office': ['office', 'study', 'work room'],
            'dining room': ['dining room', 'dining', 'dinner room'],
            'hallway': ['hallway', 'hall', 'corridor'],
            'entrance': ['entrance', 'door', 'front door', 'entry']
        }

        # Direction entities
        self.directions = {
            'left': ['left', 'to the left', 'left side'],
            'right': ['right', 'to the right', 'right side'],
            'forward': ['forward', 'straight', 'ahead', 'straight ahead'],
            'backward': ['backward', 'back', 'reverse', 'backwards'],
            'up': ['up', 'upward', 'upwards'],
            'down': ['down', 'downward', 'downwards']
        }

    def parse_command(self, text: str) -> Optional[VoiceCommand]:
        """Parse a voice command and extract intent and entities"""
        text = text.lower().strip()

        # Try to match patterns
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    entities = {}

                    # Extract entities based on pattern groups
                    groups = match.groups()
                    if intent == 'move_to' and groups:
                        entities['location'] = self._normalize_location(groups[0].strip())
                    elif intent in ['move_forward', 'turn'] and groups:
                        entities['direction'] = self._normalize_direction(groups[0].strip())
                    elif intent == 'pick_up' and groups:
                        entities['object'] = groups[0].strip()
                    elif intent == 'place' and len(groups) >= 2:
                        entities['object'] = groups[0].strip()
                        entities['location'] = self._normalize_location(groups[1].strip())
                    elif intent == 'find' and groups:
                        entities['object'] = groups[0].strip()

                    return VoiceCommand(
                        intent=intent,
                        entities=entities,
                        confidence=0.9,  # High confidence for pattern matching
                        original_text=text
                    )

        # If no pattern matched, return None
        return None

    def _normalize_location(self, location: str) -> str:
        """Normalize location to standard form"""
        location = location.lower().strip()

        for standard, variations in self.locations.items():
            for variation in variations:
                if variation in location or location in variation:
                    return standard

        return location  # Return original if no match found

    def _normalize_direction(self, direction: str) -> str:
        """Normalize direction to standard form"""
        direction = direction.lower().strip()

        for standard, variations in self.directions.items():
            for variation in variations:
                if variation in direction or direction in variation:
                    return standard

        return direction  # Return original if no match found

# Example usage
parser = CommandParser()
command = parser.parse_command("Go to the kitchen")
if command:
    print(f"Intent: {command.intent}, Entities: {command.entities}")
```

## ROS 2 Integration for Command Execution

Now let's create a command execution node that processes voice commands and executes them:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
import json

class VoiceCommandExecutor(Node):
    """Executes voice commands by converting them to robot actions"""

    def __init__(self):
        super().__init__('voice_command_executor')

        # Subscribe to voice commands
        self.voice_sub = self.create_subscription(
            String, '/voice_command', self.voice_callback, 10)

        # Publisher for command status
        self.status_pub = self.create_publisher(String, '/command_status', 10)

        # Publisher for robot movement
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Command parser
        self.parser = CommandParser()

        self.get_logger().info('Voice Command Executor initialized')

    def voice_callback(self, msg):
        """Process incoming voice commands"""
        text = msg.data
        self.get_logger().info(f'Received voice command: {text}')

        # Parse the command
        command = self.parser.parse_command(text)

        if command:
            self.get_logger().info(f'Parsed command - Intent: {command.intent}, Entities: {command.entities}')
            self.execute_command(command)
        else:
            self.get_logger().info(f'Could not parse command: {text}')
            self.publish_status(f'Unrecognized command: {text}')

    def execute_command(self, command: VoiceCommand):
        """Execute a parsed voice command"""
        intent = command.intent

        if intent == 'move_to':
            self.execute_move_to(command.entities)
        elif intent == 'move_forward':
            self.execute_move_forward(command.entities)
        elif intent == 'turn':
            self.execute_turn(command.entities)
        elif intent == 'stop':
            self.execute_stop()
        elif intent == 'find':
            self.execute_find(command.entities)
        else:
            self.get_logger().info(f'Command not implemented: {intent}')
            self.publish_status(f'Command not implemented: {intent}')

    def execute_move_to(self, entities: Dict[str, str]):
        """Execute move to location command"""
        location = entities.get('location', '').lower()

        # Define known locations (in a real system, these would come from a map)
        known_locations = {
            'kitchen': (2.0, 1.0, 0.0),
            'living room': (0.0, 0.0, 0.0),
            'bedroom': (-2.0, 1.0, 0.0),
            'office': (1.0, -2.0, 0.0)
        }

        if location in known_locations:
            x, y, theta = known_locations[location]
            self.navigate_to_pose(x, y, theta)
            self.publish_status(f'Moving to {location}')
        else:
            self.get_logger().info(f'Unknown location: {location}')
            self.publish_status(f'Unknown location: {location}')

    def execute_move_forward(self, entities: Dict[str, str]):
        """Execute move forward command"""
        # Create a simple forward movement
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.2  # Move forward at 0.2 m/s
        cmd_vel.angular.z = 0.0

        # Publish for 2 seconds (in a real system, this would be handled differently)
        for _ in range(20):  # 2 seconds at 10Hz
            self.cmd_vel_pub.publish(cmd_vel)
            self.get_clock().sleep_for(rclpy.time.Duration(seconds=0.1))

        # Stop
        cmd_vel.linear.x = 0.0
        self.cmd_vel_pub.publish(cmd_vel)

        self.publish_status('Moved forward')

    def execute_turn(self, entities: Dict[str, str]):
        """Execute turn command"""
        direction = entities.get('direction', '').lower()

        cmd_vel = Twist()

        if direction in ['left', 'anticlockwise', 'counter clockwise']:
            cmd_vel.angular.z = 0.5  # Turn left
        elif direction in ['right', 'clockwise']:
            cmd_vel.angular.z = -0.5  # Turn right
        else:
            self.publish_status(f'Unknown direction: {direction}')
            return

        cmd_vel.linear.x = 0.0

        # Turn for 1 second
        for _ in range(10):  # 1 second at 10Hz
            self.cmd_vel_pub.publish(cmd_vel)
            self.get_clock().sleep_for(rclpy.time.Duration(seconds=0.1))

        # Stop
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)

        self.publish_status(f'Turned {direction}')

    def execute_stop(self):
        """Execute stop command"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)

        self.publish_status('Stopped')

    def execute_find(self, entities: Dict[str, str]):
        """Execute find object command"""
        obj = entities.get('object', '')
        self.publish_status(f'Searching for {obj}')

        # In a real system, this would trigger object detection
        # For now, just log the command
        self.get_logger().info(f'Searching for object: {obj}')

    def navigate_to_pose(self, x: float, y: float, theta: float):
        """Navigate to a specific pose using Nav2"""
        # Wait for the action server to be available
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available')
            return

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

        # Send goal
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.navigation_done_callback)

    def navigation_done_callback(self, future):
        """Handle navigation completion"""
        goal_result = future.result()
        if goal_result.accepted:
            self.get_logger().info('Navigation goal accepted')
        else:
            self.get_logger().info('Navigation goal rejected')

    def publish_status(self, status: str):
        """Publish command execution status"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandExecutor()

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

## Advanced Voice Command Features

### Wake Word Detection

```python
import numpy as np
import threading
import queue
from typing import Callable, Optional

class WakeWordDetector:
    """Detects wake words to activate voice command processing"""

    def __init__(self, wake_words: List[str] = None):
        self.wake_words = wake_words or ["robot", "hey robot", "assistant", "hello"]
        self.wake_word_detected_callback: Optional[Callable] = None
        self.is_active = False
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_size = 16000 * 2  # 2 seconds of audio

    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer for wake word detection"""
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

        # Limit buffer size
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer = self.audio_buffer[-int(self.buffer_size):]

    def check_wake_word(self, transcription: str) -> bool:
        """Check if transcription contains a wake word"""
        text = transcription.lower().strip()

        for wake_word in self.wake_words:
            if wake_word.lower() in text:
                return True
        return False

    def set_wake_word_callback(self, callback: Callable[[], None]):
        """Set callback for when wake word is detected"""
        self.wake_word_detected_callback = callback
```

### Context-Aware Voice Commands

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
import time

@dataclass
class RobotContext:
    """Represents the current context of the robot"""
    location: str = "unknown"
    battery_level: float = 100.0
    current_task: str = "idle"
    last_command_time: float = 0.0
    environment: str = "indoor"
    time_of_day: str = "day"

class ContextAwareCommandProcessor:
    """Processes voice commands based on robot context"""

    def __init__(self):
        self.context = RobotContext()
        self.command_history = []
        self.max_history = 10

    def update_context(self, **kwargs):
        """Update robot context"""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)

    def process_command_with_context(self, command: VoiceCommand) -> VoiceCommand:
        """Process command with context considerations"""
        # Add context information to entities if needed
        if command.intent == 'move_to' and command.entities.get('location') == 'there':
            # 'There' is relative, so we need to determine what 'there' means based on context
            command.entities['location'] = self._resolve_relative_location('there')

        # Add command to history
        self.command_history.append({
            'command': command,
            'timestamp': time.time()
        })

        # Keep history within limits
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)

        return command

    def _resolve_relative_location(self, location: str) -> str:
        """Resolve relative locations based on context"""
        if location == 'there' and self.context.location != 'unknown':
            # In a real system, this would use more sophisticated context
            return self.context.location
        return location
```

## Practical Exercise: Voice-Controlled Robot Navigation

### Exercise Overview

In this exercise, you'll implement a complete voice-controlled robot system that can understand natural language commands and execute them.

### Requirements

1. Set up Whisper for real-time voice command recognition
2. Implement command parsing and intent recognition
3. Create a navigation system that responds to voice commands
4. Test the system with various voice commands

### Implementation Steps

1. **Install Dependencies**:
```bash
pip install openai-whisper
pip install pyaudio
pip install speech-recognition
```

2. **Create Launch File** (`voice_control_launch.py`):
```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Whisper voice node
        Node(
            package='your_robot_voice',
            executable='whisper_voice_node',
            name='whisper_voice_node',
            output='screen',
            parameters=[
                {'model_size': 'base'},
                {'language': 'en'},
                {'device': 'cpu'},
                {'sensitivity': 0.01}
            ]
        ),

        # Command executor node
        Node(
            package='your_robot_voice',
            executable='voice_command_executor',
            name='voice_command_executor',
            output='screen'
        )
    ])
```

3. **Test Voice Commands**:
   - "Go to the kitchen"
   - "Move forward 2 meters"
   - "Turn left"
   - "Stop"
   - "Find the red ball"

## Best Practices and Optimization

### Performance Optimization

1. **Model Selection**: Use appropriate model size based on your hardware capabilities
2. **Audio Quality**: Ensure good audio input quality to improve recognition accuracy
3. **Buffer Management**: Optimize audio buffer sizes to balance latency and accuracy
4. **GPU Acceleration**: Use GPU if available for faster processing

### Accuracy Improvements

1. **Custom Vocabulary**: Fine-tune for specific robot commands
2. **Audio Preprocessing**: Apply noise reduction and audio enhancement
3. **Context Integration**: Use context to improve command interpretation
4. **Error Handling**: Implement robust error handling and fallback mechanisms

### Security Considerations

1. **Privacy**: Consider privacy implications of voice data processing
2. **Authentication**: Implement voice authentication for sensitive commands
3. **Validation**: Validate voice commands before execution
4. **Rate Limiting**: Prevent command flooding

## Troubleshooting Common Issues

1. **Poor Recognition**: Check audio input quality and adjust sensitivity
2. **High Latency**: Reduce buffer sizes or use smaller Whisper models
3. **False Activations**: Fine-tune wake word detection thresholds
4. **Command Misinterpretation**: Improve command parsing patterns

## Summary

This chapter covered the implementation of voice command recognition using OpenAI Whisper in robotics applications. We explored the architecture of Whisper, real-time audio processing, ROS 2 integration, command parsing, and execution systems.

Key takeaways include:
- Understanding Whisper's capabilities and model options for robotics
- Implementing real-time audio capture and processing
- Creating robust command parsing and intent recognition
- Integrating voice commands with robot navigation and control systems
- Following best practices for performance and accuracy

Voice commands provide a natural interface for human-robot interaction, enabling more intuitive control of robotic systems.

## Exercises

1. **Basic Voice Recognition**: Set up Whisper for voice command recognition and test with simple commands.

2. **Command Parsing**: Implement additional command patterns for more complex robot behaviors.

3. **Context Awareness**: Add context-aware processing to handle relative commands like "go there".

4. **Wake Word Detection**: Implement wake word detection to activate voice command processing.

5. **Error Handling**: Create robust error handling and feedback mechanisms for voice command failures.
