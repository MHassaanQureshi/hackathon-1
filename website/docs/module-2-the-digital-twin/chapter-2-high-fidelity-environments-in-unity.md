---
sidebar_label: 'Chapter 2: High-fidelity Environments in Unity'
sidebar_position: 7
---

# Chapter 2: High-fidelity Environments in Unity

## Overview

In this chapter, you'll learn how to create high-fidelity simulation environments using Unity, a powerful game engine that excels at rendering realistic 3D scenes. Unity complements the physics simulation capabilities of Gazebo with advanced visualization and rendering features, making it ideal for creating immersive digital twin environments for robotics applications.

## Learning Objectives

By the end of this chapter, you will be able to:

- Set up Unity for robotics simulation and visualization
- Create realistic 3D environments with advanced lighting and materials
- Integrate Unity with ROS 2 for bidirectional communication
- Implement sensor simulation in Unity environments
- Create interactive and dynamic elements in simulation scenes
- Optimize Unity scenes for real-time performance in robotics applications

## Introduction to Unity for Robotics

Unity is a cross-platform game engine that has gained significant traction in robotics for creating high-fidelity simulation environments. Unity's strengths for robotics include:

- **Advanced rendering**: Physically-based rendering (PBR) materials, realistic lighting, and post-processing effects
- **Asset ecosystem**: Extensive library of 3D models, materials, and environments
- **Scripting flexibility**: C# scripting for custom behaviors and logic
- **XR support**: Virtual and augmented reality capabilities
- **ROS integration**: Unity Robotics Hub for seamless ROS 2 communication

### Unity Robotics Ecosystem

Unity provides several tools specifically for robotics:

1. **Unity Robotics Hub**: Collection of packages for ROS integration
2. **Unity ML-Agents**: For reinforcement learning and AI training
3. **Unity Perception**: For synthetic data generation
4. **Unity Simulation**: For large-scale distributed simulation

## Setting Up Unity for Robotics

### Installing Unity

Unity Hub is the recommended way to manage Unity installations:

1. Download Unity Hub from the Unity website
2. Install Unity 2021.3 LTS or later (recommended for robotics)
3. Add the Unity Robotics package during installation

### Installing ROS-TCP-Connector

The ROS-TCP-Connector package enables communication between Unity and ROS 2:

1. In Unity, go to Window > Package Manager
2. Click the + button and select "Add package from git URL..."
3. Enter: `https://github.com/Unity-Technologies/ROS-TCP-Connector.git`
4. Install the package

### Alternative: Unity Robotics Package

You can also install the complete Unity Robotics package:

1. Add the Unity Robotics Package git URL: `https://github.com/Unity-Technologies/Unity-Robotics-Hub.git`
2. This includes ROS-TCP-Connector, tutorials, and examples

## Basic Unity Concepts for Robotics

### GameObjects and Components

In Unity, everything is a GameObject with various Components:

```csharp
using UnityEngine;

public class RobotController : MonoBehaviour
{
    // Reference to robot parts
    public Transform[] wheels;
    public float wheelRadius = 0.1f;
    public float maxSpeed = 1.0f;

    // ROS communication
    private ROSConnection ros;
    private string robotTopic = "cmd_vel";

    void Start()
    {
        ros = ROSConnection.instance;
        ros.Subscribe<Twist>(robotTopic, ReceiveRobotCommand);
    }

    void ReceiveRobotCommand(Twist command)
    {
        // Process ROS command and update robot movement
        float linearVelocity = command.linear.x;
        float angularVelocity = command.angular.z;

        // Apply movement to wheels
        foreach (Transform wheel in wheels)
        {
            wheel.Rotate(Vector3.right, linearVelocity / wheelRadius * Mathf.Rad2Deg * Time.deltaTime);
        }
    }
}
```

### Physics in Unity

Unity uses NVIDIA PhysX for physics simulation:

```csharp
using UnityEngine;

public class PhysicsRobot : MonoBehaviour
{
    public float mass = 1.0f;
    public float friction = 0.5f;
    public float bounciness = 0.1f;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.mass = mass;

        // Configure physics material
        PhysicMaterial material = new PhysicMaterial();
        material.staticFriction = friction;
        material.dynamicFriction = friction;
        material.bounciness = bounciness;

        // Apply to all colliders
        Collider[] colliders = GetComponentsInChildren<Collider>();
        foreach (Collider col in colliders)
        {
            col.material = material;
        }
    }
}
```

## Creating Realistic Environments

### Terrain System

Unity's Terrain system is ideal for outdoor robotics environments:

```csharp
using UnityEngine;

public class TerrainGenerator : MonoBehaviour
{
    public int terrainSize = 1000;
    public int terrainHeight = 50;
    public int resolution = 257; // Must be 2^n + 1

    void Start()
    {
        Terrain terrain = GetComponent<Terrain>();
        terrain.terrainData = new TerrainData();

        // Set terrain size
        terrain.terrainData.size = new Vector3(terrainSize, terrainHeight, terrainSize);

        // Generate heightmap
        float[,] heights = new float[resolution, resolution];
        for (int x = 0; x < resolution; x++)
        {
            for (int y = 0; y < resolution; y++)
            {
                heights[x, y] = Mathf.PerlinNoise(
                    x * 0.01f,
                    y * 0.01f
                );
            }
        }

        terrain.terrainData.SetHeights(0, 0, heights);

        // Add textures
        AddTerrainTextures(terrain.terrainData);
    }

    void AddTerrainTextures(TerrainData terrainData)
    {
        // Create and assign textures
        SplatPrototype[] textures = new SplatPrototype[2];

        // Grass texture
        textures[0] = new SplatPrototype();
        textures[0].texture = Resources.Load<Texture2D>("GrassTexture");
        textures[0].tileSize = new Vector2(10, 10);

        // Dirt texture
        textures[1] = new SplatPrototype();
        textures[1].texture = Resources.Load<Texture2D>("DirtTexture");
        textures[1].tileSize = new Vector2(5, 5);

        terrainData.splatPrototypes = textures;

        // Create alphamap for texture blending
        float[,,] alphamap = new float[resolution, resolution, 2];
        for (int x = 0; x < resolution; x++)
        {
            for (int y = 0; y < resolution; y++)
            {
                float height = terrainData.GetHeight(x, y) / terrainHeight;
                alphamap[x, y, 0] = height; // Grass
                alphamap[x, y, 1] = 1 - height; // Dirt
            }
        }

        terrainData.SetAlphamaps(0, 0, alphamap);
    }
}
```

### Lighting and Post-Processing

Realistic lighting is crucial for high-fidelity environments:

```csharp
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class LightingSetup : MonoBehaviour
{
    public Light sunLight;
    public float intensity = 1.0f;
    public Color color = Color.white;

    void Start()
    {
        SetupLighting();
        SetupPostProcessing();
    }

    void SetupLighting()
    {
        // Configure directional light (sun)
        sunLight.type = LightType.Directional;
        sunLight.intensity = intensity;
        sunLight.color = color;
        sunLight.transform.rotation = Quaternion.Euler(50, -30, 0);

        // Enable shadows
        sunLight.shadows = LightShadows.Soft;
        sunLight.shadowStrength = 0.8f;
        sunLight.shadowResolution = UnityEngine.Rendering.LightShadowResolution.High;
    }

    void SetupPostProcessing()
    {
        // Add post-processing volume if using URP
        var volume = gameObject.AddComponent<UnityEngine.Rendering.Volume>();

        // Add common post-processing effects
        AddBloom(volume);
        AddAmbientOcclusion(volume);
    }

    void AddBloom(UnityEngine.Rendering.Volume volume)
    {
        var bloom = ScriptableObject.CreateInstance<UnityEngine.Rendering.Universal.Bloom>();
        bloom.active = true;
        bloom.threshold.value = 1.0f;
        bloom.intensity.value = 0.5f;
        bloom.scatter.value = 0.7f;

        volume.profile = new UnityEngine.Rendering.VolumeProfile();
        volume.profile.Add(bloom);
    }

    void AddAmbientOcclusion(UnityEngine.Rendering.Volume volume)
    {
        var ao = ScriptableObject.CreateInstance<UnityEngine.Rendering.Universal.AmbientOcclusion>();
        ao.active = true;
        ao.intensity.value = 0.5f;
        ao.radius.value = 2.0f;

        volume.profile.components.Add(ao);
    }
}
```

## Unity-ROS Integration

### Setting Up ROS Connection

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;

public class UnityROSIntegration : MonoBehaviour
{
    ROSConnection ros;
    string rosIP = "127.0.0.1";
    int rosPort = 10000;

    void Start()
    {
        ros = ROSConnection.instance;
        ros.Initialize(rosIP, rosPort);

        // Start publishers and subscribers
        InitializeROSCommunication();
    }

    void InitializeROSCommunication()
    {
        // Subscribe to robot commands
        ros.Subscribe<Twist>("cmd_vel", ReceiveRobotCommand);

        // Publish sensor data
        InvokeRepeating("PublishLaserScan", 0.1f, 0.1f);
        InvokeRepeating("PublishOdometry", 0.05f, 0.05f);
    }

    void ReceiveRobotCommand(Twist cmd)
    {
        // Process command and update robot
        Vector3 linear = new Vector3((float)cmd.linear.x, (float)cmd.linear.y, (float)cmd.linear.z);
        Vector3 angular = new Vector3((float)cmd.angular.x, (float)cmd.angular.y, (float)cmd.angular.z);

        // Apply to robot movement
        transform.Translate(linear * Time.deltaTime);
        transform.Rotate(angular * Mathf.Rad2Deg * Time.deltaTime);
    }

    void PublishLaserScan()
    {
        LaserScanMsg laserScan = new LaserScanMsg();

        // Fill in laser scan data
        laserScan.header = new std_msgs.Header();
        laserScan.header.stamp = new builtin_interfaces.Time();
        laserScan.header.frame_id = "laser_frame";

        laserScan.angle_min = -Mathf.PI / 2;
        laserScan.angle_max = Mathf.PI / 2;
        laserScan.angle_increment = Mathf.PI / 180; // 1 degree
        laserScan.time_increment = 0.0;
        laserScan.scan_time = 0.1f;
        laserScan.range_min = 0.1f;
        laserScan.range_max = 10.0f;

        // Simulate range data using raycasting
        int numRays = 181; // 181 rays from -90 to +90 degrees
        float[] ranges = new float[numRays];

        for (int i = 0; i < numRays; i++)
        {
            float angle = laserScan.angle_min + i * laserScan.angle_increment;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));

            RaycastHit hit;
            if (Physics.Raycast(transform.position, transform.TransformDirection(direction), out hit, laserScan.range_max))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = laserScan.range_max;
            }
        }

        laserScan.ranges = ranges;
        laserScan.intensities = new float[numRays]; // Empty intensities

        ros.Publish("scan", laserScan);
    }

    void PublishOdometry()
    {
        // Publish odometry data
        var odom = new nav_msgs.Odometry();
        odom.header = new std_msgs.Header();
        odom.header.stamp = new builtin_interfaces.Time();
        odom.header.frame_id = "odom";
        odom.child_frame_id = "base_link";

        // Position
        odom.pose.pose.position.x = transform.position.x;
        odom.pose.pose.position.y = transform.position.z; // Unity Z -> ROS Y
        odom.pose.pose.position.z = transform.position.y; // Unity Y -> ROS Z

        // Orientation (convert Unity quaternion to ROS quaternion)
        Quaternion unityRot = transform.rotation;
        odom.pose.pose.orientation.x = unityRot.x;
        odom.pose.pose.orientation.y = unityRot.z; // Unity Z -> ROS Y
        odom.pose.pose.orientation.z = unityRot.y; // Unity Y -> ROS Z
        odom.pose.pose.orientation.w = unityRot.w;

        ros.Publish("odom", odom);
    }
}
```

## Sensor Simulation in Unity

### Camera Sensor

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class CameraSensor : MonoBehaviour
{
    public Camera camera;
    public string topicName = "camera/image_raw";
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float publishRate = 30.0f; // Hz

    private ROSConnection ros;
    private RenderTexture renderTexture;
    private Texture2D texture2D;

    void Start()
    {
        ros = ROSConnection.instance;

        // Create render texture
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        camera.targetTexture = renderTexture;

        // Create texture for reading
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

        // Start publishing
        InvokeRepeating("PublishImage", 0.0f, 1.0f/publishRate);
    }

    void PublishImage()
    {
        // Set the active render texture
        RenderTexture.active = renderTexture;

        // Read pixels from the render texture
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();

        // Convert to ROS image format
        var imageMsg = new sensor_msgs.Image();
        imageMsg.header = new std_msgs.Header();
        imageMsg.header.stamp = new builtin_interfaces.Time();
        imageMsg.header.frame_id = "camera_frame";

        imageMsg.height = (uint)imageHeight;
        imageMsg.width = (uint)imageWidth;
        imageMsg.encoding = "rgb8";
        imageMsg.is_bigendian = 0;
        imageMsg.step = (uint)(imageWidth * 3); // 3 bytes per pixel (RGB)

        // Get raw pixel data
        Color32[] colors = texture2D.GetPixels32();
        byte[] data = new byte[colors.Length * 3];

        for (int i = 0; i < colors.Length; i++)
        {
            data[i * 3] = colors[i].r;
            data[i * 3 + 1] = colors[i].g;
            data[i * 3 + 2] = colors[i].b;
        }

        imageMsg.data = data;

        ros.Publish(topicName, imageMsg);
    }

    void OnDisable()
    {
        if (renderTexture != null)
            renderTexture.Release();
    }
}
```

### IMU Sensor

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class IMUSensor : MonoBehaviour
{
    public string topicName = "imu/data";
    public float publishRate = 100.0f; // Hz

    private ROSConnection ros;
    private Rigidbody rb;

    void Start()
    {
        ros = ROSConnection.instance;
        rb = GetComponent<Rigidbody>();

        InvokeRepeating("PublishIMU", 0.0f, 1.0f/publishRate);
    }

    void PublishIMU()
    {
        var imuMsg = new sensor_msgs.Imu();
        imuMsg.header = new std_msgs.Header();
        imuMsg.header.stamp = new builtin_interfaces.Time();
        imuMsg.header.frame_id = "imu_frame";

        // Orientation (from Unity rotation)
        Quaternion unityRot = transform.rotation;
        imuMsg.orientation.x = unityRot.x;
        imuMsg.orientation.y = unityRot.z; // Unity Z -> ROS Y
        imuMsg.orientation.z = unityRot.y; // Unity Y -> ROS Z
        imuMsg.orientation.w = unityRot.w;

        // Angular velocity (from rigidbody)
        if (rb != null)
        {
            Vector3 angularVel = rb.angularVelocity;
            imuMsg.angular_velocity.x = angularVel.x;
            imuMsg.angular_velocity.y = angularVel.z; // Unity Z -> ROS Y
            imuMsg.angular_velocity.z = angularVel.y; // Unity Y -> ROS Z
        }

        // Linear acceleration (from gravity and movement)
        Vector3 linearAcc = Physics.gravity;
        if (rb != null)
        {
            linearAcc += rb.velocity / Time.fixedDeltaTime;
        }

        imuMsg.linear_acceleration.x = linearAcc.x;
        imuMsg.linear_acceleration.y = linearAcc.z; // Unity Z -> ROS Y
        imuMsg.linear_acceleration.z = linearAcc.y; // Unity Y -> ROS Z

        ros.Publish(topicName, imuMsg);
    }
}
```

## Advanced Unity Features for Robotics

### NavMesh for Path Planning

Unity's NavMesh system is excellent for pathfinding:

```csharp
using UnityEngine;
using UnityEngine.AI;

public class RobotNavigator : MonoBehaviour
{
    public Transform target;
    public NavMeshAgent agent;
    public string goalTopic = "move_base_simple/goal";

    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        ROSConnection.instance.Subscribe<geometry_msgs.PoseStamped>(goalTopic, SetGoal);
    }

    void SetGoal(geometry_msgs.PoseStamped goal)
    {
        Vector3 goalPos = new Vector3(
            (float)goal.pose.position.x,
            (float)goal.pose.position.z, // ROS Z -> Unity Y
            (float)goal.pose.position.y  // ROS Y -> Unity Z
        );

        agent.SetDestination(goalPos);
    }

    void Update()
    {
        // Publish feedback on navigation progress
        if (agent.remainingDistance < agent.stoppingDistance)
        {
            // Goal reached
            PublishGoalStatus(3); // SUCCEEDED
        }
        else
        {
            PublishGoalStatus(1); // ACTIVE
        }
    }

    void PublishGoalStatus(int status)
    {
        // Publish actionlib status
        var statusMsg = new actionlib_msgs.GoalStatus();
        statusMsg.status = (byte)status;

        ROSConnection.instance.Publish("move_base/status", statusMsg);
    }
}
```

### Animation and Inverse Kinematics

For humanoid robots, Unity's animation system is very useful:

```csharp
using UnityEngine;
using UnityEngine.Animations;

public class HumanoidController : MonoBehaviour
{
    public Animator animator;
    public float walkSpeed = 1.0f;
    public float runSpeed = 3.0f;

    void Start()
    {
        if (animator == null)
            animator = GetComponent<Animator>();
    }

    void Update()
    {
        // Get input (could come from ROS messages)
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");

        // Calculate movement
        Vector3 movement = new Vector3(horizontal, 0, vertical).normalized;

        // Set animation parameters
        animator.SetFloat("Speed", movement.magnitude);
        animator.SetFloat("Direction", Mathf.Atan2(movement.x, movement.z));

        // Apply movement to character
        transform.Translate(movement * walkSpeed * Time.deltaTime);
    }

    public void SetJointPositions(float[] jointAngles)
    {
        // Set specific joint angles for precise control
        // This would typically interface with ROS joint state messages
        for (int i = 0; i < jointAngles.Length; i++)
        {
            // Apply joint angles to specific transforms
            // This requires mapping ROS joint names to Unity transforms
        }
    }
}
```

## Optimization Techniques

### Level of Detail (LOD)

For performance in large environments:

```csharp
using UnityEngine;

[RequireComponent(typeof(Renderer))]
public class RobotLOD : MonoBehaviour
{
    public float[] lodDistances = {10f, 30f, 60f};
    public Renderer[] lodRenderers;

    private Camera mainCamera;
    private float[] lodSqrDistances;

    void Start()
    {
        mainCamera = Camera.main;
        lodSqrDistances = new float[lodDistances.Length];

        for (int i = 0; i < lodDistances.Length; i++)
        {
            lodSqrDistances[i] = lodDistances[i] * lodDistances[i];
        }
    }

    void Update()
    {
        float distanceSqr = (transform.position - mainCamera.transform.position).sqrMagnitude;

        for (int i = 0; i < lodRenderers.Length; i++)
        {
            lodRenderers[i].enabled = (distanceSqr < lodSqrDistances[i]);
        }
    }
}
```

### Occlusion Culling

Unity's occlusion culling can significantly improve performance:

```csharp
using UnityEngine;

public class OcclusionCullingSetup : MonoBehaviour
{
    public bool enableOcclusionCulling = true;

    void Start()
    {
        // This is typically configured in the Unity Editor
        // but we can access it programmatically
        if (enableOcclusionCulling)
        {
            // Enable occlusion culling for the scene
            // This requires baking the occlusion data in Editor
            StaticOcclusionCulling.OCCLUSION_AREA_LAYER = 31; // Use the highest layer
        }
    }
}
```

## Practical Exercise: Unity Robotics Environment

Let's create a complete Unity scene for robotics simulation:

### 1. Robot Prefab Setup

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class UnityRobot : MonoBehaviour
{
    [Header("Robot Configuration")]
    public float maxLinearSpeed = 1.0f;
    public float maxAngularSpeed = 1.0f;

    [Header("ROS Topics")]
    public string cmdVelTopic = "cmd_vel";
    public string laserScanTopic = "scan";
    public string odomTopic = "odom";

    private ROSConnection ros;
    private Rigidbody rb;
    private Vector3 targetVelocity;

    void Start()
    {
        ros = ROSConnection.instance;
        rb = GetComponent<Rigidbody>();

        // Subscribe to ROS topics
        ros.Subscribe<geometry_msgs.Twist>(cmdVelTopic, ProcessCommand);

        // Start publishing sensors
        InvokeRepeating("PublishLaserScan", 0.1f, 0.1f);
        InvokeRepeating("PublishOdometry", 0.05f, 0.05f);
    }

    void ProcessCommand(geometry_msgs.Twist cmd)
    {
        // Convert ROS command to Unity movement
        float linear = (float)cmd.linear.x;
        float angular = (float)cmd.angular.z;

        // Calculate target velocity in world space
        Vector3 forwardVel = transform.forward * linear * maxLinearSpeed;
        Vector3 angularVel = Vector3.up * angular * maxAngularSpeed;

        targetVelocity = forwardVel;
        transform.Rotate(angularVel * Time.deltaTime);
    }

    void FixedUpdate()
    {
        // Apply movement
        rb.MovePosition(rb.position + targetVelocity * Time.fixedDeltaTime);
    }

    void PublishLaserScan()
    {
        // Implementation similar to the earlier example
        // Uses raycasting to simulate LiDAR
    }

    void PublishOdometry()
    {
        // Implementation similar to the earlier example
        // Publishes position and orientation
    }
}
```

### 2. Environment Setup Script

```csharp
using UnityEngine;

public class RoboticsEnvironment : MonoBehaviour
{
    [Header("Environment Configuration")]
    public int robotCount = 5;
    public Transform robotPrefab;
    public Vector3 spawnArea = new Vector3(20, 0, 20);

    [Header("Obstacles")]
    public GameObject[] obstaclePrefabs;
    public int obstacleCount = 10;

    void Start()
    {
        SpawnRobots();
        SpawnObstacles();
    }

    void SpawnRobots()
    {
        for (int i = 0; i < robotCount; i++)
        {
            Vector3 spawnPos = new Vector3(
                Random.Range(-spawnArea.x/2, spawnArea.x/2),
                0.5f, // Height to place above ground
                Random.Range(-spawnArea.z/2, spawnArea.z/2)
            );

            Instantiate(robotPrefab, spawnPos, Quaternion.identity);
        }
    }

    void SpawnObstacles()
    {
        for (int i = 0; i < obstacleCount; i++)
        {
            GameObject obstaclePrefab = obstaclePrefabs[Random.Range(0, obstaclePrefabs.Length)];

            Vector3 spawnPos = new Vector3(
                Random.Range(-spawnArea.x/2, spawnArea.x/2),
                0, // Position on ground
                Random.Range(-spawnArea.z/2, spawnArea.z/2)
            );

            Instantiate(obstaclePrefab, spawnPos, Quaternion.identity);
        }
    }
}
```

## Best Practices for Unity Robotics

### 1. Performance Optimization

- Use object pooling for frequently instantiated objects
- Implement frustum culling for distant objects
- Use shader variants to reduce draw calls
- Optimize physics by reducing collision complexity where possible

### 2. Accuracy Considerations

- Ensure Unity coordinate system matches ROS (typically Unity Z = ROS Y)
- Use fixed timesteps for physics simulation consistency
- Implement proper time synchronization between Unity and ROS

### 3. Integration Best Practices

- Use appropriate data types and units (meters, radians, etc.)
- Implement proper error handling for ROS connections
- Include sensor noise models to match real-world behavior
- Validate simulation results against physical robot behavior

## Troubleshooting Common Issues

### 1. Coordinate System Mismatches

```csharp
// Correct coordinate system conversion
Vector3 RosToUnity(geometry_msgs.Point rosPoint)
{
    return new Vector3(
        (float)rosPoint.x,
        (float)rosPoint.z, // ROS Z -> Unity Y
        (float)rosPoint.y  // ROS Y -> Unity Z
    );
}

geometry_msgs.Point UnityToRos(Vector3 unityPoint)
{
    var rosPoint = new geometry_msgs.Point();
    rosPoint.x = unityPoint.x;
    rosPoint.y = unityPoint.z; // Unity Z -> ROS Y
    rosPoint.z = unityPoint.y; // Unity Y -> ROS Z
    return rosPoint;
}
```

### 2. Performance Issues

- Reduce the number of active physics objects
- Use simplified collision meshes
- Implement Level of Detail (LOD) systems
- Use occlusion culling for large environments

### 3. Network Communication Issues

```csharp
// Implement connection health checks
void CheckConnection()
{
    if (ros == null || !ros.IsConnected())
    {
        Debug.LogWarning("ROS connection lost, attempting to reconnect...");
        ros.Initialize(rosIP, rosPort);
    }
}
```

## Summary

In this chapter, you've learned how to create high-fidelity simulation environments using Unity for robotics applications. You've understood how to set up Unity for robotics, create realistic environments with advanced lighting and materials, integrate Unity with ROS 2 for bidirectional communication, and implement sensor simulation in Unity environments.

Unity complements Gazebo's physics capabilities with advanced visualization and rendering features, making it ideal for creating immersive digital twin environments. The combination of accurate physics simulation (Gazebo) and high-fidelity visualization (Unity) provides a complete simulation solution for robotics development and testing.

## Exercises

1. **Unity Environment**: Create a Unity scene with a realistic indoor environment (rooms, furniture, lighting) and integrate it with ROS 2.

2. **Sensor Simulation**: Implement a complete sensor suite in Unity including camera, LiDAR, and IMU sensors with proper ROS message publishing.

3. **Multi-Robot Simulation**: Create a scene with multiple robots navigating in the same environment with collision avoidance.

## Next Steps

In the next chapter, you'll learn about sensor simulation (LiDAR, depth, IMU) which will build upon both Gazebo and Unity concepts to create comprehensive sensor models for your digital twin environments.