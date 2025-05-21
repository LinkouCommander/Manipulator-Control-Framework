# Manipulator Control Framework

This is a Python-based control framework for a 10-DOF robotic manipulator equipped with Dynamixel servos, a slider, a webcam, force-sensitive resistors (FSRs), and an IMU sensor. The original purpose of this project is to serve as a real-world environment for training a robotic arm to play ball using reinforcement learning (RL).

## Table of Contents

- [Features](#features)
- [Code Structure](#code-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Control and Learning (`main.py`)](#control-and-learning-mainpy)
- [Dynamixel Servo (`dxl_module.py`)](#dynamixel-servo-dxl_modulepy)
- [IMU (`imu_module.py`)](#imu-imu_modulepy)
- [Slider & FSRs (`fsr_slider_module.py`)](#slider--fsrs-fsr_slider_modulepy)
- [Webcam (`cam_module.py`)](#webcam-cam_modulepy)
- [Possible Improvements](#possible-improvements)

## Features

- **10 DOF Robotic System**: Supports 9 Dynamixel servos and 1 slider motor with position control.

- **FSR Sensor**: Reads analog force data and converts it into binary contact signals based on a configurable threshold.

- **IMU Sensor**: Retrieves angular velocity data from a BLE-connected IMU device, which is used to compute the ball's angular velocity for rotation reward calculation.

- **Webcam**: Detects and tracks a red ball in video frame, computing a lifting reward based on vertical movement.

- **Threaded Data Collection**: Utilizes multithreading for efficient sensor data acquisition without blocking motor control.

- **Data Visualization**: Provides plotting of sensor and reward data using Matplotlib.

- **Modular Design**: Organized into separate modules for each sensor and actuator.

## Code Structure
```graphql
manipulator/
│
├── main.py                # PPO reinforcement learning training script
├── module/                # Modules for interfacing with sensors and actuators
│   ├── cam_module.py
│   ├── fsr_slider_module.py
│   ├── imu_module.py
│   └── dxl_module.py
├── tests/                 # Unit tests and validation scripts
├── arduino/               # Arduino sketches for FSR and slider control
├── README.md              # Project documentation
└── ...
```

## Requirements
### Hardware
- Dynamixel servo
- Force-sensitive resistors
- Slider motor
- WITMOTION WT9011DCL IMU sensor

### Dependencies
- Python 3
- Dynamixel SDK
- pyserial (for Arduino communication)
- bleak (for BLE communication)
- stable-baselines3 (RL algorithm library)
- gymnasium (build RL environments)
- numpy
- opencv-python
- matplotlib
- imutils

## Usage
1. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
2. In `main.py`, allocate the **port numbers** for the DXL handler and FSR/slider's Arduino:
    ```python
    fsr = FSRSerialReader(port='COM4', baudrate=115200, threshold=50)
    dxl = DXLHandler(device_name='COM3', baudrate=1000000)
    ```
3. Turn on the hardware devices:
   - Connect DXL, FSR/Slider and ensure they are supplied with sufficient voltage.
   - Connect the webcam.
   - Turn on the IMU.
4. Run `main.py`
    ```bash
    python main.py
    ```

## Control and Learning (`main.py`)
`HandEnv` class in `main.py` is a custom reinforcement learning environment based on the OpenAI Gymnasium interface, designed for controlling a robotic hand with multiple sensors.
### Main Methods
- `step(action)`: Executes one step, returns observation, reward, done, truncated, info.
- `reset()`: Resets the environment, returns initial observation.
- `render()`: Displays the camera feed.
- `close()`: Closes all sensors and hardware connections.
### Observation and Action Spaces
| Space        | Dim | Description                         | Range    |
|--------------|-----|-------------------------------------|----------|
| Action       | 7   | 6 motor controls + 1 slider         | [-1, 1]  |
| Observation  | 10  | 6 joint positions + 1 slider + 3 FSR| [-4, 4]  |
### Workflow of `step()` Function
1. **Action Execution**:
    - Receives a 7-element action array:
        - **First 6 values**: Motor controls mapped to Dynamixel servo positions using linear interpolation between [-1,1] →[1800][2200]
        - **7th value**: Slider control (currently commented out in code)
    - Interact with hardware:
        - Sends position commands to Dynamixel motors via `dxl.move_to_position()`
        - Sends position commands to slider using `self.move_slider(action[6])`
2. **Hardware Safety Monitor**:
    - Monitors motor temperatures (servo overheated)
    - Monitors movement timeout (claw is stucked)
3. **Observation Construction**: Formatted as 10-dimensional numpy array
    - 6 Dynamixel servos positions (mapped to [-1,1])
    - Slider position (mapped to [-1,1])
    - 3 FSR readings from `fsr.get_fsr()`
4. **Reward Calculation**:
    - Lifting reward: From camera tracking `cam.get_rewards()`
    - Rotation reward: Calculated from IMU angular velocities
        ```
        rotation_reward = np.sqrt(x_velocity**2 + y_velocity**2 + z_velocity**2)
        ```
    - Weighted sum with curriculum learning:
        ```
        reward = lifting_reward * lift_weight + rotation_reward * rot_weight
        ```
5. **Episode Management**:
    - Termination: `check_done()` based on accumulated rewards
    - Truncation: 
        - Step limit (`self._ij` > 20)
        - DXL overheating
        - Hardware failures (DXL stucked)

## Dynamixel Servo (`dxl_module.py` )
This project uses nine Dynamixel servos (IDs: 10–12, 20–22, 30–32) to form a multi-joint robotic system. Each servo is controlled by specifying a goal position in the range 0–4095, where:
- `0` corresponds to minimum angle (0°)
- `4095` corresponds to maximum angle (approximate 360°)

To ensure system safety and hardware longevity, the system integrates:
- **Collision Safety**: The command is considered a timeout if a motor fails to reach its destination within a given time.
- **Overheat Check**: The system monitors motor temperatures using `read_temperature()` to prevent thermal overload.

### Core Functions
The control system of Dynamixel Servo is built using the Dynamixel SDK, following are the important functions:
- **PortHandler**: Handles low-level communication with the servo via a specified port (e.g., `COM4`).
- **PacketHandler**: Manages the communication protocol (here, Protocol 2.0) and handles encoding/decoding of instructions and status packets.
- `enable_torque()` / `disable_torque()`: Turn on/off motor actuation by writing to the torque control register.
- `move_to_position()`: Commands specificed servos to move to target positions.
- `read_positions()` / `read_temperature()`: Read latest motor position and temperature data.

### Usage
```python
from dxl_module import DXLHandler

# Create a DXLHandler instance 
dxl = DXLHandler(device_name='DXL COM port', baudrate=1000000) 
# Initialize the servos, including enabling torque and setting velocity
dxl.start_dxl() 

# Move Dynamixel servos with IDs 10, 11, and 12 to the specified positions.
dxl.move_to_position([10, 11, 12], [1024, 1536, 2048])
# Read the current position and temperature from servos with IDs 10, 11, and 12.
positions = dxl.read_positions([10, 11, 12])
temperatures = dxl.read_temperature([10, 11, 12])

# Close the Dynamixel connection and disable torque
dxl.stop_dxl()
```

## IMU (`imu_module.py`)
The `BLEIMUHandler` class is a major modification made to the original Python script from the manufacturer. It serves as an interface to connect to, disconnect from, and retrieve data from IMU.

### Core Functions
- `scan()`: Performs a continuous search for BLE devices in the vicinity based on the specified target MAC address.
- `start_imu()`: Initiates the scanning process, connects to the target device, and starts a separate thread to handle data acquisition. It ensures that the device is open and ready for communication.
- `stop_imu`: Gracefully closes the connection to the IMU device and joins the data acquisition thread to ensure proper cleanup.
- `updateIMUData()`: Retrieves and processes data from the IMU device. It returns the angular velocity values (AsX, AsY, AsZ) in radians, converting them from degrees to radians.

### Usage
```python
from imu_module import BLEIMUHandler

imu = BLEIMUHandler(target_device="Mac Addr. of IMU")
imu.start_imu()
x_velocity, y_velocity, z_velocity = imu.updateIMUData()
imu.stop_imu()
```

## Slider & FSRs (`fsr_slider_module.py`)
The robot arm uses one slider motor and three Force-Sensitive Resistors (FSRs), all controlled by a single Arduino. 
### Core Functions
Key functions include:
- `start_collection()`: Launches a background thread to continuously collect and threshold digital FSR data from the serial port.
- `get_fsr()`: Retrieves the latest binary readings (0 or 1) from the three FSRs.
- `send_slider_position()`: Sends a position command to the Arduino to control the slider motor. The valid position range is 75 (top position) to 145 (bottom position).
### Usage
```python
from fsr_slider_module import FSRSerialReader

# Create a FSRSerialReader instance
fsr = FSRSerialReader(port='Arduino COM port', threshold=50)
# Start collecting FSR data in background
fsr.start_collection()

# Get the latest FSR data
d0, d1, d2 = get_fsr()
# Sends a slider motor position command.
respond = send_slider_position(100)

# Safe shut down
fsr.stop_collection()
```

## Webcam (`cam_module.py`)
The webcam plays a crucial role in this system by detecting and tracking a red ball in real-time video frames, and computing a lifting reward based on the ball’s vertical movement.

### Workflow for Calculating Lifting Reward
1. **Capture Frame**: Obtain a frame from the camera and apply Gaussian Blur to reduce noise.
2. **Color Filtering**: Use a red color filter to create a mask that highlights red objects in the frame.
3. **Morphological Operations**: Apply Erosion and Dilation to smooth the edges of the colored region in the mask.
4. **Contour Detection**: Use OpenCV functions to detect the largest contours in the red mask, treating them as the red ball. Obtain the position and radius of the detected red ball.
5. **Reward Calculation**: Calculate the distance between the y-coordinate of the ball and the height threshold, and compute the lifting reward based on this distance.

### Color Filtering
Since the hue values of red span both ends of the HSV color space, two separate ranges is defined to filter out orange-tinted red and purple-tinted red markers, improving the accuracy of red object detection.
- **Lower Hue Range** (`lower_red1 = [0, 50, 50]`, `upper_red1 = [15, 255, 255]`)  
  → This range covers orange-tinted red to pure red.

- **Upper Hue Range** (`lower_red2 = [170, 50, 50]`, `upper_red2 = [180, 255, 255]`)  
  → This range covers deep red to purple-tinted red.

By combining these two ranges using `cv2.bitwise_or`, we ensure comprehensive red color detection, capturing variations across different lighting conditions and object surfaces.

### Image Preprocessing

#### Gaussian Blur (`cv2.GaussianBlur`)
Used to reduce high-frequency noise in the original image and eliminate excessive details, making edges smoother.

#### Morphological Operations (`cv2.erode + cv2.dilate`)
Applied after the image has been filtered into a binary mask using the red color filter.  
- **Erosion (`cv2.erode`)** removes small noise along the edges, making object boundaries sharper.  
- **Dilation (`cv2.dilate`)** restores objects that were shrunk due to erosion and fills small holes within color regions.  

### Object Detection
- `cv2.findContours`: Finds the contours in a binary image.

- `cv2.minEnclosingCircle`: Finds the smallest enclosing circle around the largest contour to determine the ball's position `(x, y)` and radius.

- `cv2.circle`: Draws the detected ball's contour and marks its center point.

### Reward Calculating
A horizontal line is drawn in the frame using the `height_threshold` to represent the desired height in the real world that we aim to reach.

The closer the y-coordinate of the ball's center point is to the set `height_threshold`, the higher the lifting reward the robot can achieve.

Currently, the reward range is from -3 to 0:
- The reward is -3 when the ball is **not present in the image**.
- The reward is 0 when the ball is **exactly at the `height_threshold`**.

### Usage
```python
# Create a BallTracker instance and initializes the video stream
cam = BallTracker(buffer_size=64, height_threshold=300, alpha=0.2)
vs = cv2.VideoCapture(1)

_, frame = vs.read()
# Process the frame with the tracker
cam.track_ball(frame)
# Returns the most recent computed lifting reward.
lifting_reward = cam.get_rewards()
# Returns the latest processed frame with rendering
rendered_frame = cam.get_frame()

vs.release()
```

## Possible Improvements
- `model.learn()` needs a callback capable of recognizing episode boundaries (termination occured).
- FSR should  monitor data over a short time window to avoid missing brief fingertip contacts due to timing.
