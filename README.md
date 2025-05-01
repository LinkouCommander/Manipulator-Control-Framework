# Manipulator Control Framework

This is a Python-based control framework for a 10-DOF robotic manipulator equipped with Dynamixel servos, a slider, a webcam, force-sensitive resistors (FSRs), and an IMU sensor. The original purpose of this project is to serve as a real-world environment for training a robotic arm to play ball using reinforcement learning (RL).

## Repository Structure
```graphql
manipulator/
├── arduino/               # Arduino sketches for FSR and slider control
├── module/                # Training scripts and reinforcement learning experiments
├── tmp/                   # Temporary files and logs
├── dxl_handler.py         # Dynamixel motor control module
├── fsr_serial_reader.py   # FSR sensor data acquisition module
├── ble_imu_handler.py     # BLE IMU communication module
├── TODO.txt               # Development notes and pending tasks
└── README.md              # Project documentation
```

## Requirements
### Hardware
* Dynamixel servo
* Force-sensitive resistors
* Slider motor
* BLE-compatible IMU sensor

### Dependencies
* Python 3
* Dynamixel SDK
* pyserial (for Arduino communication)
* bleak (for BLE communication)
* stable-baselines3 (RL algorithm library)
* gymnasium (build RL environments)

```bash
pip install -r requirements.txt
```

## Usage
