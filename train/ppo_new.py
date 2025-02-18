import serial
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import asyncio
import bleak

import gymnasium as gym
from dynamixel_sdk import PortHandler, PacketHandler
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from imutils.video import VideoStream

from cam_module import BallTracker
from fsr_slider_module import FSRSerialReader
from imu_module import BLEIMUHandler

# DYNAMIXEL Model definition
MY_DXL = 'X_SERIES'

# Control table address
if MY_DXL == 'X_SERIES' or MY_DXL == 'MX_SERIES':
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    ADDR_PROFILE_VELOCITY = 112
    DXL_MINIMUM_POSITION_VALUE = 1900
    DXL_MAXIMUM_POSITION_VALUE = 2100
    BAUDRATE = 1000000

PROTOCOL_VERSION = 2.0
DEVICENAME = 'COM4'
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
DXL_MOVING_STATUS_THRESHOLD = 20

# Initialize PortHandler instance
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
packetHandler = PacketHandler(PROTOCOL_VERSION)

class HandEnv(gym.Env):
    def __init__(self, render_mode='human'):
        super(HandEnv, self).__init__()

        self.render_mode = render_mode
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        # Preprocessing (grayscale conversion or downscaling) could be applied to speed up training
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)

        self.dxl_ids = [10, 11, 12, 20, 21, 22, 30, 31, 32] # define idx of motors
        self.camera = None

        # initial sensors
        # start camera
        self.cam = BallTracker(buffer_size=64, height_threshold=300, alpha=0.2)
        # start fsr & slider
        self.fsr = FSRSerialReader(port='COM4', baudrate=115200, threshold=50)
        self.fsr.start_collection()
        # start imu
        self.imu = BLEIMUHandler()
        debug_code = self.imu.start_imu()
        if debug_code < 0:
            exit()

        self._ij = 0

        self.accumulated_rewards = []

        if not portHandler.openPort():
            print("Failed to open the port")
            self.return_to_initial_state_and_disable_torque()
            quit()
        # print("Succeeded to open the port")

        if not portHandler.setBaudRate(BAUDRATE):
            print("Failed to change the baudrate")
            self.return_to_initial_state_and_disable_torque()
            quit()
        # print("Succeeded to change the baudrate")

        # Enable torque for all motors in the list
        for id in self.dxl_ids:
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))
            # else:
            #     print(f"Dynamixel ID {id} has been successfully connected")

        # Set the desired velocity for all motors
        desired_velocity = 65
        for id in self.dxl_ids:
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, id, ADDR_PROFILE_VELOCITY, desired_velocity)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))
            # else:
            #     print(f"Dynamixel ID {id} velocity set successfully")

################################################################################################
# main function
################################################################################################

    def step(self, action):
        # move dclaw using 0-5th value in action
        idx = [11, 12, 21, 22, 31, 32]
        self.move_actuators(idx=idx, action=action[:6])
        # move slider using 6th value in action
        self.move_slider(action[6])

        
        ##############################
        # needs to be done
        # define different curriculum 
        if self._ij > 999999:   
            rot_weight = 1
            lift_weight = 1
        else:
            rot_weight = 1 
            lift_weight = 1 
        ##############################

        img = self.capture_camera_image()
        frame = self.cam.track_ball(frame)  # Process the frame with the tracker

        lifting_reward, _ = self.cam.get_rewards()
        _, rotation_reward, __ = self.imu.updateIMUData()
        reward = lifting_reward * lift_weight + rotation_reward * rot_weight

        info = {}

        # retrieve tactile force (FSR) values
        force_D0, force_D1, force_D2 = self.fsr.get_fsr()
        # Combine camera image and FSR data into observation
        observation = {
            "fsr": [force_D0, force_D1, force_D2],  # Use 'vec' for FSR values
            "img": img,  # Include the image in observation
        }

        self.accumulated_rewards.append(reward)
        self._ij += 1 

        ##############################
        # needs to be done
        # define terminate condition
        done = self.check_done()
        ##############################

        return observation, reward, done, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Pass the seed to the parent class if necessary

        # Implement reset logic here
        init_pos = [1024, 1536, 2560, 1024, 1536, 2560, 1024, 1536, 2560]
        self.move_actuators(idx=self.dxl_ids, action=init_pos)  # Move actuators to initial positions

        # Return initial observation
        img = self.capture_camera_image()
        # retrieve force values
        force_D0, force_D1, force_D2 = self.fsr.get_fsr()
        # Combine camera image and FSR data into observation
        observation = {
            "fsr": [force_D0, force_D1, force_D2],  # Use 'vec' for FSR values
            "img": img,  # Include the image in observation
        }

        return observation, {}

    def render(self, mode='human'):
        frame = self.capture_camera_image()
        frame = self.cam.get_frame(frame)

        cv2.imshow('Camera Output', frame)
        cv2.waitKey(1)

    def close(self):
        self.return_to_initial_state_and_disable_torque()
        portHandler.closePort()
        self.fsr.stop_collection()
        self.imu.stop_imu()
        # self.plot_ball_positions()
        self.plot_accumulated_rewards()

################################################################################################
# other definition
################################################################################################

    # move dclaw
    def move_actuators(self, idx, action):
        # map action value to Dynamixel position
        positions = np.interp(action, [-0.4, 0.4], [DXL_MINIMUM_POSITION_VALUE, DXL_MAXIMUM_POSITION_VALUE])
        goal_positions = [int(pos) for pos in positions]

        # Iterate over the motor IDs and their corresponding goal positions
        for i, id in enumerate(idx):
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, id, ADDR_GOAL_POSITION, goal_positions[i])
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))

    # move slider
    def move_slider(self, action):
        slider_position = np.interp(action, [-1.0, 1.0], [75, 145])
        respond = self.fsr.send_slider_position(slider_position)
        print(respond)

    def return_to_initial_state_and_disable_torque(self):
        self.move_actuators(np.zeros(6))
        for id in self.dxl_ids:
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()

    def capture_camera_image(self):
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
        _, frame = self.camera.read()
        return frame
    
    # termination func
    def check_done(self):
        if self._ij >= 1000:
            return True
        return False

    def plot_ball_positions(self):
        ball_positions = [pos[1] if pos is not None else 0 for pos in self.ball_positions]
        timesteps = range(len(ball_positions))

        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, ball_positions, label='Vertical Position')
        plt.gca().invert_yaxis()
        plt.title('Vertical Position of the Ball over Time')
        plt.xlabel('Timesteps')
        plt.ylabel('Vertical Position')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_accumulated_rewards(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.accumulated_rewards)), self.accumulated_rewards, label='Accumulated Reward')
        plt.title('Accumulated Reward over Time')
        plt.xlabel('Timesteps')
        plt.ylabel('Accumulated Reward')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    env = HandEnv(render_mode="human")
    check_env(env)  # Check if the environment is valid

    # Define the model
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the model
    model.learn(total_timesteps=1000)

    # Save the model
    model.save("ppo_hand_env")

    # Load the model
    model = PPO.load("ppo_hand_env")

    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)  # Let the model decide the action
        obs, reward, done, _, _ = env.step(action)
        env.render()  # Render the environment (optional)

    # Close the environment
    env.close()