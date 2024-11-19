import gymnasium as gym
import numpy as np
import time
import serial
from dynamixel_sdk import PortHandler, PacketHandler
import cv2
import os
from dynamixel_sdk import *
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

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
DEVICENAME = '/dev/ttyUSB0'
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
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)

        self.dxl_ids = [10, 11, 12, 20, 21, 22, 30, 31, 32]
        self.ser = initialize_serial(port='/dev/ttyACM0', baud_rate=9600)
        self.camera = None

        if not portHandler.openPort():
            print("Failed to open the port")
            self.return_to_initial_state_and_disable_torque()
            quit()
        print("Succeeded to open the port")

        if not portHandler.setBaudRate(BAUDRATE):
            print("Failed to change the baudrate")
            self.return_to_initial_state_and_disable_torque()
            quit()
        print("Succeeded to change the baudrate")

        for id in self.dxl_ids:
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))
            else:
                print(f"Dynamixel ID {id} has been successfully connected")

        self.previous_ball_position = None

        desired_velocity = 65
        for id in self.dxl_ids:
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, id, ADDR_PROFILE_VELOCITY, desired_velocity)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))
            else:
                print(f"Dynamixel ID {id} velocity set successfully")

        self.ball_positions = []
        self.accumulated_rewards = []

    def control_cost(self, action):
        return np.sum(np.square(action))

    def step(self, action):
        self.move_actuators(action[:6])
        self.move_slider(action[6])

        observation = self.capture_camera_image() # add tactile here?
        ball_position = self.get_ball_position(observation)
        reward = self.calculate_reward(ball_position, action)
        done = self.check_done()

         # Determine if episode is done based on task-specific logic
        terminated = self.check_done()  # Changed done to terminated
        truncated = False  # For now, assume no truncation


        info = {}
        self.ball_positions.append(ball_position)
        self.accumulated_rewards.append(reward)

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Pass the seed to the parent class if necessary

        # Implement reset logic here
        self.move_actuators(np.zeros(6))  # Move actuators to initial positions
        self.set_initial_positions([10, 20, 30], 1000)  # Set motors 10, 20, and 30 to position 1000

        # Return initial observation
        observation = self.capture_camera_image()
        return observation, {}


    def render(self, mode='human'):
        if self.render_mode == 'human':
            if self.camera is None or not self.camera.isOpened():
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    print("Failed to open camera")
                    return

            _, frame = self.camera.read()
            ball_position = self.get_ball_position(frame)

            if ball_position is not None:
                x, y = ball_position
                cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            cv2.imshow('Camera Output', frame)
            cv2.waitKey(1)

    def close(self):
        self.return_to_initial_state_and_disable_torque()
        self.ser.close()
        portHandler.closePort()
        self.plot_ball_positions()
        self.plot_accumulated_rewards()

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

    def move_actuators(self, action):
        positions = np.interp(action, [-0.4, 0.4], [DXL_MINIMUM_POSITION_VALUE, DXL_MAXIMUM_POSITION_VALUE])
        goal_positions = [int(pos) for pos in positions]
        ids_to_move = [11, 12, 21, 22, 31, 32]
        for i, id in enumerate(ids_to_move):
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, id, ADDR_GOAL_POSITION, goal_positions[i])
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))

    def set_initial_positions(self, ids, position):
        for id in ids:
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, id, ADDR_GOAL_POSITION, position)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))
            else:
                print(f"Dynamixel ID {id} set to initial position {position}")

    def move_slider(self, action):
        slider_position = np.interp(action, [-1.0, 1.0], [50, 150])
        send_command(self.ser, f"{int(slider_position)}")

    def capture_camera_image(self):
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
        _, frame = self.camera.read()
        return frame

    def get_ball_position(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w // 2
            center_y = y + h // 2
            return (center_x, center_y)
        else:
            return None

    def calculate_reward(self, ball_position, action):
        if ball_position is None:
            return -1
        x, y = ball_position
        distance_to_target = np.sqrt((x - 320)**2 + (y - 240)**2)
        return -distance_to_target / 100.0 

    def calculate_reward_with_rotation(self, ball_position, radians, action):
        if ball_position is None:
            return -1
        x, y = ball_position
        distance_to_target = np.sqrt((x - 320)**2 + (y - 240)**2)
        return -distance_to_target * 0.49 + radians * 0.51

    def check_done(self):
        return False

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

def initialize_serial(port, baud_rate):
    ser = serial.Serial(port, baud_rate, timeout=1)
    return ser

def send_command(ser, command):
    if ser.is_open:
        command = command + "\n"
        ser.write(command.encode())
        ser.flush()

if __name__ == "__main__":
    env = HandEnv(render_mode="human")
    check_env(env)  # Check if the environment is valid

    # Define the model
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the model
    model.learn(total_timesteps=1)

    # Save the model
    model.save("ppo_hand_env")

    # Load the model
    model = PPO.load("ppo_hand_env")

    # Run the trained model
    obs = env.reset()
    total_reward = 0
    num_timesteps = 1
    try:
        for _ in range(num_timesteps):
            action, _states = model.predict(obs)
            print(f"timestep = {_}")
            # time.sleep(0.1)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            env.render()
            if done:
                obs = env.reset()
    except Exception as e:
        print(f"Exception caught: {e}")
        env.move_actuators(np.zeros(6))  # Move actuators to the initial position
        env.close()
    except KeyboardInterrupt:
        env.return_to_initial_state_and_disable_torque()
        print("\nTraining interrupted by user.")
    finally:
        env.close()
        print(f"Total reward after {num_timesteps} timesteps: {total_reward}")  # Output total reward