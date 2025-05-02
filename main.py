import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import concurrent.futures

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from imutils.video import VideoStream

# Import custom modules for each sensor
from module.cam_module import BallTracker
from module.fsr_slider_module import FSRSerialReader
from module.imu_module import BLEIMUHandler
from module.dxl_module import DXLHandler

class HandEnv(gym.Env):
    DXL_MINIMUM_POSITION_VALUE = 1800
    DXL_MAXIMUM_POSITION_VALUE = 2200
    SLD_MINIMUM_POSITION_VALUE = 80
    SLD_MAXIMUM_POSITION_VALUE = 140

    def __init__(self, render_mode='human'):
        super(HandEnv, self).__init__()

        self.render_mode = render_mode
        # Action space includes 6 motor commands and 1 slider command
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        # Observation space includes 6 joint positions, 1 slider pos, and 3 FSR readings
        self.observation_space = gym.spaces.Box(low=-4.0, high=4.0, shape=(10,), dtype=np.float32)

        self.dxl_ids = [10, 11, 12, 20, 21, 22, 30, 31, 32] # Motor IDs for DXL

        # ==================================
        # initial sensors
        start_init = time.perf_counter() # Measure init time

        self.vs = None
        results = {}

        # Initialize all sensors in parallel using threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                "cam": executor.submit(self.init_cam),
                "fsr": executor.submit(self.init_fsr),
                "dxl": executor.submit(self.init_dxl),
                "imu": executor.submit(self.init_imu)
            }
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as e:
                    raise RuntimeError(e)

        # Save initialized sensor instances
        self.cam = results["cam"]
        self.fsr = results["fsr"]
        self.dxl = results["dxl"]
        self.imu = results["imu"]

        # use serial init method if parallel initialization doesn't work
        '''
        # start camera
        self.vs = None
        self.cam = BallTracker(buffer_size=64, height_threshold=300, alpha=0.2)
        print("[CAM] Camera ready")
        # # start fsr & slider
        self.fsr = FSRSerialReader(port='COM4', baudrate=115200, threshold=50)
        self.fsr.start_collection()
        print("[FSR] FSR Slider ready")
        time.sleep(1)
        self.dxl = DXLHandler(device_name='COM3', baudrate=1000000)
        self.dxl.start_dxl()
        print("[DXL] DXL ready")
        # start imu
        self.imu = BLEIMUHandler()
        self.imu.start_imu()
        print("[IMU] IMU ready")
        '''
        
        end_init = time.perf_counter()
        print("[SYSTEM] All sensors initialized. Continue main program.")
        print(f"[SYSTEM] Total init time: {end_init - start_init:.2f} seconds")
        # ==================================

        # Initialize counters and reward trackers
        self._ij = 0
        self.lifting_rewards = []
        self.rotation_rewards = []
        self.accumulated_rewards = []
        self.episode_rewards = 0

# ==============================================================================================
# Initialization Functions
# ==============================================================================================

    def init_cam(self):
        cam = BallTracker(buffer_size=64, height_threshold=300, alpha=0.2)
        print("[CAM] Camera ready")
        return cam

    def init_fsr(self):
        fsr = FSRSerialReader(port='COM4', baudrate=115200, threshold=50)
        fsr.start_collection()
        print("[FSR] FSR Slider ready")
        return fsr

    def init_dxl(self):
        dxl = DXLHandler(device_name='COM3', baudrate=1000000)
        dxl.start_dxl()
        print("[DXL] DXL ready")
        return dxl

    def init_imu(self):
        imu = BLEIMUHandler()
        imu.start_imu()
        print("[IMU] IMU ready")
        return imu

# ==============================================================================================
# Gym Environment Core Methods
# ==============================================================================================

    def step(self, action):
        """
        Executes one environment step using the given action.

        Args:
            action (list or np.ndarray): A 7-element array where the first 6 values control the DClaw motors 
                                        and the 7th value controls the slider position.

        Returns:
            observation (np.ndarray): The current environment state after the action.
            reward (float): The reward obtained from this step.
            done (bool): Whether the episode has reached a terminal state.
            truncated (bool): Whether the episode was truncated due to hardware or episode limits.
            info (dict): Additional debug or logging information.
        """
        print(f"[Step] >>> {self._ij:2d}")
        truncated = False

        # move slider using 6th value in action
        # self.move_slider(action[6])

        # move dclaw using 0-5th value in action
        idx = [11, 12, 21, 22, 31, 32]
        positions = self.map_array(action[:6], [-1, 1], [self.DXL_MINIMUM_POSITION_VALUE, self.DXL_MAXIMUM_POSITION_VALUE])
        positions = [int(x) for x in positions]
        
        pos_code, obs_pos = self.dxl.move_to_position(idx, positions)
        if pos_code <= 0:
            truncated = True
        
        # Map actual motor positions to [-1, 1]
        obs_pos = self.map_array(obs_pos, [self.DXL_MINIMUM_POSITION_VALUE, self.DXL_MAXIMUM_POSITION_VALUE], [-1, 1])
        force_D0, force_D1, force_D2 = self.fsr.get_fsr()

        # Build observation vector
        observation = [*obs_pos, action[6], force_D0, force_D1, force_D2]
        observation = np.array(observation, dtype=np.float32)

        # Monitor motor temperature
        temperature_list = self.dxl.read_temperature()
        if np.any(np.array(temperature_list) == -1):
            return observation, 0, True, False, {}

        # Reward weights depending on curriculum stage
        if self._ij > 999999:
            rot_weight = 1
            lift_weight = 1
        else:
            rot_weight = 0
            lift_weight = 1

        self.camera_update()

        # Compute rewards from camera and IMU
        lifting_reward = self.cam.get_rewards()
        x_velocity, y_velocity, z_velocity = self.imu.updateIMUData()
        rotation_reward = np.sqrt(x_velocity**2 + y_velocity**2 + z_velocity**2)
        reward = lifting_reward * lift_weight + rotation_reward * rot_weight

        # Store rewards
        self.lifting_rewards.append(lifting_reward)
        self.rotation_rewards.append(rotation_reward)
        self.accumulated_rewards.append(reward)
        self._ij += 1 
        self.episode_rewards += reward

        # Check for termination and truncation
        done = self.check_done()
        truncated = self.check_episode()

#         print(f"""Lift     : {lifting_reward:.3f}
# Rotation : {rotation_reward:.3f}
# FSR : {force_D0:.3f}, {force_D1:.3f}, {force_D2:.3f}
# """)

        self.render()
        info = {}

        return observation, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Returns:
            observation (np.ndarray): The initial observation of the environment.
            info (dict): Additional information about the reset, usually empty.
        """
        print("[Reset]")
        super().reset(seed=seed)  # Pass the seed to the parent class if necessary

        self._ij = 0
        self.episode_rewards = 0

        self.dxl.disable_torque(self.dxl_ids)
        time.sleep(0.5)
        self.dxl.enable_torque(self.dxl_ids)

        # Reset motors and slider
        dxl_code, obs_pos = self.dxl.move_to_position(self.dxl.DXL_IDs, self.dxl.DXL_INIT_POS)
        if dxl_code <= 0:
            raise Exception("[DXL] DXL is stuck or can't read motor position")

        slider_init_pos = 1
        self.move_slider(slider_init_pos)

        # retrieve force values
        force_D0, force_D1, force_D2 = self.fsr.get_fsr()

        # map current motor position to [-1, 1]
        obs_pos_of_six_motor = [obs_pos[i] for i in [1,2,4,5,7,8]]
        obs_pos_of_six_motor = self.map_array(obs_pos_of_six_motor, [self.DXL_MINIMUM_POSITION_VALUE, self.DXL_MAXIMUM_POSITION_VALUE], [-1, 1])
        
        # Build observation vector
        observation = [*obs_pos_of_six_motor, slider_init_pos, force_D0, force_D1, force_D2]
        observation = np.array(observation, dtype=np.float32)

        return observation, {}

    def render(self, mode='human'):
        """
        Renders the current state of the environment using the camera feed.

        Args:
            mode (str): Rendering mode. Default is 'human' for visual display via Webcam.
        """
        self.camera_update()
        frame = self.cam.get_frame()
        cv2.imshow('Camera Output', frame)
        cv2.waitKey(1)

    def close(self):
        """
        Closes the environment and all connected hardware or parallel data streams,
        and present results.
        """
        self.fsr.plot_data()
        if self.dxl is not None:
            self.dxl.stop_dxl()
        if self.fsr is not None:
            self.fsr.stop_collection()
        if self.imu is not None:
            self.imu.stop_imu()
        if self.vs is not None:
            self.vs.release()
        # self.plot_ball_positions()
        # self.plot_accumulated_rewards()
        # print(self.accumulated_rewards)
        

# ==============================================================================================
# helper function
# ==============================================================================================

    def move_slider(self, action):
        """
        Converts a normalized slider action value to an actual motor position and sends the command.

        Args:
            action (float): A value in the range [-1.0, 1.0] representing the slider's desired position.
        """
        slider_position = np.interp(action, [-1.0, 1.0], [self.SLD_MINIMUM_POSITION_VALUE, self.SLD_MAXIMUM_POSITION_VALUE])
        slider_position = str(int(round(slider_position)))
        respond = self.fsr.send_slider_position(slider_position)

    def camera_update(self):
        """
        Updates the camera feed and processes the current frame for ball tracking.
        Initializes the video stream if not already opened.
        """
        if self.vs is None or not self.vs.isOpened():
            self.vs = cv2.VideoCapture(1)
        _, frame = self.vs.read()
        self.cam.track_ball(frame)  # Process the frame with the tracker
    
    def check_done(self):
        """
        Checks whether the episode has met the termination condition based on the reward.

        Returns:
            bool: True if episode is done, False otherwise.
        """
        if self.episode_rewards >= -20:
            return True
        return False
    
    # truncation func
    def check_episode(self):
        """
        Checks whether the episode should be truncated based on the timestep count.

        Returns:
            bool: True if the episode should be truncated, False otherwise.
        """
        if self._ij > 20:
            return True
        return False

    def map_array(self, arr, a, b):
        """
        Maps a value or list of values from range `a` to range `b` linearly.

        Args:
            arr (list or float): Input value(s) to be mapped.
            a (list or tuple): The original range [min, max].
            b (list or tuple): The target range [min, max].

        Returns:
            float or list: Mapped value(s) in the target range.
        """
        # Convert arr to a NumPy array for mathematical operations
        arr = np.array(arr, dtype=np.float64)
        
        # Perform linear mapping calculation
        mapped_arr = (arr - a[0]) * (b[1] - b[0]) / (a[1] - a[0]) + b[0]
        
        # Return a single float if the input is a single value
        if np.isscalar(arr) or isinstance(arr, (int, float)):
            return float(mapped_arr)  # Ensure the return value is a pure Python number
        else:
            return mapped_arr.tolist()  # Convert back to a Python list

    def plot_accumulated_rewards(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.lifting_rewards)), self.lifting_rewards, label='Lifting Reward', color='blue')
        plt.plot(range(len(self.rotation_rewards)), self.rotation_rewards, label='Rotation Reward', color='orange')
        plt.plot(range(len(self.accumulated_rewards)), self.accumulated_rewards, label='Accumulated Reward', color='green')
        plt.title('Accumulated Reward over Time')
        plt.xlabel('Timesteps')
        plt.ylabel('Accumulated Reward')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# ==============================================================================================
# ==============================================================================================


if __name__ == "__main__":
    env = HandEnv(render_mode="human")
    try:
        check_env(env)  # Check if the environment is valid
        print("[SYSTEM] Check environment done")

        # Define the model
        model = PPO('MlpPolicy', env, verbose=1)

        print("[SYSTEM] Start training model")
        # Train the model
        # model.learn(total_timesteps=1_000_000, callback=stop_callback)
        model.learn(total_timesteps=1_000_000)

        # Save the model
        model.save("ppo_hand_env")

        # Load the model
        model = PPO.load("ppo_hand_env")

        obs, _ = env.reset()
        done = False
        
        print("[SYSTEM] Start test the trained model")

        while not done:
            # action, _states = model.predict(obs)  # Let the model decide the action
            action = env.action_space.sample()
            # print("random action:", action)
            obs, reward, done, truncated, _ = env.step(action)
            if truncated:
                env.reset()
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("Interrupt")
    finally:
        env.close()