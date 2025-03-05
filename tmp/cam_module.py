from collections import deque
import numpy as np
import cv2
import imutils
import time
import matplotlib.pyplot as plt
import math
from scipy.signal import medfilt
import threading
from imutils.video import VideoStream
import argparse

class RealTimeIIR:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.last_output = None

    def process(self, new_value):
        if self.last_output is None:
            self.last_output = new_value
        else:
            self.last_output = self.alpha * new_value + (1 - self.alpha) * self.last_output
        return self.last_output

class BallTracker:
    def __init__(self, buffer_size=64, height_threshold=300, alpha=0.2):
        self.pts = deque(maxlen=buffer_size)
        self.height_threshold = height_threshold
        self.filter = RealTimeIIR(alpha=alpha)
        self.counter = 0
        self.lifting_reward_list = []
        self.rotation_reward_list = []
        self.total_reward_list = []
        self.velocities = []
        self.angles = []
        self.prev_rect_list = []
        self.frame = None
        self.stop_collecting = False
        self.vs = None

        # Argument parsing
        ap = argparse.ArgumentParser()
        ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
        self.args = vars(ap.parse_args())


    def get_ball_position(self, frame):
        mask = self.get_red_mask(frame)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            return (int(x), int(y)), radius
        else:
            return None, None

    def get_red_mask(self, hsv_frame):
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

        return cv2.bitwise_or(mask1, mask2)

    def get_mark_mask(self, hsv_frame):
        lower = np.array([0, 0, 155])
        upper = np.array([180, 30, 255])
        return cv2.inRange(hsv_frame, lower, upper)

    def track_ball(self):
        while not self.stop_collecting:
            frame = self.vs.read()
            frame = frame[1] if self.args.get("video", False) else frame
            if frame is None:
                print("Warning: No frame captured")
                # self.store_rewards(-3, 0, None)
                break

            frame = imutils.resize(frame, width=800)
            # Apply Gaussian blur and Convert the frame to HSV color space
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # render of height threshold line
            cv2.line(frame, (0, self.height_threshold), (frame.shape[1], self.height_threshold), (0, 255, 0), 2)
            lifting_reward = -3

            # find available ball object
            center, radius = self.get_ball_position(hsv)
            if center is None:
                return frame, lifting_reward, 0
            x, y = center

            if radius < 40: # Ignore too small detected objects
                return frame, lifting_reward, 0

            # render of ball
            cv2.circle(frame, center, int(radius), (0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Reward is based on the absolute value of rotation velocity
            rotation_reward = abs(0)

            # Compute lifting reward based on distance from threshold height
            distance_to_target = abs(y - self.height_threshold)
            lifting_reward = -distance_to_target / 100

            # Store reward values for tracking
            self.store_rewards(lifting_reward, rotation_reward, frame)

            # Store the processed frame
            

            # return frame

    def store_rewards(self, lifting_reward, rotation_reward, frame):
        self.lifting_reward_list.append(lifting_reward)
        self.rotation_reward_list.append(rotation_reward)
        self.counter += 1
        self.frame = frame

    def get_rewards(self):
        return self.lifting_reward_list[-1], self.rotation_reward_list[-1]
    
    def get_frame(self):
        return self.frame

    def start_cam(self):
        if self.vs is None:
            self.vs = VideoStream(src=0).start()
            time.sleep(2.0)
        self.stop_collecting = False
        self.cam_thread = threading.Thread(target=self.track_ball)
        self.cam_thread.start()
    
    def stop_cam(self):
        self.stop_collecting = True
        if self.cam_thread:
            self.cam_thread.join()
        self.vs.stop()

    def plot_results(self):
        window_size = 11
        filtered_data = medfilt(self.velocities, kernel_size=window_size)
        max_angular_velocity = np.max(abs(filtered_data))
        print("Maximum velocity:", max_angular_velocity, "(radians/sec)")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
        ax1.plot(range(len(self.angles)), self.angles, label="Angle")
        ax1.set_title("Angle of Ball Over Time")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Angle (degrees)")
        ax1.grid(True)
        ax1.legend()

        ax2.plot(range(len(self.velocities)), self.velocities, label="Velocity")
        ax2.set_title("Velocity of Ball Over Time")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Velocity (radians/sec)")
        ax2.grid(True)
        ax2.legend()

        ax3.plot(range(len(self.lifting_reward_list)), self.lifting_reward_list, label="Lifting Reward")
        ax3.set_title("Lifting Reward Over Time")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Lifting Reward")
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        plt.show()
