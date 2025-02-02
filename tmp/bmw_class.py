from collections import deque
import numpy as np
import cv2
import imutils
import time
import matplotlib.pyplot as plt
import math
from scipy.signal import medfilt

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
        self.total_reward_list = []
        self.velocities = []
        self.angles = []

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

    def track_ball(self, frame):
        frame = imutils.resize(frame, width=800)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = self.get_red_mask(hsv)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        lifting_reward = -3

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))

            if radius > 40:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                distance_to_target = abs(y - self.height_threshold)
                lifting_reward = -distance_to_target / 100

        self.lifting_reward_list.append(lifting_reward)
        self.counter += 1

        return frame, lifting_reward

    def get_rewards(self):
        return self.lifting_reward_list[-1]

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