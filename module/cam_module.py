from collections import deque
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

class BallTracker:
    """
    A class for detecting and tracking a red ball in a video frame, computing a lifting reward based on vertical movement.

    Attributes:
        height_threshold (int): Y-coordinate used as the lifting reference line.
        lifting_reward_list (list): History of lifting rewards.
        velocities (list): Placeholder for future velocity tracking (not currently used).
        angles (list): Placeholder for future angle tracking (not currently used).
        frame (np.ndarray): Last processed frame with overlays.
        lifting_reward (float): Most recent lifting reward.
    """
    def __init__(self, buffer_size=64, height_threshold=300, alpha=0.2):
        self.pts = deque(maxlen=buffer_size)
        self.height_threshold = height_threshold
        self.lifting_reward_list = []
        self.velocities = []
        self.angles = []
        self.frame = None
        self.lifting_reward = 0

    def get_ball_position(self, frame):
        """
        Detect the largest red ball-like contour in the frame.

        Args:
            frame (np.ndarray): The HSV-processed image.

        Returns:
            tuple: (center, radius) of the detected ball, or (None, None) if not found.
        """
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
        """
        Create a mask for detecting red color in HSV space.

        Args:
            hsv_frame (np.ndarray): HSV-converted image.

        Returns:
            np.ndarray: Binary mask isolating red regions.
        """
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

        return cv2.bitwise_or(mask1, mask2)

    def track_ball(self, frame):
        """
        Track the ball in the frame and compute lifting reward based on its vertical position.

        Args:
            frame (np.ndarray): BGR image from camera feed.
        """
        frame = imutils.resize(frame, width=800)
        # Apply Gaussian blur and Convert the frame to HSV color space
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # render of height threshold line
        cv2.line(frame, (0, self.height_threshold), (frame.shape[1], self.height_threshold), (0, 255, 0), 2)
        lifting_reward = -3

        # find available ball object
        center, radius = self.get_ball_position(hsv)
        if center is None or radius is None or radius < 40:
            self.frame = frame
            self.lifting_reward = lifting_reward
            self.lifting_reward_list.append(lifting_reward)
            return

        # render of ball
        cv2.circle(frame, center, int(radius), (0, 0, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # Compute lifting reward based on distance from threshold height
        _, y = center
        distance_to_target = abs(y - self.height_threshold)
        lifting_reward = -distance_to_target / 100

        # Store reward values for tracking
        self.lifting_reward = lifting_reward
        self.lifting_reward_list.append(lifting_reward)

        # Store the processed frame
        self.frame = frame

    def get_rewards(self):
        """
        Get the latest computed lifting reward.

        Returns:
            float: Last reward, or -3 if no reward was computed yet.
        """
        return self.lifting_reward_list[-1] if self.lifting_reward_list else -3
    
    def get_frame(self):
        """
        Get the most recent frame with annotations.

        Returns:
            np.ndarray: The last processed frame with overlays.
        """
        return self.frame

    def plot_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.lifting_reward_list)), self.lifting_reward_list, label="Lifting Reward", color='b')
        plt.title("Lifting Reward Over Time")
        plt.xlabel("Frame")
        plt.ylabel("Lifting Reward")
        plt.grid(True)
        plt.legend()
        plt.show()
