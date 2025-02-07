from collections import deque
import numpy as np
import cv2
import imutils
import time
import matplotlib.pyplot as plt
import math
from scipy.signal import medfilt
import threading

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

    def track_ball(self, frame):
        frame = imutils.resize(frame, width=800)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        center, radius = self.get_ball_position(hsv)

        lifting_reward = -3
        cv2.line(frame, (0, self.height_threshold), (frame.shape[1], self.height_threshold), (0, 255, 0), 2)

        if center is None:
            return frame, lifting_reward, 0
        x, y = center

        if radius < 40:
            return frame, lifting_reward, 0
        
        cv2.circle(frame, center, int(radius), (0, 0, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

        def get_mark_position(hsv):
            # get non red mask
            red_mask = self.get_red_mask(hsv)
            red_mask = cv2.erode(red_mask, None, iterations=2)
            red_mask = cv2.dilate(red_mask, None, iterations=2)
            non_red_mask = cv2.bitwise_not(red_mask)
            non_red_mask = cv2.erode(non_red_mask, None, iterations=2)
            non_red_mask = cv2.dilate(non_red_mask, None, iterations=2)
            # get circular mask
            circular_mask = np.zeros_like(non_red_mask)
            cv2.circle(circular_mask, center, int(radius), 255, -1)
            circular_mask_1 = cv2.bitwise_and(non_red_mask, circular_mask)
            # find white stickers that are within the red ball's radius
            mark_cnts = cv2.findContours(circular_mask_1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mark_cnts = imutils.grab_contours(mark_cnts)

            # if mark_cnts is None:
            #     return None

            rect_list = []
            
            for mc in mark_cnts:
                area = cv2.contourArea(mc)

                # ignore small contours
                if area < 300:
                    continue
                
                # Compute the minimum bounding rectangle and extract its integer vertices
                rect = cv2.minAreaRect(mc)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Calculate rectangle are
                rect_area = cv2.contourArea(box)

                # Filter out if the ratio of the contour area to the rectangle area is less than min_area_ratio
                area_ratio = area / rect_area
                if area_ratio < 0.7:
                    continue

                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                rect_list.append(rect)

            return rect_list

        rect_list = get_mark_position(hsv)

        def miniDistance(point1, point2):
            return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        def get_angular_velocity(degrees):
            # 將角度轉換為弧度
            radians = math.radians(degrees)
            
            interval = 1
            # 計算角速度
            time_seconds = interval / 30
            angular_velocity = radians / time_seconds
            return angular_velocity

        def calculate_velocity(rect_list, prev_rect_list):
            nearest_points = []
            closest_map = {}

            for curr_point in rect_list:
                min_distance = float('inf')
                closest_point = None

                for prev_point in prev_rect_list:
                    # print(prev_point[0], curr_point[0])
                    dist = miniDistance(prev_point[0], curr_point[0])
                    # if dist > 500:
                    #     continue
                    if dist < min_distance:
                        min_distance = dist
                        closest_point = prev_point

                if closest_point is not None:  # 確保找到了最近的點
                    # 將 closest_point 轉換為 tuple，以便作為鍵
                    closest_point_key = tuple(closest_point)  # 將 closest_point 轉為 tuple
                    
                    if closest_point_key not in closest_map:
                        closest_map[closest_point_key] = (curr_point, min_distance)
                    elif min_distance < closest_map[closest_point_key][1]:
                        closest_map[closest_point_key] = (curr_point, min_distance)

            nearest_points = [(v[0], k) for k, v in closest_map.items()]

            total_angular_velocity = 0
            total_angle = 0
            for (prev_point, curr_point) in nearest_points:
                total_angle += curr_point[2] - prev_point[2]
                total_angular_velocity += get_angular_velocity(curr_point[2] - prev_point[2])

            velocity = total_angular_velocity / len(nearest_points)
            velocity = self.filter.process(velocity)

            return velocity

        if rect_list and self.prev_rect_list:
            velocity = calculate_velocity(rect_list, self.prev_rect_list)
            self.prev_rect_list = rect_list
        elif rect_list:
            self.prev_rect_list = rect_list
            velocity = 0
        else:
            self.prev_rect_list = []
            velocity = 0

        rotation_reward = abs(velocity)

        distance_to_target = abs(y - self.height_threshold)
        lifting_reward = -distance_to_target / 100

        self.lifting_reward_list.append(lifting_reward)
        self.rotation_reward_list.append(rotation_reward)
        self.counter += 1

        # total_reward = self.rotation_reward_list[-1] * 0.51 - self.lifting_reward_list[-1] * 0.49
        # self.total_reward_list.append(total_reward)
        self.frame = frame

        return frame

    def get_rewards(self):
        return self.lifting_reward_list[-1], self.rotation_reward_list[-1]
    
    def get_frame(self):
        return self.frame


    def start_cam(self):
        self.stop_collecting = False
    
    def stop_cam(self):
        self.stop_collecting = True

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
