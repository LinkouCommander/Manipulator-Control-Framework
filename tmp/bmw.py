from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import matplotlib.pyplot as plt
import math
from scipy.signal import medfilt

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# Initialize tracking points buffer and angle list
pts = deque(maxlen=args["buffer"])

# Initialize the video stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

class RealTimeIIR:
    def __init__(self, alpha=0.6):  # alpha 越小越平滑
        self.alpha = alpha
        self.last_output = None

    def process(self, new_value):
        if self.last_output is None:
            self.last_output = new_value
        else:
            self.last_output = self.alpha * new_value + (1 - self.alpha) * self.last_output
        return self.last_output

# Define color limits and threshold for ball height
height_threshold = 300

def get_red_mask(hsv_frame):
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    
    red_mask = cv2.bitwise_or(mask1, mask2)
    return red_mask

def get_non_red_mask(hsv_frame):
    # 取得紅色遮罩
    red_mask = get_red_mask(hsv_frame)
    
    # 反轉紅色遮罩，得到非紅色區域
    non_red_mask = cv2.bitwise_not(red_mask)
    return non_red_mask

def apply_circular_mask(mask, center, radius):
    # 建立一個與 mask 相同大小的黑色遮罩
    circular_mask = np.zeros_like(mask)
    
    # 在遮罩上繪製一個白色圓形，僅限於紅色球的範圍
    cv2.circle(circular_mask, center, int(radius), 255, -1)
    
    # 將圓形遮罩應用於非紅色遮罩上，僅保留圓形範圍內的區域
    masked_result = cv2.bitwise_and(mask, circular_mask)
    return masked_result

def get_mark_mask(hsv_frame):
    # Define HSV range for black
    # lower = np.array([0, 0, 0])
    # upper = np.array([180, 255, 50])  # Adjust upper value based on lighting

    lower = np.array([0, 0, 155])
    upper = np.array([180, 30, 255])

    # Create mask for black
    mask = cv2.inRange(hsv_frame, lower, upper)
    return mask

interval = 1
def get_angular_velocity(degrees):
    # 將角度轉換為弧度
    radians = math.radians(degrees)
    
    # 計算角速度
    time_seconds = interval / 30
    angular_velocity = radians / time_seconds
    return angular_velocity

def miniDistance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

filter = RealTimeIIR(alpha=0.2)

angles = []  # List to store angles for plotting
velocities = []
velocity = 0
mv_velocities = []
curr_angle = 0
prev_angle = None

rect_list = []
prev_rect_list = []

counter = 0
zero_counter = 0

lifting_reward = 0
lifting_reward_list = []
# rotation_reward = 0
total_reward_list = []
# Ball tracking loop
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break

    frame = imutils.resize(frame, width=800)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create a mask to identify the red ball
    mask = get_red_mask(hsv)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Create a mask to identify the mark
    mark_mask = get_mark_mask(hsv)
    mark_mask = cv2.erode(mark_mask, None, iterations=2)
    mark_mask = cv2.dilate(mark_mask, None, iterations=2)

    non_red_mask = get_non_red_mask(hsv)
    non_red_mask = cv2.erode(non_red_mask, None, iterations=2)
    non_red_mask = cv2.dilate(non_red_mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        # M = cv2.moments(c)
        center = (int(x), int(y))

        distance_to_target = abs(y - height_threshold)
        lifting_reward = distance_to_target / 100
        lifting_reward_list.append(lifting_reward)
        
        if radius > 40:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            non_red_mask = apply_circular_mask(non_red_mask, (int(x), int(y)), radius * 0.91)

            # Now find white stickers that are within the red ball's radius
            mark_cnts = cv2.findContours(non_red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mark_cnts = imutils.grab_contours(mark_cnts)

            # Initialize variable to hold the largest contour
            largest_contour = None
            largest_center = None
            largest_area = 0
            largest_radius = 0
            largest_rect = None
            for mc in mark_cnts:
                area = cv2.contourArea(mc)
                if area < 300:  # 忽略面積過小的 contours
                    continue

                rect = cv2.minAreaRect(mc)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # 畫出旋轉矩形

                # 計算矩形面積
                rect_area = cv2.contourArea(box)  # 這是矩形的面積

                # 比例檢查：如果標記輪廓的面積與矩形的面積比率小於 min_area_ratio，則過濾掉
                area_ratio = area / rect_area

                if area_ratio < 0.7:
                    continue  # 面積比例小於閾值，跳過此輪廓

                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                rect_list.append(rect)

                if rect_area > largest_area:
                    largest_contour = mc
                    largest_rect = rect

            if counter % interval == 0:
                if rect_list and prev_rect_list:
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
                    curr_angle = total_angle / len(nearest_points)
                    prev_rect_list = rect_list
                elif rect_list:
                    prev_rect_list = rect_list
                else:
                    prev_rect_list = []

                if abs(velocity) > 10:
                    if velocities:
                        velocity = velocities[-1]
                    else:
                        velocity = 0

                velocity = filter.process(velocity)
                # if curr_angle == 0 and angles and angles[-1] != 0 and zero_counter < :
                #     curr_angle = angles[-1]
                #     velocity = velocities[-1]
                #     zero_counter = counter

                total_reward = abs(velocity) * 0.51 - lifting_reward * 0.49
                total_reward_list.append(total_reward)
                print("Frame", counter, ", angular velocity = ", velocity, ", Lifting Reward = ", lifting_reward, ", total_reward = ", total_reward)

                angles.append(curr_angle)  # 儲存每幀的弧度
                velocities.append(velocity)
                rect_list = []

            # num_sections = 4
            # section_angle = 360 / num_sections
            # for i in range(num_sections):
            #     start_angle = int(curr_angle + i * section_angle)
            #     end_angle = int(curr_angle + (i + 1) * section_angle)
            #     section_color = (0, 0, 0) if i % 2 == 0 else (0, 255, 255)
            #     cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, section_color, -1)

    counter += 1
    pts.appendleft(center)

    cv2.line(frame, (0, height_threshold), (frame.shape[1], height_threshold), (0, 255, 0), 2)

    cv2.imshow("Ball Tracking", frame)
    # cv2.imshow('None Red mask', non_red_mask)
    # cv2.imshow('Red mask', mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# print(velocities)
window_size = 11
filtered_data = medfilt(velocities, kernel_size=window_size)
max_angular_velocity = np.max(abs(filtered_data))
print("Maximum velocity:", max_angular_velocity, "(radians/sec)")
print(velocities)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Plot the angle graph
ax1.plot(range(len(angles)), angles, label="Angle")
ax1.set_title("Angle of Ball Over Time")
ax1.set_xlabel("Frame")
ax1.set_ylabel("Angle (degrees)")
ax1.grid(True)
ax1.legend()

ax2.plot(range(len(velocities)), velocities, label="Velocity")
# plt.plot(range(len(velocities)), filtered_data, label='Median Filter Data', linewidth=2)
ax2.set_title("Velocity of Ball Over Time")
ax2.set_xlabel("Frame")
ax2.set_ylabel("Velocity (radians/sec)")
ax2.grid(True)
ax2.legend()

ax3.plot(range(len(lifting_reward_list)), lifting_reward_list, label="Lifting Reward")
ax3.plot(range(len(total_reward_list)), total_reward_list, label="Total Reward")
ax3.set_title("Lifting Reward Over Time")
ax3.set_xlabel("Frame")
ax3.set_ylabel("Lifting Reward")
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()

# Release resources
cv2.destroyAllWindows()
vs.stop()