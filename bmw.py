from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import matplotlib.pyplot as plt
import math

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# Initialize tracking points buffer and angle list
pts = deque(maxlen=args["buffer"])

# Initialize the video stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Define color limits and threshold for ball height
height_threshold = 350

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

def get_angular_velocity(degrees):
    # 將角度轉換為弧度
    radians = math.radians(degrees)
    
    # 計算角速度
    interval = 1
    time_seconds = interval / 30
    angular_velocity = radians / time_seconds
    return angular_velocity

def miniDistance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

window_size = 10
velocity_buffer = []
def moving_average(velocity, buffer, window_size):
    if len(buffer) == window_size:
        mean = np.mean(buffer)
        std_dev = np.std(buffer)

        if abs(velocity - mean) > 5 * std_dev:
            buffer.append(buffer[-1])
            velocity = buffer[-1]
        else:
            buffer.append(velocity)
    else:
        buffer.append(velocity)

    # 如果緩衝區超過窗口大小，則移除最舊的元素
    if len(buffer) > window_size:
        buffer.pop(0)
    
    # 計算移動平均
    # avg_velocity = sum(buffer) / len(buffer)
    return velocity

angles = []  # List to store angles for plotting
velocities = []
mv_velocities = []
curr_angle = 0
prev_angle = None

rect_list = []
prev_rect_list = []

counter = 0
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
        M = cv2.moments(c)
        # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        center = (int(x), int(y))
        
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
                # Calculate the white sticker's center and radius
                # ((x_w, y_w), radius_w) = cv2.minEnclosingCircle(mc)
                # M_w = cv2.moments(mc)
                # mark_center = (int(M_w["m10"] / M_w["m00"]), int(M_w["m01"] / M_w["m00"]))  # Calculate the center
                # mark_center_distance = miniDistance(mark_center, center)

                # # Record mark center if within distance and radius threshold
                # if mark_center_distance < radius and radius_w > 10:
                #     area = cv2.contourArea(mc)
                #     if area > largest_area:
                #         # Update largest contour and area if this one is bigger
                #         largest_contour = mc
                #         largest_area = area
                #         largest_radius = radius_w
                #         largest_center = mark_center

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

                # rect_list.append(np.array(rect[0]) - np.array(center))

                if rect_area > largest_area:
                    largest_contour = mc
                    largest_rect = rect

            if largest_contour is not None and len(c) >= 5 and counter % 2 == 0:
                curr_angle = largest_rect[2]
                # ellipse = cv2.fitEllipse(largest_contour)
                # ellipse_center, ellipse_axes, ellipse_angle = ellipse

                # if counter % 10 == 0:
                #     print(ellipse)
                angles.append(curr_angle)  # Store the angle for plotting
                print(curr_angle)

                if prev_angle is not None:
                    velocity = get_angular_velocity(curr_angle - prev_angle)
                    # if abs(velocity) > 10: 
                    #     if velocities:
                    #         velocity = velocities[-1]
                    #     else:
                    #         velocity = 0
                    velocities.append(abs(velocity))

                    # mv_velocity = moving_average(velocity, velocity_buffer, window_size)
                    # mv_velocities.append(mv_velocity)
                prev_angle = curr_angle

            #     ellipse_center = (int(ellipse_center[0]), int(ellipse_center[1]))
            #     major_axis, minor_axis = int(ellipse_axes[0] // 2), int(ellipse_axes[1] // 2)
            num_sections = 4
            section_angle = 360 / num_sections
            for i in range(num_sections):
                start_angle = int(curr_angle + i * section_angle)
                end_angle = int(curr_angle + (i + 1) * section_angle)
                section_color = (0, 0, 0) if i % 2 == 0 else (0, 255, 255)
                cv2.ellipse(frame, center, (int(radius), int(radius)), 0, start_angle, end_angle, section_color, -1)

    counter += 1
    pts.appendleft(center)
    cv2.imshow("Ball Tracking", frame)
    # cv2.imshow('None Red mask', non_red_mask)
    # cv2.imshow('Red mask', mask)

    # mean_velocity = np.mean(velocities)
    # measurement_noise_covariance = np.var(velocities, ddof=1)
    # print("variance (R):", measurement_noise_covariance)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the angle graph
ax1.plot(range(len(angles)), angles, label="Angle")
ax1.set_title("Angle of Ball Over Time")
ax1.set_xlabel("Frame")
ax1.set_ylabel("Angle (degrees)")
ax1.grid(True)
ax1.legend()

ax2.plot(range(len(velocities)), velocities, label="Velocity")
ax2.set_title("Velocity of Ball Over Time")
ax2.set_xlabel("Frame")
ax2.set_ylabel("Velocity (radians/sec)")
ax2.grid(True)
ax2.legend()

# ax2.plot(range(len(mv_velocities)), mv_velocities, label="Moving Average Velocity")
# ax2.set_title("Moving Average Velocity of Ball Over Time")
# ax2.set_xlabel("Frame")
# ax2.set_ylabel("Velocity (radians/sec)")
# ax2.grid(True)
# ax2.legend()

plt.tight_layout()
plt.show()

# Release resources
cv2.destroyAllWindows()
vs.stop()