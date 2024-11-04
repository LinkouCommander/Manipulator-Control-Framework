from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import math
import matplotlib.pyplot as plt

interval = 5

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# # Define color limits and initialize a buffer for tracking points
# pts = deque(maxlen=args["buffer"])  # Deque to store tracked points

# Initialize the video stream
vs = VideoStream(src=1).start()  # Start the webcam
time.sleep(2.0)  # Allow the camera to warm up

# Define the threshold for ball height
height_threshold = 350  # Minimum height for the ball to trigger an action

# Function to get red mask
def get_red_mask(hsv_frame):
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    return red_mask

# Function to get white mask
def get_white_mask(hsv_frame):
    # white
    # lower = np.array([0, 0, 155])
    # upper = np.array([180, 30, 255])

    # black
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 50])
    
    white_mask = cv2.inRange(hsv_frame, lower, upper)
    
    return white_mask

def get_radians(point1, point2):
    # 計算向量內積
    if np.array_equal(point1, [0, 0]) or np.array_equal(point2, [0, 0]):
        return 0
    
    x1, y1 = point1
    x2, y2 = point2

    dot_product = x1 * x2 + y1 * y2

    # 計算向量的長度（模）
    magnitude_A = math.sqrt(x1**2 + y1**2)
    magnitude_B = math.sqrt(x2**2 + y2**2)

    # 檢查向量模是否為零，防止除以零
    if magnitude_A == 0 or magnitude_B == 0:
        return 0

    # 計算 cos_theta 並限制在 [-1, 1] 之間
    cos_theta = dot_product / (magnitude_A * magnitude_B)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 計算夾角的弧度
    angle_radians = math.acos(cos_theta)    

    return angle_radians

def get_angular_velocity(radians):
    fps = 30
    time_interval = interval / fps

    omega = radians / time_interval

    return omega

def miniDistance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Frame track
red_coordinates = []
white_coordinates = []
prev_white_coordinates = []
i = 0
cumulated_rotation = [0, 0]

radians_list  = []
radians = 0
angular_velocity_list = []
angular_velocity = 0

while True:
    if i % interval == 0:
        if white_coordinates and prev_white_coordinates:
            # x1, y1 = prev_white_coordinates[0] 
            # x2, y2 = white_coordinates[0]

            # # delta_x = x2 - x1
            # # delta_y = y2 - y1
            # # radians = math.atan2(delta_y, delta_x)

            # print(prev_white_coordinates[0], white_coordinates[0])
            # radians = get_radians(prev_white_coordinates[0], white_coordinates[0])
            # angular_velocity = get_angular_velocity(radians)

            # print("Frame", i, "radians:", radians, ", angular velocity = ", angular_velocity)

            # prev_white_coordinates = white_coordinates

            nearest_points = []
            closest_map = {}
            for curr_point in white_coordinates:
                min_distance = float('inf')
                closest_point = None

                for prev_point in prev_white_coordinates:
                    dist = miniDistance(prev_point, curr_point)
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
            for (prev_point, curr_point) in nearest_points:
                print(prev_point, curr_point)
                radians = get_radians(prev_point, curr_point)
                angular_velocity = get_angular_velocity(radians)
                total_angular_velocity += angular_velocity

            average_angular_velocity = total_angular_velocity / len(nearest_points)
            print("Frame", i, "radians:", radians, ", angular velocity = ", average_angular_velocity)
            prev_white_coordinates = white_coordinates
        elif white_coordinates:
            prev_white_coordinates = white_coordinates
        else:
            prev_white_coordinates = []
        radians_list.append(radians)  # 儲存每幀的弧度
        angular_velocity_list.append(angular_velocity)
        # print("Frame", i, "radians:", radians)
        radians = 0
        angular_velocity = 0
        white_coordinates = []

    # Capture frame from the video source
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame  # Get the frame if from a video source

    if frame is None:
        break  # Break loop if no frame captured

    # Preprocessing: resize, blur, and convert to HSV color space
    frame = imutils.resize(frame, width=800)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create a mask to identify the red ball
    red_mask = get_red_mask(hsv)
    red_mask = cv2.erode(red_mask, None, iterations=2)
    red_mask = cv2.dilate(red_mask, None, iterations=2)

    # Create a mask to identify the white sticker
    white_mask = get_white_mask(hsv)
    white_mask = cv2.erode(white_mask, None, iterations=2)
    white_mask = cv2.dilate(white_mask, None, iterations=2)

    # Find contours to track the red ball
    cnts = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None  # Initialize the ball's center

    if len(cnts) > 0:
        # Identify the largest contour to track the red ball
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)  # Get circle parameters around the ball
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # Calculate the center of the ball

        if radius > 40:
            # Draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Now find white stickers that are within the red ball's radius
            white_cnts = cv2.findContours(white_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            white_cnts = imutils.grab_contours(white_cnts)

            for wc in white_cnts:
                # Calculate the white sticker's center and radius
                ((x_w, y_w), radius_w) = cv2.minEnclosingCircle(wc)
                M_w = cv2.moments(wc)
                white_center = (int(M_w["m10"] / M_w["m00"]), int(M_w["m01"] / M_w["m00"]))  # Calculate the center

                # Check if the white sticker is within the red ball
                distance = np.sqrt((white_center[0] - center[0])**2 + (white_center[1] - center[1])**2)
                if distance < radius:
                    # Draw the white sticker only if it's inside the red ball
                    if radius_w > 10:  # Adjust this radius threshold based on your sticker size
                        # cv2.circle(frame, (int(x_w), int(y_w)), int(radius_w), (255, 255, 255), 2)
                        cv2.circle(frame, white_center, 5, (255, 0, 255), -1)
                        if i % interval == 0:
                            white_coordinates.append(np.array(white_center) - np.array(center))

            # if len(white_center) > 0:
            #     if len(prev_white_coordinates) > 0:
            #         local_rotation = np.mean(white_coordinates, axis=0) - np.mean(prev_white_coordinates, axis=0)
            #         cumulated_rotation += local_rotation
            #     prev_white_coordinates = white_center
            # else:
            #     prev_white_coordinates = []
            # if i % 10 == 0:
            #     if len(white_coordinates) > 0:
            #         if len(white_coordinates) == len(prev_white_coordinates):
            #             x1, y1 = prev_white_coordinates[0]  # 前一幀的座標
            #             x2, y2 = white_coordinates[0]       # 當前幀的座標

            #             # 計算Δx 和Δy
            #             delta_x = x2 - x1
            #             delta_y = y2 - y1

            #             # 使用 atan2 計算弧度
            #             radians = math.atan2(delta_y, delta_x)
            #             # print("Frame", i, "radians:", radians)
            #         prev_white_coordinates = white_coordinates
            #     else:
            #         prev_white_coordinates = []
            #         radians = 0

    i = i + 1

    # pts.appendleft(center)  # Update the tracked points
    cv2.imshow("Ball and Sticker Tracking", frame)  # Display the frame with tracking info
    cv2.imshow('Red mask', red_mask)  # Show the red mask
    cv2.imshow('White mask', white_mask)  # Show the white mask
    key = cv2.waitKey(1) & 0xFF  # Check for user input

    if key == ord("q"):
        break  # Break the loop if 'q' is pressed

# # 繪製弧度圖表
# plt.plot(radians_list, label="Radians over frames")
# plt.xlabel("Frame")
# plt.ylabel("Radians")
# plt.title("Radians of white sticker movement")
# plt.legend()
# plt.show()

# 繪製角度和角速度的圖表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 繪製弧度圖表
ax1.plot(radians_list, label="Radians over frames", color='b')
ax1.set_xlabel("Frame intervals")
ax1.set_ylabel("Radians")
ax1.set_title("Radians of White Sticker Movement")
ax1.legend()
ax1.grid()

# 繪製角速度圖表
ax2.plot(angular_velocity_list, label="Angular Velocity (radians/sec)", color='r')
ax2.set_xlabel("Frame intervals")
ax2.set_ylabel("Angular Velocity (radians/sec)")
ax2.set_title("Angular Velocity of White Sticker Movement")
ax2.legend()
ax2.grid()

plt.tight_layout()  # 調整佈局以防止重疊
plt.show()

# Release resources and close windows
cv2.destroyAllWindows()
vs.stop()
