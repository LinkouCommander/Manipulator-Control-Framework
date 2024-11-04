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
vs = VideoStream(src=1).start()
time.sleep(2.0)

# Define color limits and threshold for ball height
height_threshold = 350

# def get_red_mask(hsv_frame):
#     lower_red1 = np.array([0, 50, 50])
#     upper_red1 = np.array([15, 255, 255])
#     lower_red2 = np.array([170, 50, 50])
#     upper_red2 = np.array([180, 255, 255])
    
#     mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
#     mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    
#     red_mask = cv2.bitwise_or(mask1, mask2)
#     return red_mask

def get_red_mask(hsv_frame):
    # Define HSV range for black
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])  # Adjust upper value based on lighting

    # Create mask for black
    black_mask = cv2.inRange(hsv_frame, lower_black, upper_black)
    return black_mask

interval = 1
def get_angular_velocity(degrees):
    """
    計算角速度 (rad/s)
    
    :param degrees: 角度 (degrees)
    :param time_seconds: 時間 (秒)
    :return: 角速度 (rad/s)
    """
    # 將角度轉換為弧度
    radians = math.radians(degrees)
    
    # 計算角速度
    time_seconds = interval / 30
    angular_velocity = radians / time_seconds
    return angular_velocity

angles = []  # List to store angles for plotting
velocities = []
angle = None
prev_angle = None

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
    mask = get_red_mask(hsv)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        if radius > 40:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            num_sections = 4
            ellipse = cv2.fitEllipse(c)

            if counter % 10 == 0:
                print(ellipse)
            angle = ellipse[2]
            angles.append(angle)  # Store the angle for plotting

            if prev_angle is not None:
                velocity = get_angular_velocity(angle - prev_angle)
                if abs(velocity) > 20: 
                    velocity = 0
                velocities.append(velocity)
            prev_angle = angle

            section_angle = 360 / num_sections
            for i in range(num_sections):
                start_angle = int(angle + i * section_angle)
                end_angle = int(angle + (i + 1) * section_angle)
                section_color = (0, 0, 0) if i % 2 == 0 else (0, 255, 255)
                cv2.ellipse(frame, center, (int(radius), int(radius)), 0, start_angle, end_angle, section_color, -1)

    counter += 1
    pts.appendleft(center)
    cv2.imshow("Ball Tracking", frame)
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

plt.tight_layout()
plt.show()

# Release resources
cv2.destroyAllWindows()
vs.stop()