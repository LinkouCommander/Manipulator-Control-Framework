from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# Define color limits and initialize a buffer for tracking points
# colorLower = (90, 50, 70)  # Lower bound of the "BLUE" ball color in HSV
# colorUpper = (128, 255, 255)  # Upper bound of the "BLUE" ball color in HSV
pts = deque(maxlen=args["buffer"])  # Deque to store tracked points

# Initialize the video stream
vs = VideoStream(src=1).start()  # Start the webcam
time.sleep(2.0)  # Allow the camera to warm up

# Define the threshold for ball height
height_threshold = 350  # Minimum height for the ball to trigger an action

def get_red_mask(hsv_frame):
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    return red_mask

i = 0
# Ball tracking loop
while True:
    # Capture frame from the video source
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame  # Get the frame if from a video source

    if frame is None:
        break  # Break loop if no frame captured

    # Preprocessing: resize, blur, and convert to HSV color space
    frame = imutils.resize(frame, width=800)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create a mask to identify the "BLUE" ball
    # mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = get_red_mask(hsv)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours to track the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None  # Initialize the ball's center

    if len(cnts) > 0:
        # Identify the largest contour to track the ball
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)  # Get circle parameters around the ball
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # Calculate the center of the ball

        # if center[1] is not None and center[1] < height_threshold:
        #     # Draw a line at the specified height and perform an action
        #     cv2.line(frame, (0, height_threshold), (frame.shape[1], height_threshold), (0, 255, 0), 2)
        #     print("Threshold reached")

        if radius > 40:
            # Draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Draw colored sections based on rotation
            num_sections = 4  # You can adjust the number of sections as needed
            ellipse = cv2.fitEllipse(c)

            if i % 10 == 0:
                print(ellipse)
            # angle = ellipse[2]
            # section_angle = 360 / num_sections
            # for i in range(num_sections):
            #     start_angle = int(angle + i * section_angle)
            #     end_angle = int(angle + (i + 1) * section_angle)
            #     section_color = (0, 0, 0) if i % 2 == 0 else (0, 255, 255)
            #     cv2.ellipse(frame, center, (int(radius), int(radius)), 0, start_angle, end_angle, section_color, -1)

    i = i + 1
    pts.appendleft(center)  # Update the tracked points
    cv2.imshow("Ball Tracking", frame)  # Display the frame with tracking info
    key = cv2.waitKey(1) & 0xFF  # Check for user input

    if key == ord("q"):
        break  # Break the loop if 'q' is pressed

# Release resources and close windows
cv2.destroyAllWindows()
vs.stop()