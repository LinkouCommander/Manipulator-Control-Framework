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

# # Define color limits and initialize a buffer for tracking points
# pts = deque(maxlen=args["buffer"])  # Deque to store tracked points

# Initialize the video stream
vs = VideoStream(src=0).start()  # Start the webcam
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
    lower = np.array([0, 0, 155])
    upper = np.array([180, 30, 255])

    # black
    # lower = np.array([0, 0, 0])
    # upper = np.array([180, 255, 50])
    
    white_mask = cv2.inRange(hsv_frame, lower, upper)
    
    return white_mask

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

    i = i + 1
    # pts.appendleft(center)  # Update the tracked points
    cv2.imshow("Ball and Sticker Tracking", frame)  # Display the frame with tracking info
    cv2.imshow('Red mask', red_mask)  # Show the red mask
    cv2.imshow('White mask', white_mask)  # Show the white mask
    key = cv2.waitKey(1) & 0xFF  # Check for user input

    if key == ord("q"):
        break  # Break the loop if 'q' is pressed

# Release resources and close windows
cv2.destroyAllWindows()
vs.stop()
