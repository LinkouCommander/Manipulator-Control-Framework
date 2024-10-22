import cv2
import numpy as np
from collections import deque
from imutils.video import VideoStream
import imutils

# Define a function to get the red mask
def get_red_mask(hsv_frame):
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    return red_mask

# Define a function to get the white mask
def get_sticker_mask(hsv_frame):
    # lower = np.array([35, 100, 100]) # green
    # upper = np.array([85, 255, 255])

    lower = np.array([0, 0, 180]) # white
    upper = np.array([180, 50, 255])

    # lower = np.array([0, 0, 0])  # lower bound for black in HSV
    # upper = np.array([180, 255, 100])
    
    # lower = np.array([0, 100, 50])  # blue
    # upper = np.array([150, 255, 255])

    mask = cv2.inRange(hsv_frame, lower, upper)
    
    return mask

def get_black_mask(hsv_frame):
    # lower = np.array([35, 100, 100]) # green
    # upper = np.array([85, 255, 255])

    # lower = np.array([0, 0, 180]) # white
    # upper = np.array([180, 50, 255])

    lower = np.array([0, 0, 200])  # lower bound for black in HSV
    upper = np.array([180, 30, 255])
    
    # lower = np.array([0, 100, 50])  # blue
    # upper = np.array([150, 255, 255])

    mask = cv2.inRange(hsv_frame, lower, upper)
    
    return mask

# Initialize video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

# List to store the coordinates of the red ball and white patch
red_coordinates = []
prev_white_coordinates = []
i = 0
cumulated_rotation = [0, 0]

while True:
    if i % 10 == 0:
        print("Frame", i, ": ", cumulated_rotation)
        cumulated_rotation = [0, 0]
    i = i + 1

    white_coordinates = []
    # Read a frame from the video capture
    ret, frame = cap.read(1)
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to HSV color space
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # no noise
    # hsv_frame = cv2.morphologyEx(hsv_frame, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # blur_frame = cv2.GaussianBlur(hsv_frame, (5, 5), 0)

    # Get the red mask
    red_mask = get_red_mask(hsv_frame)
    red_mask = cv2.erode(red_mask, None, iterations=2)
    red_mask = cv2.dilate(red_mask, None, iterations=2)
    
    # Get the white mask
    sticker_mask = get_sticker_mask(hsv_frame)
    # black_mask = get_black_mask(hsv_frame)

    # Find contours in the red mask
    red_contours = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours = imutils.grab_contours(red_contours)

    # Initialize the center of the red ball
    red_ball_center = None
    
    # Draw contours and find the largest one for red
    if red_contours:
        largest_red_contour = max(red_contours, key=cv2.contourArea)

        # if cv2.contourArea(largest_red_contour) > 500:
        #     x, y, w, h = cv2.boundingRect(largest_red_contour)
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     red_ball_center = (int(x + w / 2), int(y + h / 2))
        #     cv2.circle(frame, red_ball_center, 5, (255, 0, 0), -1)
        #     red_coordinates.append(red_ball_center)

        # draw a circle bounding box
        ((x, y), radius) = cv2.minEnclosingCircle(largest_red_contour)  # Get circle parameters around the ball
        M = cv2.moments(largest_red_contour)
        red_ball_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # Calculate the center of the ball

        if radius > 40:
            # Draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.circle(frame, red_ball_center, 5, (0, 0, 255), -1)
    
    # If the red ball is detected, search for the white patch within its area
    if red_ball_center is not None:
        # Create a region of interest around the red ball
        # roi = sticker_mask[y:y+h, x:x+w]

        # Create a circular mask for the red ball area
        mask_circle = np.zeros_like(sticker_mask)
        cv2.circle(mask_circle, (int(x), int(y)), int(radius), 255, -1)
        # Apply the circular mask to the white sticker mask
        roi = cv2.bitwise_and(sticker_mask, sticker_mask, mask=mask_circle)

        # Find contours in the white mask within the ROI
        sticker_contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize the center of the white patch
        sticker_patch_center = None
        
        # Draw contours and find the largest white patch
        # if sticker_contours:
        #     # largest_white_contour = max(white_contours, key=cv2.contourArea)

        #     for sticker_contour in sticker_contours:
        #         if cv2.contourArea(sticker_contour) > 300:  # Adjust threshold as needed
        #             wx, wy, ww, wh = cv2.boundingRect(sticker_contour)
        #             # Adjust the coordinates to map back to the original frame
        #             wx += x
        #             wy += y
        #             cv2.rectangle(frame, (wx, wy), (wx + ww, wy + wh), (255, 255, 255), 2)
        #             white_patch_center = (int(wx + ww / 2), int(wy + wh / 2))
        #             white_coordinates.append(white_patch_center)
        #             cv2.circle(frame, white_patch_center, 5, (255, 0, 255), -1)

        #     if len(white_coordinates) > 0:
        #         if len(prev_white_coordinates) > 0:
        #             local_rotation = np.mean(white_coordinates, axis=0) - np.mean(prev_white_coordinates, axis=0)
        #             cumulated_rotation += local_rotation
        #         prev_white_coordinates = white_coordinates
        #     else:
        #         prev_white_coordinates = []

    # Display the frame with tracking annotations
    cv2.imshow('Red Ball and White Patch Tracking', frame)
    cv2.imshow('Red mask', red_mask)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()