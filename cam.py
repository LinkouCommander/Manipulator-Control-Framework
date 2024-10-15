import cv2
import numpy as np
from collections import deque
from imutils.video import VideoStream

# Define a function to get the red mask
def get_red_mask(hsv_frame):
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
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

    # lower = np.array([0, 0, 200]) # white
    # upper = np.array([180, 25, 255])

    lower = np.array([0, 0, 0]) # black
    upper = np.array([180, 255, 50])
    
    white_mask = cv2.inRange(hsv_frame, lower, upper)
    
    return white_mask

# Initialize video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

# List to store the coordinates of the red ball and white patch
red_coordinates = []
white_coordinates = []

while True:
    # Read a frame from the video capture
    ret, frame = cap.read(1)
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Get the red mask
    red_mask = get_red_mask(hsv_frame)
    
    # Get the white mask
    sticker_mask = get_sticker_mask(hsv_frame)

    # Find contours in the red mask
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize the center of the red ball
    red_ball_center = None
    
    # Draw contours and find the largest one for red
    if red_contours:
        largest_red_contour = max(red_contours, key=cv2.contourArea)

        if cv2.contourArea(largest_red_contour) > 500:
            x, y, w, h = cv2.boundingRect(largest_red_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            red_ball_center = (int(x + w / 2), int(y + h / 2))
            cv2.circle(frame, red_ball_center, 5, (255, 0, 0), -1)
            red_coordinates.append(red_ball_center)
    
    # If the red ball is detected, search for the white patch within its area
    if red_ball_center is not None:
        # Create a region of interest around the red ball
        roi = sticker_mask[y:y+h, x:x+w]

        # Find contours in the white mask within the ROI
        sticker_contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize the center of the white patch
        sticker_patch_center = None
        
        # Draw contours and find the largest white patch
        if sticker_contours:
            # largest_white_contour = max(white_contours, key=cv2.contourArea)

            for sticker_contour in sticker_contours:
                if cv2.contourArea(sticker_contour) > 100:  # Adjust threshold as needed
                    wx, wy, ww, wh = cv2.boundingRect(sticker_contour)
                    # Adjust the coordinates to map back to the original frame
                    wx += x
                    wy += y
                    # cv2.rectangle(frame, (wx, wy), (wx + ww, wy + wh), (255, 255, 255), 2)
                    white_patch_center = (int(wx + ww / 2), int(wy + wh / 2))
                    white_coordinates.append(white_patch_center)
                    cv2.circle(frame, white_patch_center, 5, (255, 0, 255), -1)

    # Display the frame with tracking annotations
    cv2.imshow('Red Ball and White Patch Tracking', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Print the coordinates of the red ball and white patch
print("Coordinates of the red ball in each frame:")
for coord in red_coordinates:
    print(coord)

print("Coordinates of the white patch in each frame:")
for coord in white_coordinates:
    print(coord)
