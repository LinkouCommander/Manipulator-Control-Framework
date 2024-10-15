import cv2
import numpy as np

# Function to calculate the average HSV color of an image
def calculate_average_hsv(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded properly
    if image is None:
        print("Error: Could not open or find the image.")
        return None
    
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the average HSV value
    average_hsv = cv2.mean(hsv_image)[:3]  # Ignore the alpha channel if present

    return average_hsv

# Main function
if __name__ == "__main__":
    # Replace 'path_to_your_image.png' with your image file path
    image_path = 'img.png'
    
    # Calculate average HSV
    average_hsv = calculate_average_hsv(image_path)

    if average_hsv is not None:
        print(f"Average HSV Color: H = {average_hsv[0]:.2f}, S = {average_hsv[1]:.2f}, V = {average_hsv[2]:.2f}")
