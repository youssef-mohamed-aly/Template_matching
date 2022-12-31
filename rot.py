import cv2
import numpy as np

def rotate_image(image, angle):
    """
    Rotates an image by a given angle.
    """
    rows, cols = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def template_matching(image, template, min_angle, max_angle, angle_step):
    """
    Performs template matching over a range of angles and returns the image with the matched template highlighted.
    """
    # Iterate over the range of angles
    for angle in range(min_angle, max_angle, angle_step):
        # Rotate the template image by the current angle
        rotated_template = rotate_image(template, angle)

        # Apply template matching
        result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)

        # Check if the current match is good enough
        if np.amax(result) > 0.7:  # Change this threshold to suit your needs
            # Get the location of the matched template in the image
            loc = np.where(result >= np.amax(result))

            #The zip function takes the arrays in loc as input and returns an iterator of tuples, where each tuple contains the corresponding elements from each array.
            # The * operator is used to unpack the tuple loc into separate arguments for the zip function. 
            # The [::-1] slice reverses the order of the arrays in loc, so that the zip function returns the tuples in the correct order.
            # For example, suppose loc is a tuple of two arrays [0, 1, 2] and [3, 4, 5]. The zip function would return an iterator of tuples 
            # [(0, 3), (1, 4), (2, 5)], and the for loop would iterate over these tuples and assign them to the pt variable one by one.
            for pt in zip(*loc[::-1]):
                # Draw a rectangle around the matched template
                cv2.rectangle(image, pt, (pt[0] + rotated_template.shape[1], pt[1] + rotated_template.shape[0]), 255, 5)
            return image
    return image

# Load the reference template image and the image to be searched
template = cv2.imread("assets/rot.PNG", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("assets/soccer_practice.jpg", cv2.IMREAD_GRAYSCALE)

# Set the range of angles to search over
min_angle = -180
max_angle = 180
angle_step = 1

# Perform template matching
result = template_matching(image, template, min_angle, max_angle, angle_step)

# Display the image with the matched template highlighted
cv2.imshow('Result', result)
cv2.waitKey(0)
