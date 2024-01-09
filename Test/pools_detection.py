"""
Tools for detecting pools on stellite images
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

class ColorPoolDetection():

    def __init__(self, img, area_threshold=10):
        self.img = img
        self.img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
        self.area_threshold = area_threshold

    def mask_image(self):
        # lower and upper limits for the white color
        lower_limit = np.array([75, 75, 75])
        upper_limit = np.array([255, 255, 255])

        # create a mask for the specified color range
        mask = cv2.inRange(self.img_hsv, lower_limit, upper_limit)

        # Apply the mask to the original image
        self.masked_img = cv2.bitwise_and(self.img_gray, 
                                          self.img_gray, 
                                          mask=mask)
        
        return self
        

    def filter_and_draw_polygons(self, show=True):
        self.mask_image()

        # Find contours in the binary image
        contours, _ = cv2.findContours(self.masked_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image to draw filtered polygons on
        image_with_filtered_polygons = self.img.copy()

        polygons = []

        # Filter and draw polygons based on area threshold
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.area_threshold:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

                polygons.append(approx_polygon)

                # Draw the filtered polygon
                cv2.drawContours(image_with_filtered_polygons, [approx_polygon], -1, (0, 255, 0), 2)

        if show:
            # Display the original and image with filtered polygons using matplotlib
            plt.figure(figsize=(12, 12))
            plt.imshow(image_with_filtered_polygons, cmap="gray")
            plt.title('Image with Filtered Polygons')
            plt.axis('off')
            plt.show()

        else:
            return polygons
        

if __name__ == '__main__':
    img_path = "./data/Sattelite test.png"
    image = cv2.imread(img_path)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    baseline = ColorPoolDetection(image)
    baseline.filter_and_draw_polygons()

