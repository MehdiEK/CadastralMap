"""
Files containing a solution to detect houses given small 
images. 
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


class HousesDetectionBaseline():

    def __init__(self, image, blurred_ksize=(7, 7), closing_ksize=(12, 12), 
                 area_threshold=500, binary_thresh=(150, 200), epsilon=0.02):
        """
        :params image: img
            Must be RGB.
        """
        self.image = image
        self.blurred_ksize = blurred_ksize
        self.closing_ksize = closing_ksize
        self.area_threshold = area_threshold
        self.binary_thresh = binary_thresh
        self.epsilon = epsilon

    def gray_image(self, show=True):
        img = self.image
        # Convert image to gray scale
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if show:
            fig, ax = plt.subplots(1, 2, figsize=(15, 10))
            ax[0].imshow(gray_image, cmap='gray')
            ax[0].set_title('Gray image')
            ax[1].imshow(self.image)
            ax[1].set_title("Original image")
            plt.show()
        else:
            return gray_image

    def blurred_image(self, show=True):
        gray_img = self.gray_image(show=False)
        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray_img, 
                                   self.blurred_ksize, 
                                   0)

        if show:
            fig, ax = plt.subplots(1, 2, figsize=(15, 10))
            ax[0].imshow(blurred, cmap='gray')
            ax[0].set_title('Blurred image')
            ax[1].imshow(self.image)
            ax[1].set_title("Original image")
            plt.show()
        else:
            return blurred

    def closing(self, show=True):
        blurred = self.blurred_image(show=False)
        # Create a rectangular kernel for the opening operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                           self.closing_ksize)

        # Apply opening operation
        closed_image = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

        if show: 
            fig, ax = plt.subplots(1, 2, figsize=(15, 10))
            ax[0].imshow(closed_image, cmap='gray')
            ax[0].set_title('Closed image')
            ax[1].imshow(self.image)
            ax[1].set_title("Original image")
            plt.show()
        else:
            return closed_image

    def filter_and_draw_polygons(self, show=True):
        closed_image = self.closing(show=False)
        min_, max_ = self.binary_thresh

        # Apply thresholding to obtain a binary image
        _, binary_image = cv2.threshold(closed_image, min_, max_, 
                                        cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image to draw filtered polygons on
        image_with_filtered_polygons = self.image.copy()
        polygons = []

        # Filter and draw polygons based on area threshold
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.area_threshold:
                epsilon = self.epsilon * cv2.arcLength(contour, True)
                approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
                polygons.append(approx_polygon)

                # Draw the filtered polygon
                cv2.drawContours(image_with_filtered_polygons, [approx_polygon], -1, (0, 255, 0), 2)

        if show:
            # Display the original and image with filtered polygons using matplotlib
            plt.figure(figsize=(50, 50))

            plt.subplot(1, 2, 2)
            plt.imshow(image_with_filtered_polygons)
            plt.title('Image with Filtered Polygons')
            plt.axis('off')

            plt.show()
        else:
            return polygons


if __name__ == '__main__':
    img_path = "./data/Sattelite test.png"
    image = cv2.imread(img_path)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    baseline = HousesDetectionBaseline(image)
    baseline.filter_and_draw_polygons()


