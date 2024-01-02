"""
Main file of the baseline model detecting houses and 
swimming pools. 

Command: $ python baseline/main.py
"""

import cv2
import numpy as np

from matplotlib import pyplot as plt
from houses_detection import HousesDetectionBaseline
from pools_detection import ColorPoolDetection


def main(img_path='./data/Satellite test.png'):

    # load image
    image = cv2.imread(img_path)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # houses detection 
    houses_detection = HousesDetectionBaseline(image)
    houses = houses_detection.filter_and_draw_polygons(show=False)

    # pools detection 
    pools_detection = ColorPoolDetection(image)
    pools = pools_detection.filter_and_draw_polygons(show=False)

    # draw houses and pools
    contour_img = image.copy()
    cv2.drawContours(contour_img, houses, -1, (0, 255, 0), 2)
    cv2.drawContours(contour_img, pools, -1, (0, 0, 255), 2)

    plt.figure(figsize=(12, 12))
    plt.imshow(contour_img)
    plt.title('Image with Filtered Polygons')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()

