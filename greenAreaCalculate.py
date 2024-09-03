import cv2
from matplotlib import pyplot as plt
import numpy as np

def calculate_green_area(mask_path):

    mask = cv2.imread(mask_path,0)  
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    plt.imshow(binary_mask)
    plt.show()

    green_area_pixels = cv2.countNonZero(binary_mask)
    total_area_pixels = mask.shape[0] * mask.shape[1]
    green_area_percentage = green_area_pixels * 100 / total_area_pixels

    print(f"Percentage of green area in the image: {green_area_percentage:.2f}%")

    return green_area_percentage

mask_path = "C:\\Utils\\GreenAreaDetectionWithDrones\\mask_image.jpg"
calculate_green_area(mask_path)
