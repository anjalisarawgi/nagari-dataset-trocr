import cv2
import numpy as np
from PIL import Image
import pytesseract

# Load the image
image_path = "test_f.png"  # Replace with the actual image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 1: Noise Reduction
image_denoised = cv2.fastNlMeansDenoising(image, h=30)

# Step 2: Contrast Enhancement (Histogram Equalization)
image_equalized = cv2.equalizeHist(image_denoised)

# Step 3: Adaptive Thresholding
image_thresh = cv2.adaptiveThreshold(
    image_equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# Save the processed image instead of showing it
processed_image_path = "processed_test_f.png"
cv2.imwrite(processed_image_path, image_thresh)
print(f"Processed image saved at: {processed_image_path}")

 