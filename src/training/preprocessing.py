# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# image_path = "oldNepaliDataProcessed/cropped_textlines/images/DNA_0001_0006_textline_6.png"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # invert image
# # inverted = cv2.bitwise_not(image)

# # gaussian blur
# # blurred = cv2.GaussianBlur(image, (5,5), 0)
# # median_filtered = cv2.medianBlur(image, 5)
# # adaptive thresholding
# # _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# # _, binary = cv2.threshold(inverted, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# # final_output = cv2.bitwise_not(binary)
# # Apply adaptive thresholding
# result = np.where(image < 200, image, 0).astype(np.uint8)
# # smoothed = cv2.GaussianBlur(result, (3, 3), 0)


# # removing salt and pepper noise
# local_average = cv2.blur(result, (5, 5))
# mask = (result < 70 ) &  (local_average > 200)
# final_result = result.copy()
# final_result[mask] = 255
# final_result = cv2.GaussianBlur(final_result, (3, 3), 0)

# processed_image = "test_images/preprocessed/DNA_0001_0006_textline_6.png"
# cv2.imwrite(processed_image, result)

# import cv2
# import numpy as np

# def clean_image_for_ocr(input_path, output_path=None):
#     """
#     Reads an image, removes horizontal lines, and returns or saves a cleaned version.
#     """
#     # 1. Read image and convert to grayscale
#     img = cv2.imread(input_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 3)
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # 6. Save or return the cleaned image
#     if output_path:
#         cv2.imwrite(output_path, binary)
    
#     return binary

# # Example usage:
# if __name__ == "__main__":
#     cleaned_image = clean_image_for_ocr("oldNepaliDataProcessed/cropped_textlines/images/DNA_0001_0006_textline_6.png", "cleaned_output.jpg")
#     # 'cleaned_image' is now a numpy array you can feed into any OCR tool later on.


import cv2
import numpy as np

_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

# Find the component with the largest width or largest area
largest_idx = None
largest_area = 0

for i in range(1, num_labels):  # label 0 is background
    area = stats[i, cv2.CC_STAT_AREA]
    if area > largest_area:
        largest_area = area
        largest_idx = i

# Create a mask only for the largest component
component_mask = np.zeros_like(thresh)
component_mask[labels == largest_idx] = 255

# If you suspect multiple connected components belong to the same line,
# you can expand the logic to keep all bounding boxes that overlap in the same row range.

# "Cleaned" result
cleaned = cv2.bitwise_not(component_mask)  # invert if needed
cv2.imwrite("cleaned_line.png", cleaned)