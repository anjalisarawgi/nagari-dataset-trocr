# import cv2
# import numpy as np
# from PIL import Image

# import cv2
# import numpy as np
# from PIL import Image

# def denoise_and_convert_to_bw(image: Image.Image, threshold: int = 128) -> Image.Image:
#     """
#     Denoise the image using OpenCV and then convert it to a binary (black and white) image.
    
#     Parameters:
#         image (PIL.Image.Image): The input RGB image.
#         threshold (int): The threshold for binarization.
    
#     Returns:
#         PIL.Image.Image: The processed image in RGB format.
#     """
#     # Convert PIL image to NumPy array (RGB)
#     image_np = np.array(image)
#     # Convert RGB to BGR (OpenCV uses BGR)
#     image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
#     # Apply OpenCV denoising
#     denoised_cv = cv2.fastNlMeansDenoisingColored(
#         image_cv, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
#     )
    
#     # Convert the denoised image back to grayscale (for thresholding)
#     gray = cv2.cvtColor(denoised_cv, cv2.COLOR_BGR2GRAY)
    
#     # Apply thresholding to convert to binary (black and white)
#     ret, bw_cv = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
#     # Convert binary image back to 3-channel format (RGB)
#     bw_cv_rgb = cv2.cvtColor(bw_cv, cv2.COLOR_GRAY2RGB)
    
#     # Convert back to PIL Image
#     return Image.fromarray(bw_cv_rgb)


# # Load an image using PIL
# image_path = "test_c.png"
# image = Image.open(image_path)

# # Apply denoising
# denoised_image = denoise_and_convert_to_bw(image, threshold=128)

# # Save the denoised image
# denoised_image.save("denoised_image.jpg")



import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = "test_g.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# invert image
# inverted = cv2.bitwise_not(image)

# gaussian blur
# blurred = cv2.GaussianBlur(image, (5,5), 0)
# median_filtered = cv2.medianBlur(image, 5)
# adaptive thresholding
# _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# _, binary = cv2.threshold(inverted, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# final_output = cv2.bitwise_not(binary)
# Apply adaptive thresholding
result = np.where(image > 200, image, 0).astype(np.uint8)
# smoothed = cv2.GaussianBlur(result, (3, 3), 0)


# removing salt and pepper noise
local_average = cv2.blur(result, (5, 5))
mask = (result < 70 ) &  (local_average > 200)
final_result = result.copy()
final_result[mask] = 255
final_result = cv2.GaussianBlur(final_result, (3, 3), 0)

processed_image = "process_test_g.png"
cv2.imwrite(processed_image, result)