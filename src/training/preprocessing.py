import cv2
import numpy as np
from PIL import Image

import cv2
import numpy as np
from PIL import Image

def denoise_and_convert_to_bw(image: Image.Image, threshold: int = 128) -> Image.Image:
    """
    Denoise the image using OpenCV and then convert it to a binary (black and white) image.
    
    Parameters:
        image (PIL.Image.Image): The input RGB image.
        threshold (int): The threshold for binarization.
    
    Returns:
        PIL.Image.Image: The processed image in RGB format.
    """
    # Convert PIL image to NumPy array (RGB)
    image_np = np.array(image)
    # Convert RGB to BGR (OpenCV uses BGR)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Apply OpenCV denoising
    denoised_cv = cv2.fastNlMeansDenoisingColored(
        image_cv, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
    )
    
    # Convert the denoised image back to grayscale (for thresholding)
    gray = cv2.cvtColor(denoised_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to convert to binary (black and white)
    # 'ret' is the used threshold (should be the same as the one provided) and 'bw_cv' is the binary image.
    ret, bw_cv = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Convert binary image back to 3-channel format (RGB)
    bw_cv_rgb = cv2.cvtColor(bw_cv, cv2.COLOR_GRAY2RGB)
    
    # Convert back to PIL Image
    return Image.fromarray(bw_cv_rgb)


# Load an image using PIL
image_path = "test_a.png"
image = Image.open(image_path)

# Apply denoising
denoised_image = denoise_and_convert_to_bw(image, threshold=128)

# Save the denoised image
denoised_image.save("denoised_image.jpg")