# import json
# import glob

# # Path to JSON files (adjust the pattern if necessary)
# json_files = glob.glob("oldNepaliDataProcessed/labels/*.json")

# combined_data = []

# # Read each JSON file and append its content
# for file in json_files:
#     with open(file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         if isinstance(data, list):  # If the JSON content is a list, extend it
#             combined_data.extend(data)
#         else:  # If it's a dictionary, append as a single entry
#             combined_data.append(data)

# # Save combined data to labels.json
# output_path = "oldNepaliDataProcessed/labels/labels.json"
# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(combined_data, f, indent=4, ensure_ascii=False)

# print(f"Combined JSON saved as {output_path}")


import cv2
import numpy as np
import os

def clean_devanagari_line(image_path, output_path=None, debug=False):
    """
    Cleans a single-line Devanagari text image by isolating
    the main horizontal region (line) and removing partial lines above/below.

    Parameters:
    -----------
    image_path : str
        Path to the input image.
    output_path : str or None
        Where to save the cleaned image. If None, won't save.
    debug : bool
        If True, prints intermediate steps or images for debugging.

    Returns:
    --------
    cleaned : numpy.ndarray
        The final cleaned (binarized) image with extraneous text removed.
    """

    # 1. Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Threshold (binarize)
    #    - Using Otsu's threshold for adaptive binarization
    #    - THRESH_BINARY_INV: foreground (text) becomes white, background black
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 4. (Optional) Morphological opening to remove small specks
    #    - We use a small kernel, so we don't remove actual Devanagari matras (diacritics).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 5. Compute horizontal projection
    #    - Sum of white pixels (255) across each row.
    #    - Convert 255 -> 1 so sums are more convenient
    binary_line = (opened == 255).astype(np.uint8)
    h_proj = np.sum(binary_line, axis=1)

    # 6. Find row range for the main line
    #    - Use a fraction of the maximum horizontal sum as a cutoff.
    threshold_fraction = 0.2
    max_val = np.max(h_proj)
    row_cutoff = threshold_fraction * max_val

    valid_rows = np.where(h_proj > row_cutoff)[0]
    if len(valid_rows) == 0:
        # If no rows pass the threshold, fallback to the entire image
        top_row, bottom_row = 0, gray.shape[0] - 1
    else:
        top_row, bottom_row = valid_rows[0], valid_rows[-1]

    # 7. Create a mask to keep only rows in [top_row, bottom_row]
    line_mask = np.zeros_like(opened)
    line_mask[top_row:bottom_row+1, :] = 255

    # 8. Mask out everything outside that range
    #    - This effectively "crops" but without resizing the image
    #      (so the image dimension stays the same, just black/white outside the line)
    isolated = cv2.bitwise_and(opened, line_mask)

    # 9. (Optional) Invert back if your OCR expects black on white
    cleaned = cv2.bitwise_not(isolated)

    # (Optional) Debugging or saving
    if debug:
        print(f"Top row = {top_row}, Bottom row = {bottom_row}")
        cv2.imshow("Original", img)
        cv2.imshow("Thresholded", opened)
        cv2.imshow("Masked line only", cleaned)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if output_path:
        cv2.imwrite(output_path, cleaned)

    return cleaned


if __name__ == "__main__":
    # Example usage on multiple test images
    # (Assume you have images named 'line1.png', 'line2.png', etc.)
    input_images = [
       "test_images/DNA_0010_0117_i.png",
       "test_images/DNA_0010_0117_ii.png",
       "test_images/DNA_0010_0117_iii.png",
        "test_images/DNA_0010_0117_iv.png",

    ]
    output_dir = "test_images/preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    for i, img_path in enumerate(input_images, start=1):
        out_path = os.path.join(output_dir, f"cleaned_line{i}.png")
        cleaned_image = clean_devanagari_line(img_path, output_path=out_path, debug=False)
        print(f"Saved cleaned image to {out_path}")