import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image
import os

xml_folder = "data/groundTruth"       
image_folder = "data/images"          
output_image_folder = "datasetProcessed/cropped_textlines/images" 
output_json_folder = "datasetProcessed/labels" 
highlighted_folder = "datasetProcessed" 

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_json_folder, exist_ok=True)
os.makedirs(highlighted_folder, exist_ok=True)

padding = 25

def process_image(image_name):
    """
    Process one image (and its corresponding XML file) to extract text lines,
    crop them, draw bounding boxes, and save the metadata as JSON.
    """
    print(f"Processing image: {image_name}")
    xml_path = os.path.join(xml_folder, image_name + ".xml")
    image_path = os.path.join(image_folder, image_name + ".jpg")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    namespace = {'ns': 'http://www.loc.gov/standards/alto/ns-v4#'}

    image = Image.open(image_path).convert("RGB")
    image_cv = np.array(image)
    page_elem = root.find(".//ns:Page", namespace)
    page_width = int(page_elem.get("WIDTH"))
    page_height = int(page_elem.get("HEIGHT"))
    
    if image_cv.shape[1] != page_width or image_cv.shape[0] != page_height:
        image_cv = cv2.resize(image_cv, (page_width, page_height))
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    image_original = image_cv.copy()
    
    textline_counter = 1
    labels = [] 

    def draw_bounding_boxes(image, hpos, vpos, width, baseline_points, padding,
                            color=(0, 255, 0), thickness=2):
        """
        Given a text line’s attributes, compute a bounding box (with padding),
        draw it on the image, and return the bounding box coordinates.
        """
        if baseline_points:
            min_x = min(x for x, _ in baseline_points)
            max_x = max(x for x, _ in baseline_points)
            min_y = min(y for _, y in baseline_points)
            max_y = max(y for _, y in baseline_points)

            new_hpos = min(hpos, min_x)
            new_width = max(width, max_x - min_x)

            # Calculate the bounding box with padding (and ensure the coordinates are within the page)
            x1 = max(0, new_hpos - padding)
            y1 = vpos # y1 = max(0, vpos - padding)
            x2 = min(page_width, new_hpos + new_width + padding)
            y2 = min(page_height, max_y + padding)

            # Draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            return (x1, y1, x2, y2)
        return None

    # Iterate through each TextLine element in the XML
    for textline in root.findall(".//ns:TextLine", namespace):
        hpos = int(float(textline.get("HPOS", 0)))
        vpos = int(float(textline.get("VPOS", 0)))
        width = int(float(textline.get("WIDTH", 0)))
        baseline_attr = textline.get("BASELINE")

        if baseline_attr:
            baseline_values = list(map(int, baseline_attr.split()))
            if len(baseline_values) % 2 == 0:
                baseline_points = [
                    (baseline_values[i], baseline_values[i+1])
                    for i in range(0, len(baseline_values), 2)
                ]

                bbox = draw_bounding_boxes(image_cv, hpos, vpos, width, baseline_points, padding)
                if bbox is None:
                    print(f"Skipping text line {textline_counter} due to invalid bounding box.")
                    textline_counter += 1
                    continue
                x1, y1, x2, y2 = bbox

                cropped_line = image_original[y1:y2, x1:x2]
                if cropped_line.size == 0:
                    print(f"Skipping text line {textline_counter} due to invalid cropped image.")
                else:
                    cropped_filename = os.path.join(
                        output_image_folder, f"{image_name}_textline_{textline_counter}.png"
                    )
                    cv2.imwrite(cropped_filename, cropped_line)
                    print(f"Saved cropped text line: {cropped_filename}")

                strings = textline.findall("ns:String", namespace)
                text_content = " ".join([s.get("CONTENT") for s in strings if s.get("CONTENT")])

                label_info = {
                    "line_number": textline_counter,
                    "text": text_content,
                    "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "cropped_image": cropped_filename
                }
                labels.append(label_info)
                textline_counter += 1
            else:
                print(f"Skipping text line {textline_counter} due to invalid baseline format.")
                textline_counter += 1

    highlighted_image_path = os.path.join(highlighted_folder, f"{image_name}.jpg")
    cv2.imwrite(highlighted_image_path, image_cv)
    print(f"Highlighted image saved at: {highlighted_image_path}")

    json_output_path = os.path.join(output_json_folder, f"{image_name}_labels.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)
    print(f"Labels JSON saved at: {json_output_path}\n")

# Iterate through all XML files in the ground truth folder
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith(".xml"):
        image_name = os.path.splitext(xml_file)[0]
        image_path = os.path.join(image_folder, image_name + ".jpg")
        if os.path.exists(image_path):
            process_image(image_name)
        else:
            print(f"Image for {image_name} not found. Skipping.")

# image_name = "diksita1895_01" 
# process_image(image_name)