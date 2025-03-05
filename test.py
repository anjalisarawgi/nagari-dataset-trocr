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

import os
import shutil

# Specify the path to the 'raw' folder
source_dir = "oldNepaliData_2/raw"

images_dir = os.path.join(source_dir, "images")
xml_dir = os.path.join(source_dir, "ground_truth")

# Create target directories if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(xml_dir, exist_ok=True)

# Loop through all files in the source directory
for file_name in os.listdir(source_dir):
    # Full path to the file
    file_path = os.path.join(source_dir, file_name)
    
    # Check if it's a file (and not a directory)
    if os.path.isfile(file_path):
        # If it ends with .png, move it to images
        if file_name.lower().endswith(".png"):
            shutil.move(file_path, os.path.join(images_dir, file_name))
            print(f"Moved {file_name} to {images_dir}")
        
        # If it ends with .xml, move it to ground_truth
        elif file_name.lower().endswith(".xml"):
            shutil.move(file_path, os.path.join(xml_dir, file_name))
            print(f"Moved {file_name} to {xml_dir}")