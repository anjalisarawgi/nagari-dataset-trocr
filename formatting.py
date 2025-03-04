# import json
# import os

# target_dir = "IIT_HW_Hindi_V1Processed/train_4"

# with open('IIT_HW_Hindi_V1Processed/train_4/labels.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# new_data = {}
# for old_path, label in data.items():
#     filename = old_path.split('/')[-1]
#     new_key = f"IIT_HW_Hindi_V1Processed/train_4/images/{filename}"
#     new_data[new_key] = label

# output_path = os.path.join(target_dir, "processed_labels.json")
# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(new_data, f, ensure_ascii=False, indent=4)

########################################################################################################


# import json
# import os
# import shutil

# directories = [
#     "IIT_HW_Hindi_V1Processed/train_1",
#     "IIT_HW_Hindi_V1Processed/train_2",
#     "IIT_HW_Hindi_V1Processed/train_4"
# ]

# processed_dir = "IIT_HW_Hindi_V1Processed"
# processed_images_dir = os.path.join(processed_dir, "images")

# combined_labels = {}
# counter = 1

# # Loop over each train directory
# for d in directories:
#     label_file = os.path.join(d, "labels.json")
    
#     with open(label_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     for old_path, label in data.items():
#         old_filename = os.path.basename(old_path)
#         old_image_path = os.path.join(d, "images", old_filename)
#         new_filename = f"{counter}.jpg"
#         new_image_path = os.path.join(processed_images_dir, new_filename)

#         shutil.copyfile(old_image_path, new_image_path)
#         new_key = f"IIT_HW_Hindi_V1Processed/processed/images/{new_filename}"
        
#         combined_labels[new_key] = label
#         counter += 1

# output_labels_path = os.path.join(processed_dir, "labels.json")
# with open(output_labels_path, 'w', encoding='utf-8') as f:
#     json.dump(combined_labels, f, ensure_ascii=False, indent=4)

# print(f"Combined {counter - 1} images into '{processed_images_dir}'")
# print(f"Created merged labels file at '{output_labels_path}'")


########################################################################################################

# import json
# import os

# labels_dir = "IIT_HW_Hindi_V1Processed"
# input_path = os.path.join(labels_dir, "labels.json")
# output_path = os.path.join(labels_dir, "labels_processed.json")

# with open(input_path, 'r', encoding='utf-8') as f:
#     data = json.load(f) 
# result = []

# for image_path, text_value in data.items():
#     entry = {
#         "text": text_value,
#         "image_path": image_path
#     }
#     result.append(entry)

# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(result, f, ensure_ascii=False, indent=4)

# print(f"Conversion complete. New labels saved to {output_path}")


########################################################################################################

import json
import os

base_dir = "IIT_HW_Hindi_V1Processed"

input_file = os.path.join(base_dir, "labels.json")
output_file = os.path.join(base_dir, "labels_updated.json")

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

updated_data = []

for entry in data:
    old_path = entry["image_path"]
    new_path = old_path.replace("processed/", "")
    updated_entry = {
        "text": entry["text"],
        "image_path": new_path
    }
    updated_data.append(updated_entry)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=4)

print(f"Updated labels saved to {output_file}")