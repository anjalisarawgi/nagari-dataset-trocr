import os
import json
import xml.etree.ElementTree as ET

# Define ALTO XML namespace
NAMESPACE = {'alto': "http://www.loc.gov/standards/alto/ns-v4#"}

# Directory containing ALTO XML files
xml_dir = "data/groundTruth"
image_prefix = "data/images/"
labels = {}

# Process each XML file
for file in os.listdir(xml_dir):
    if file.endswith(".xml"):
        xml_path = os.path.join(xml_dir, file)

        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract image filename safely
        file_name_elem = root.find(".//alto:sourceImageInformation/alto:fileName", NAMESPACE)
        if file_name_elem is not None:
            file_name = file_name_elem.text
        else:
            print(f"Warning: Missing fileName in {file}")
            file_name = file.replace(".xml", ".jpg")  # Fallback: assume filename

        image_path = os.path.join(image_prefix, file_name)

        # Extract text content
        text_lines = root.findall(".//alto:String", NAMESPACE)
        full_text = " ".join([t.attrib.get("CONTENT", "") for t in text_lines]).strip()

        # Ensure we store valid mappings
        if file_name and full_text:
            labels[image_path] = full_text
        else:
            print(f"Skipping {file} due to missing data.")

# Save to labels.json
output_path = os.path.join(xml_dir, "labels.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=4)

print(f"✅ labels.json generated successfully at {output_path}!")