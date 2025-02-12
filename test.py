import json
import glob

# Path to JSON files (adjust the pattern if necessary)
json_files = glob.glob("oldNepaliDataProcessed/labels/*.json")

combined_data = []

# Read each JSON file and append its content
for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):  # If the JSON content is a list, extend it
            combined_data.extend(data)
        else:  # If it's a dictionary, append as a single entry
            combined_data.append(data)

# Save combined data to labels.json
output_path = "oldNepaliDataProcessed/labels/labels.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, indent=4, ensure_ascii=False)

print(f"Combined JSON saved as {output_path}")