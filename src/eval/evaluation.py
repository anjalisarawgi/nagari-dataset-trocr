#!/usr/bin/env python
import argparse
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

def main(args):
    model_dir = args.model_dir
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = Image.open(args.image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("Predicted Text:", predicted_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the TrOCR model on a sample image")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./trocr-nagari-finetune",
        help="Directory where the saved model and processor are located",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the image file for evaluation",
    )
    args = parser.parse_args()
    main(args)