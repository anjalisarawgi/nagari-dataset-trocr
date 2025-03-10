import os
from datasets import load_dataset, Dataset
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    PreTrainedTokenizerBase, 
    set_seed
)
from PIL import Image
import torch
from typing import Any, Dict, List
import wandb
from jiwer import wer, cer
import json

set_seed(42) 

wandb.init(
    project="oldNepali-OCR-Finetuning", 
    name="oldNepali-nepali-run", 
)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("trocr-nagari-finetune")  # encoder-decoder model

# Load the augmented labels JSON (this file includes original and augmented images)
dataset = load_dataset("json", data_files={"train": "oldNepaliDataCombined/data_augmented/labels_train.json"})
full_labels = list(dataset["train"])
print(f"Total augmented samples: {len(full_labels)}")

# --- Group Split Code ---
# List of all augmentation suffixes (make sure "orig" is included since we copied originals too)
augmentation_suffixes = ["orig", "blur", "rot", "stretch_w", "stretch_h", "morph", "bright", "noisy"]

def get_group_key(filename: str, augmentation_suffixes: List[str]) -> str:
    """
    Given a filename (without extension), if it ends with one of the known augmentation suffixes
    (preceded by an underscore), remove that suffix to recover the original image name.
    """
    for suffix in augmentation_suffixes:
        if filename.endswith("_" + suffix):
            return filename[:-(len(suffix) + 1)]
    return filename

def group_split_labels(labels: List[Dict[str, Any]], augmentation_suffixes: List[str], test_ratio=0.1):
    """
    Groups label entries by their original image (determined by stripping the augmentation suffix),
    then splits the groups sequentially: first 90% groups for training, last 10% for testing.
    For evaluation, only the original image entries (with suffix 'orig') are kept.
    Returns two lists of labels.
    """
    groups = {}
    for label in labels:
        base = os.path.basename(label["image_path"])  # e.g., "DNA_0001_0006_textline_1_orig.png"
        name, ext = os.path.splitext(base)
        group_key = get_group_key(name, augmentation_suffixes)
        groups.setdefault(group_key, []).append(label)
    
    sorted_keys = sorted(groups.keys())
    n = len(sorted_keys)
    n_test = int(n * test_ratio)
    train_keys = sorted_keys[:-n_test]
    test_keys = sorted_keys[-n_test:]
    print("Train Groups (original images):")
    print(train_keys)
    print("Test Groups (original images):")
    print(test_keys)
    train_labels = []
    test_labels = []
    for k in train_keys:
        train_labels.extend(groups[k])
    for k in test_keys:
        # For eval, keep only the original image (suffix "orig")
        for label in groups[k]:
            filename = os.path.splitext(os.path.basename(label["image_path"]))[0]
            if filename.endswith("_orig"):
                test_labels.append(label)
    
    return train_labels, test_labels

# Group-split the augmented labels so that no original image appears in both splits.
# train_labels, test_labels = group_split_labels(full_labels, augmentation_suffixes, test_ratio=0.1)
# print(f"Training groups: {len(train_labels)} samples")
# print(f"Test groups (original images only): {len(test_labels)} samples")

# # Convert lists back to HF Datasets
# train_dataset = Dataset.from_list(train_labels)
# test_dataset = Dataset.from_list(test_labels)# Load training labels (augmented) and test labels (unaugmented) directly
train_dataset = load_dataset("json", data_files={"train": "oldNepaliDataCombined/data_augmented/labels_train.json"})["train"]
test_dataset = load_dataset("json", data_files={"eval": "oldNepaliDataCombined/labels_test.json"})["eval"]

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.max_length = 128

def process_data(example):
    image_path = example["image_path"] 
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]
    
    with processor.as_target_processor():
        labels = processor(example["text"], return_tensors="pt").input_ids[0]
    example["pixel_values"] = pixel_values
    example["labels"] = labels
    return example

train_dataset = train_dataset.map(process_data, remove_columns=train_dataset.column_names)
train_dataset.set_format(type="torch", columns=["pixel_values", "labels"])
eval_dataset = test_dataset.map(process_data, remove_columns=test_dataset.column_names)
eval_dataset.set_format(type="torch", columns=["pixel_values", "labels"])

training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-nagari-oldNepalinew-finetune-dataaug",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    eval_steps=8000,
    save_strategy="steps",
    save_steps=10000,
    logging_steps=100,
    warmup_steps=1000,
    num_train_epochs=15, 
    learning_rate=3e-5,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    report_to=["wandb"],
)

class ImageToTextCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([feature["pixel_values"] for feature in features])
        labels = [feature["labels"] for feature in features]
        padded_labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt"
        )["input_ids"]

        return {
            "pixel_values": pixel_values,
            "labels": padded_labels,
        }

data_collator = ImageToTextCollator(tokenizer=processor.tokenizer)

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    exact_matches = [p.strip() == l.strip() for p, l in zip(pred_str, label_str)]
    accuracy = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0

    corpus_pred = "\n".join(pred_str)
    corpus_label = "\n".join(label_str)
    wer_score = wer(corpus_label, corpus_pred)
    cer_score = cer(corpus_label, corpus_pred)

    return {
        "accuracy": accuracy,
        "wer": wer_score,
        "cer": cer_score,
    }

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./trocr-nagari-oldNepalinew-finetune-dataaug")
processor.save_pretrained("./trocr-nagari-oldNepalinew-finetune-dataaug")

wandb.finish()