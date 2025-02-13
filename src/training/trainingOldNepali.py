import os
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments,  DataCollatorForSeq2Seq,  PreTrainedTokenizerBase, set_seed, EarlyStoppingCallback
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Any, Dict, List
import torch
import wandb
from jiwer import wer, cer
from torchvision.utils import save_image

set_seed(42) 

wandb.init(
    project="oldNepali-OCR-Finetuning", 
    name="trocr-nepali-run", 
)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("trocr-nagari-finetune") # encoder-decoder model
dataset = load_dataset("json", data_files={"train": "oldNepaliDataProcessed/labels/labels.json"})
train_dataset = dataset["train"]
# train_dataset = train_dataset.select(range(100))


# image = Image.open("test_a.png").convert("RGB")
# pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]  
# save_image(pixel_values, "processed_image.png")

# split 
split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

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
    output_dir="./trocr-nagari-oldNepali-finetune-4",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    eval_steps = 400,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
    warmup_steps=200,
    num_train_epochs=30, 
    learning_rate=3e-5,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    report_to=["wandb"],
)
# to convert images to stacked tensors and labels to padded tensors -- a format that is ready for training
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

# Set up the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator, 
    tokenizer=processor.feature_extractor,  
    compute_metrics = compute_metrics,
    eval_dataset=eval_dataset,
)



trainer.train()
trainer.save_model("./trocr-nagari-oldNepali-finetune-4")
processor.save_pretrained("./trocr-nagari-oldNepali-finetune-4")

wandb.finish()

