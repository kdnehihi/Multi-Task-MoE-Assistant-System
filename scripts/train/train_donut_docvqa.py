from pathlib import Path
import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import DonutProcessor, Trainer, TrainingArguments, VisionEncoderDecoderModel, default_data_collator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_PATH = PROCESSED_DIR / "multitask_dataset.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "donut_docvqa_baseline"
CACHE_DIR = Path("/tmp/moe_hf_cache")
TMP_DIR = Path("/tmp/moe_tmp")

MODEL_NAME = "naver-clova-ix/donut-base-finetuned-docvqa"
TARGET_TASK = "docvqa"
MAX_PROMPT_LENGTH = 96
MAX_ANSWER_LENGTH = 64
VAL_SIZE = 0.1
SEED = 42
MAX_SAMPLES = 500

BATCH_SIZE = 1
LEARNING_RATE = 5e-5
NUM_EPOCHS = 2
WEIGHT_DECAY = 0.01


def normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def build_labels(token_ids, pad_token_id):
    return [[-100 if token_id == pad_token_id else token_id for token_id in row] for row in token_ids]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DonutDocVQADataset(Dataset):
    def __init__(self, dataset, processor, task_start_token, max_prompt_length, max_answer_length):
        self.dataset = dataset
        self.processor = processor
        self.task_start_token = task_start_token
        self.max_prompt_length = max_prompt_length
        self.max_answer_length = max_answer_length
        self.eos_token = processor.tokenizer.eos_token
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        question = normalize_text(sample["question"])
        answer = normalize_text(sample["answer"])
        prompt = f"{self.task_start_token}<s_question>{question}</s_question><s_answer>"

        pixel_values = self.processor(
            images=sample["image"],
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        full_target = f"{prompt}{answer}{self.eos_token}"
        labels = self.processor.tokenizer(
            full_target,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_prompt_length + self.max_answer_length,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        labels = torch.where(labels == self.pad_token_id, torch.full_like(labels, -100), labels)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def main() -> None:
    device = get_device()
    print(f"Using device: {device}", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(CACHE_DIR)
    os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "transformers")
    os.environ["TMPDIR"] = str(TMP_DIR)

    print(f"Loading multitask dataset and filtering task={TARGET_TASK}...", flush=True)
    dataset = load_dataset("parquet", data_files=str(DATA_PATH), split="train")
    dataset = dataset.filter(lambda example: example["task"] == TARGET_TASK)
    if MAX_SAMPLES is not None:
        max_samples = min(MAX_SAMPLES, len(dataset))
        dataset = dataset.shuffle(seed=SEED).select(range(max_samples))
        print(f"Using {max_samples} samples for DocVQA baseline.", flush=True)
    dataset = dataset.train_test_split(test_size=VAL_SIZE, seed=SEED)

    print(f"Loading processor and model: {MODEL_NAME}", flush=True)
    processor = DonutProcessor.from_pretrained(MODEL_NAME, use_fast=False, cache_dir=str(CACHE_DIR))
    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        use_safetensors=False,
    )
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_docvqa>")
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.gradient_checkpointing_enable()
    model.to(device)

    task_start_token = "<s_docvqa>"
    print("Building on-the-fly train and validation datasets...", flush=True)
    train_dataset = DonutDocVQADataset(
        dataset["train"],
        processor,
        task_start_token,
        MAX_PROMPT_LENGTH,
        MAX_ANSWER_LENGTH,
    )
    val_dataset = DonutDocVQADataset(
        dataset["test"],
        processor,
        task_start_token,
        MAX_PROMPT_LENGTH,
        MAX_ANSWER_LENGTH,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
        use_cpu=device.type != "mps",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    print("Starting DocVQA training...", flush=True)
    trainer.train()

    print("Running evaluation...", flush=True)
    metrics = trainer.evaluate()
    print(metrics, flush=True)
    print("Checkpoint saving is disabled in this local-safe mode.", flush=True)


if __name__ == "__main__":
    main()
