from pathlib import Path

import torch
from datasets import load_dataset
from transformers import BlipForQuestionAnswering, BlipProcessor, Trainer, TrainingArguments

try:
    from peft import LoraConfig, get_peft_model
except ImportError as exc:
    raise ImportError(
        "peft is required for LoRA training. Install it with: pip install peft"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_PATH = PROCESSED_DIR / "multitask_dataset.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "blip_multitask_lora"

MODEL_NAME = "Salesforce/blip-vqa-base"
MAX_QUESTION_LENGTH = 64
MAX_ANSWER_LENGTH = 32
VAL_SIZE = 0.1
SEED = 42

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 4
WEIGHT_DECAY = 0.01

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TARGET_MODULES = ["query", "value"]


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


def main() -> None:
    device = get_device()
    print(f"Using device: {device}", flush=True)

    print("Loading multitask dataset...", flush=True)
    dataset = load_dataset("parquet", data_files=str(DATA_PATH), split="train")
    dataset = dataset.train_test_split(test_size=VAL_SIZE, seed=SEED)

    print(f"Loading processor and model: {MODEL_NAME}", flush=True)
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=TARGET_MODULES,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    def preprocess_batch(batch):
        questions = [normalize_text(question) for question in batch["question"]]
        answers = [normalize_text(answer) for answer in batch["answer"]]

        model_inputs = processor(
            images=batch["image"],
            text=questions,
            padding="max_length",
            truncation=True,
            max_length=MAX_QUESTION_LENGTH,
        )

        answer_tokens = processor.tokenizer(
            answers,
            padding="max_length",
            truncation=True,
            max_length=MAX_ANSWER_LENGTH,
        )

        model_inputs["labels"] = build_labels(answer_tokens["input_ids"], processor.tokenizer.pad_token_id)
        return model_inputs

    print("Tokenizing train split...", flush=True)
    train_dataset = dataset["train"].map(
        preprocess_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print("Tokenizing validation split...", flush=True)
    val_dataset = dataset["test"].map(
        preprocess_batch,
        batched=True,
        remove_columns=dataset["test"].column_names,
    )

    train_dataset.set_format(type="torch")
    val_dataset.set_format(type="torch")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
        use_cpu=device.type != "mps",
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting LoRA training...", flush=True)
    trainer.train()

    print("Running evaluation...", flush=True)
    metrics = trainer.evaluate()
    print(metrics, flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR / "final_adapter"))
    processor.save_pretrained(str(OUTPUT_DIR / "final_adapter"))

    print("Saved LoRA adapter to:", OUTPUT_DIR / "final_adapter", flush=True)


if __name__ == "__main__":
    main()
