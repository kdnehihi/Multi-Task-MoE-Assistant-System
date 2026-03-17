from pathlib import Path

import torch
from datasets import load_dataset
from transformers import DonutProcessor, Trainer, TrainingArguments, VisionEncoderDecoderModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_PATH = PROCESSED_DIR / "multitask_dataset.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "donut_docvqa_baseline"

MODEL_NAME = "naver-clova-ix/donut-base-finetuned-docvqa"
TARGET_TASK = "docvqa"
MAX_PROMPT_LENGTH = 96
MAX_ANSWER_LENGTH = 64
VAL_SIZE = 0.1
SEED = 42

BATCH_SIZE = 2
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


def main() -> None:
    device = get_device()
    print(f"Using device: {device}", flush=True)

    print(f"Loading multitask dataset and filtering task={TARGET_TASK}...", flush=True)
    dataset = load_dataset("parquet", data_files=str(DATA_PATH), split="train")
    dataset = dataset.filter(lambda example: example["task"] == TARGET_TASK)
    dataset = dataset.train_test_split(test_size=VAL_SIZE, seed=SEED)

    print(f"Loading processor and model: {MODEL_NAME}", flush=True)
    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.to(device)

    task_start_token = "<s_docvqa>"
    eos_token = processor.tokenizer.eos_token

    def preprocess_batch(batch):
        questions = [normalize_text(question) for question in batch["question"]]
        answers = [normalize_text(answer) for answer in batch["answer"]]

        prompts = [
            f"{task_start_token}<s_question>{question}</s_question><s_answer>"
            for question in questions
        ]

        image_inputs = processor(
            images=batch["image"],
            random_padding=False,
            return_tensors="pt",
        )
        prompt_tokens = processor.tokenizer(
            prompts,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=MAX_PROMPT_LENGTH,
        )

        full_targets = [f"{prompt}{answer}{eos_token}" for prompt, answer in zip(prompts, answers)]
        label_tokens = processor.tokenizer(
            full_targets,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=MAX_PROMPT_LENGTH + MAX_ANSWER_LENGTH,
        )

        labels = build_labels(label_tokens["input_ids"], processor.tokenizer.pad_token_id)
        return {
            "pixel_values": image_inputs["pixel_values"],
            "decoder_input_ids": prompt_tokens["input_ids"],
            "labels": labels,
        }

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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting DocVQA training...", flush=True)
    trainer.train()

    print("Running evaluation...", flush=True)
    metrics = trainer.evaluate()
    print(metrics, flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR / "final_model"))
    processor.save_pretrained(str(OUTPUT_DIR / "final_model"))

    print("Saved model to:", OUTPUT_DIR / "final_model", flush=True)


if __name__ == "__main__":
    main()
