from pathlib import Path

from datasets import load_dataset
from PIL import Image
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = PROCESSED_DIR / "multitask_dataset.parquet"
OUTPUT_DIR = PROCESSED_DIR / "training_dataset"

TOKENIZER_NAME = "bert-base-uncased"
IMAGE_SIZE = (224, 224)
QUESTION_MAX_LENGTH = 64
ANSWER_MAX_LENGTH = 32


def normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize(IMAGE_SIZE, Image.BILINEAR)


def main() -> None:
    print("Loading multitask dataset...", flush=True)
    dataset = load_dataset("parquet", data_files=str(INPUT_PATH), split="train")

    print(f"Loading tokenizer: {TOKENIZER_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print("Preprocessing dataset for training...", flush=True)

    def preprocess_example(example):
        question = normalize_text(example["question"])
        answer = normalize_text(example["answer"])

        question_tokens = tokenizer(
            question,
            truncation=True,
            padding="max_length",
            max_length=QUESTION_MAX_LENGTH,
        )
        answer_tokens = tokenizer(
            answer,
            truncation=True,
            padding="max_length",
            max_length=ANSWER_MAX_LENGTH,
        )

        return {
            "task": example["task"],
            "image": preprocess_image(example["image"]),
            "question": question,
            "answer": answer,
            "input_ids": question_tokens["input_ids"],
            "attention_mask": question_tokens["attention_mask"],
            "labels": answer_tokens["input_ids"],
        }

    training_dataset = dataset.map(preprocess_example)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving training dataset to {OUTPUT_DIR} ...", flush=True)
    training_dataset.save_to_disk(str(OUTPUT_DIR))

    print("Training preprocessing complete.", flush=True)
    print(training_dataset, flush=True)


if __name__ == "__main__":
    main()
