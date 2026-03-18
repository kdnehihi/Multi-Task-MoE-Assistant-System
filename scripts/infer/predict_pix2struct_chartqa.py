from pathlib import Path

import torch
from datasets import load_dataset
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "outputs" / "pix2struct_chartqa_baseline" / "final_model"
DATA_PATH = PROCESSED_DIR / "multitask_dataset.parquet"

TARGET_TASK = "chartqa"
MAX_PATCHES = 1024
MAX_GENERATION_LENGTH = 32
NUM_SAMPLES_TO_PRINT = 10


def normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = get_device()
    print(f"Using device: {device}", flush=True)

    print("Loading model and processor...", flush=True)
    processor = Pix2StructProcessor.from_pretrained(str(MODEL_DIR))
    model = Pix2StructForConditionalGeneration.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    print(f"Loading dataset and filtering task={TARGET_TASK}...", flush=True)
    dataset = load_dataset("parquet", data_files=str(DATA_PATH), split="train")
    dataset = dataset.filter(lambda example: example["task"] == TARGET_TASK)

    for idx in range(min(NUM_SAMPLES_TO_PRINT, len(dataset))):
        sample = dataset[idx]
        question = normalize_text(sample["question"])
        answer = normalize_text(sample["answer"])

        inputs = processor(
            images=sample["image"],
            text=question,
            return_tensors="pt",
            max_patches=MAX_PATCHES,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_GENERATION_LENGTH)

        prediction = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        print(f"\nSample {idx}", flush=True)
        print(f"question: {question}", flush=True)
        print(f"target: {answer}", flush=True)
        print(f"prediction: {prediction}", flush=True)


if __name__ == "__main__":
    main()
