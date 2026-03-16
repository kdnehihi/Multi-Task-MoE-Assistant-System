from pathlib import Path

from datasets import concatenate_datasets, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DOCVQA_PATH = RAW_DIR / "docvqa_sample.parquet"
CHARTQA_PATH = RAW_DIR / "chartqa_sample.parquet"
OUTPUT_PATH = PROCESSED_DIR / "multitask_dataset.parquet"


def first_answer(value):
    if isinstance(value, list):
        return value[0] if value else ""
    return value if value is not None else ""


def main() -> None:
    print("Loading sampled DocVQA...")
    docvqa = load_dataset("parquet", data_files=str(DOCVQA_PATH), split="train")

    print("Loading sampled ChartQA...")
    chartqa = load_dataset("parquet", data_files=str(CHARTQA_PATH), split="train")

    print("Normalizing DocVQA schema...")
    docvqa_processed = docvqa.map(
        lambda ex: {
            "task": "docvqa",
            "image": ex["image"],
            "question": ex["question"],
            "answer": first_answer(ex["answers"]),
        },
        remove_columns=docvqa.column_names,
    )

    print("Normalizing ChartQA schema...")
    chartqa_processed = chartqa.map(
        lambda ex: {
            "task": "chartqa",
            "image": ex["image"],
            "question": ex["query"],
            "answer": first_answer(ex["label"]),
        },
        remove_columns=chartqa.column_names,
    )

    print("Merging datasets...")
    multitask_dataset = concatenate_datasets([docvqa_processed, chartqa_processed])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    multitask_dataset.to_parquet(str(OUTPUT_PATH))

    print("Saved multitask dataset to:", OUTPUT_PATH)
    print(multitask_dataset)


if __name__ == "__main__":
    main()
