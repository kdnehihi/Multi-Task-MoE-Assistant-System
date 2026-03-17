from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PREVIEW_DIR = PROJECT_ROOT / "data" / "processed" / "inspection_previews"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect sampled parquet datasets.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to print and export preview images for each dataset.",
    )
    return parser.parse_args()


def dataset_specs() -> Iterable[tuple[str, Path, str, str]]:
    return (
        ("docvqa", RAW_DATA_DIR / "docvqa_sample.parquet", "question", "answers"),
        ("chartqa", RAW_DATA_DIR / "chartqa_sample.parquet", "query", "label"),
    )


def inspect_dataset(name: str, parquet_path: Path, text_key: str, answer_key: str, num_samples: int) -> None:
    print(f"\n=== {name.upper()} ===")
    print(f"Source: {parquet_path}")

    dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
    print(dataset)
    print("Columns:", dataset.column_names)
    print("Features:", dataset.features)

    preview_dir = PREVIEW_DIR / name
    preview_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        print(f"\nSample {idx}")
        print(f"{text_key}: {sample[text_key]}")
        print(f"{answer_key}: {sample[answer_key]}")

        image = sample["image"]
        image_path = preview_dir / f"{name}_{idx}.png"
        image.save(image_path)
        print(f"image_preview: {image_path}")


def main() -> None:
    args = parse_args()

    for name, parquet_path, text_key, answer_key in dataset_specs():
        if not parquet_path.exists():
            print(f"\nSkipping {name}: file not found at {parquet_path}")
            continue

        inspect_dataset(name, parquet_path, text_key, answer_key, args.num_samples)


if __name__ == "__main__":
    main()
