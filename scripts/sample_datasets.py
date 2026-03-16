from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


DOCVQA_DATASET_NAME = "lmms-lab/DocVQA"
DOCVQA_CONFIG_NAME = "DocVQA"
CHARTQA_DATASET_NAME = "HuggingFaceM4/ChartQA"

DOCVQA_SAMPLE_SIZE = 5000
CHARTQA_SAMPLE_SIZE = 3000
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


def get_train_split(dataset_dict: DatasetDict) -> Dataset:
    if "train" in dataset_dict:
        return dataset_dict["train"]

    first_split = next(iter(dataset_dict.keys()))
    return dataset_dict[first_split]


def sample_dataset(dataset: Dataset, sample_size: int) -> Dataset:
    num_samples = min(sample_size, len(dataset))
    shuffled_dataset = dataset.shuffle(seed=RANDOM_SEED)
    return shuffled_dataset.select(range(num_samples))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading DocVQA...")
    docvqa = load_dataset(DOCVQA_DATASET_NAME, DOCVQA_CONFIG_NAME)
    docvqa_train = get_train_split(docvqa)
    docvqa_sample = sample_dataset(docvqa_train, DOCVQA_SAMPLE_SIZE)
    docvqa_output = OUTPUT_DIR / "docvqa_sample.parquet"
    docvqa_sample.to_parquet(str(docvqa_output))
    print(f"Saved DocVQA sample to {docvqa_output}")

    print("Loading ChartQA...")
    chartqa = load_dataset(CHARTQA_DATASET_NAME)
    chartqa_train = get_train_split(chartqa)
    chartqa_sample = sample_dataset(chartqa_train, CHARTQA_SAMPLE_SIZE)
    chartqa_output = OUTPUT_DIR / "chartqa_sample.parquet"
    chartqa_sample.to_parquet(str(chartqa_output))
    print(f"Saved ChartQA sample to {chartqa_output}")

    print("Sampling complete.")
    print(f"DocVQA sample size: {len(docvqa_sample)}")
    print(f"ChartQA sample size: {len(chartqa_sample)}")


if __name__ == "__main__":
    main()
