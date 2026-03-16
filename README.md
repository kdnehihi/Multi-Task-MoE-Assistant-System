# Multi-Task MoE Assistant System

This project explores building a multimodal assistant with a Mixture-of-Experts (MoE) architecture, where different experts specialize in different task types and the system learns to route inputs to the right experts.

The current implementation focus is on the data pipeline needed for a first multitask vision-language training setup. At this stage, the repository is centered on preparing a unified dataset from document and chart question answering benchmarks before moving on to model training and routing experiments.

## Motivation

Most language and vision-language models are dense: every input is processed by the same parameters regardless of task type.

This project investigates whether a shared backbone with task-specialized experts can:

- encourage functional specialization
- improve modularity
- make routing behavior easier to analyze
- provide a clean research setup for multitask multimodal learning

## Current Scope

The current phase focuses on a multimodal multitask dataset built from:

- `DocVQA` (`lmms-lab/DocVQA`, config: `DocVQA`)
- `ChartQA` (`HuggingFaceM4/ChartQA`)

Both datasets are sampled into smaller subsets for faster experimentation, then normalized into a shared format suitable for downstream MoE-style training.

Target unified sample format:

```python
{
    "task": "docvqa" or "chartqa",
    "image": PIL image,
    "question": "...",
    "answer": "..."
}
```

## Current Status

The project is in the data engineering and research prototyping stage.

What is already implemented:

- dataset download scripts for `DocVQA` and `ChartQA`
- lightweight sampling for rapid experimentation
- inspection utilities for schema and image preview
- notebook-based data exploration
- preprocessing into a unified multitask dataset
- baseline training preprocessing:
  - image resize
  - text normalization
  - tokenizer-based conversion to `input_ids`, `attention_mask`, and `labels`

What is not implemented yet:

- core MoE layer
- routing module
- training loop
- expert utilization tracking
- evaluation pipeline

## Data Pipeline

The current intended workflow is:

1. Download source datasets
2. Sample smaller subsets for quick iteration
3. Inspect data quality and schema
4. Merge both datasets into a single multitask dataset
5. Apply basic preprocessing for training
6. Train a first multitask baseline
7. Introduce MoE routing and expert specialization experiments

## Environment

- Python 3.10
- Conda environment: `moe-assistant`
- Main libraries:
  - `torch`
  - `transformers`
  - `datasets`
  - `pillow`

## Repository Structure

```text
Multi-Task-MoE-Assistant-System
├── README.md
├── resources/
│   ├── DLMulti-Task Learning.pdf
│   ├── MixtralOfExperts.pdf
│   └── SwitchTransformers.pdf
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_data_inspection.ipynb
└── scripts/
    ├── DownloadDataset.py
    ├── InspectDataset.py
    ├── sample_datasets.py
    ├── inspect_sampled_data.py
    ├── preprocess_multitask_dataset.py
    └── prepare_training_dataset.py
```

## Scripts

`scripts/DownloadDataset.py`

- downloads the original HuggingFace datasets

`scripts/sample_datasets.py`

- samples:
  - `5000` examples from `DocVQA`
  - `3000` examples from `ChartQA`
- saves sampled parquet files into `data/raw/`

`scripts/inspect_sampled_data.py`

- loads sampled parquet files
- prints schema and example rows
- exports preview images for manual inspection

`scripts/preprocess_multitask_dataset.py`

- loads sampled datasets
- normalizes both sources into a shared schema
- merges them into a multitask dataset
- saves the merged parquet file into `data/processed/`

`scripts/prepare_training_dataset.py`

- loads the multitask dataset
- normalizes text
- resizes images
- tokenizes question and answer text
- saves a HuggingFace dataset artifact for training

## Notebook

`notebooks/01_data_inspection.ipynb`

- explores the sampled datasets interactively
- visualizes image-question-answer samples
- checks schema, lengths, and quick statistics
- helps prototype preprocessing logic before moving it into Python scripts

## Research Direction

The medium-term plan is:

1. establish a clean multitask multimodal baseline
2. define a shared model backbone
3. introduce MoE feed-forward experts
4. study routing behavior across task types
5. compare dense vs. expert-routed training

## References

This repository is inspired by:

- Switch Transformer (Fedus et al., 2021)
- Mixtral 8x7B Technical Report (Mistral AI, 2023)
- MoME: Mixture of Multimodal Experts for Generalist
Multimodal Large Language Models

The goal is not to reproduce these systems exactly, but to build a smaller research-oriented implementation for understanding multitask expert specialization in multimodal settings.
