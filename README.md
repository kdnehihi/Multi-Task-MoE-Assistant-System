# Multi-Task MoE Assistant System

Research code for testing multimodal multitask baselines and a first soft-routed Mixture-of-Experts idea on:

- `DocVQA`
- `ChartQA`

The repo is intentionally lightweight and experimental. It is meant to help answer questions like:

- can the data pipeline train end-to-end?
- what do single-task baselines look like?
- what happens if we route by question into multiple visual encoders before answer generation?

This is not a polished framework. It is a compact sandbox for trying baseline and MoE-style ideas quickly.

## Current Scope

The project currently covers three layers:

1. data preparation for a unified multimodal QA dataset
2. single-task baselines for each source dataset
3. Colab notebooks for multitask dense and MoE-style experiments

The shared sample format used across the repo is:

```python
{
    "task": "docvqa" or "chartqa",
    "image": PIL.Image,
    "question": "...",
    "answer": "..."
}
```

## Datasets

The current experiments use:

- `lmms-lab/DocVQA` with config `DocVQA`
- `HuggingFaceM4/ChartQA`

They are sampled and normalized into a shared multitask parquet file for local scripts, while the Colab notebooks can also pull from Hugging Face directly.

## What Is Implemented

- dataset download and inspection scripts
- sampled-data preprocessing into a single multitask parquet file
- single-task `ChartQA` baseline with `Pix2Struct`
- single-task `DocVQA` baseline with `Donut`
- quick inference script for the `ChartQA` baseline
- Colab notebook for single-task training
- Colab notebook for multitask dense baseline
- Colab notebook for a soft-routed two-encoder MoE prototype

## What Is Not Implemented

- a full MoME reproduction
- a production-ready training framework
- a stable evaluation harness with benchmark metrics
- large-scale checkpoint orchestration

## Repository Structure

```text
Multi-Task-MoE-Assistant-System
├── README.md
├── resources/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 02_colab_train_donut_docvqa.ipynb
│   ├── 03_colab_end_to_end_single_task_baseline.ipynb
│   ├── 04_colab_multitask_pix2struct_baseline.ipynb
│   └── 05_colab_multitask_pix2struct_query_router_moe.ipynb
└── scripts/
    ├── data/
    │   ├── DownloadDataset.py
    │   ├── InspectDataset.py
    │   ├── inspect_sampled_data.py
    │   ├── sample_datasets.py
    │   ├── preprocess_multitask_dataset.py
    │   └── prepare_training_dataset.py
    ├── infer/
    │   └── predict_pix2struct_chartqa.py
    └── train/
        ├── train_donut_docvqa.py
        └── train_pix2struct_chartqa.py
```

## Local Workflow

Local is mainly useful for:

- data preparation
- quick inspection
- `ChartQA` baseline runs
- small `DocVQA` smoke tests

Suggested flow:

1. sample datasets
2. inspect samples
3. build the multitask parquet
4. train one of the single-task baselines

### Data Preparation

```bash
python scripts/data/sample_datasets.py
python scripts/data/inspect_sampled_data.py
python scripts/data/preprocess_multitask_dataset.py
```

### Train ChartQA Baseline

```bash
python scripts/train/train_pix2struct_chartqa.py
```

### Inspect ChartQA Predictions

```bash
python scripts/infer/predict_pix2struct_chartqa.py
```

### Train DocVQA Baseline

```bash
python scripts/train/train_donut_docvqa.py
```

Notes:

- `train_donut_docvqa.py` is intentionally configured as a local-safe research script
- it uses a small sample cap and disables checkpoint saving
- for longer or heavier `DocVQA` runs, Colab is the preferred path

## Colab Workflow

For heavier experiments, the notebooks are the main interface.

### 1. Single-Task End-to-End

`notebooks/03_colab_end_to_end_single_task_baseline.ipynb`

Use this when you want one notebook that:

- downloads data
- samples a subset
- preprocesses it
- trains one single-task baseline

Supported modes:

- `TASK = 'docvqa'`
- `TASK = 'chartqa'`

### 2. Dense Multitask Baseline

`notebooks/04_colab_multitask_pix2struct_baseline.ipynb`

This notebook trains a multitask dense baseline using a shared `Pix2Struct` backbone across both `DocVQA` and `ChartQA`.

Use it as the dense baseline to compare against later routing experiments.

### 3. Soft-Routed Two-Encoder MoE Prototype

`notebooks/05_colab_multitask_pix2struct_query_router_moe.ipynb`

This notebook is the current MoE-style experiment.

It uses:

- a router that reads the question
- a softmax over two visual encoders
- weighted fusion of the two encoder outputs
- a shared answer decoder

Current prototype design:

- encoder 1: `ViT`
- encoder 2: `Swin`
- router input: question text

This is not meant to be the final architecture. It is the first research testbed for soft routing at the visual-encoder stage.

## Baseline Summary

At the moment, the repo gives you three useful comparison points:

1. `ChartQA` single-task baseline with `Pix2Struct`
2. `DocVQA` single-task baseline with `Donut`
3. multitask dense baseline in Colab

Then the next comparison is the soft-routed two-encoder notebook.

### Recorded Baseline Losses

Losses currently recovered from saved trainer checkpoints in `outputs/`:

- `Pix2Struct + ChartQA`
  - final train log near epoch end: `0.3754`
  - `eval_loss = 0.6266`
- `BLIP + ChartQA`
  - final train log near epoch end: `2.4986`
  - `eval_loss = 3.3492`
- `BLIP multitask baseline`
  - final train log near epoch end: `3.8854`
  - `eval_loss = 3.7679`
- `BLIP multitask LoRA`
  - final train log near epoch end: `17.6419`
  - `eval_loss = 4.4062`

These numbers are not meant as polished benchmark reporting. They are just the current research reference points preserved from previous runs.

## Environment

Local environment used during development:

- Python `3.10`
- Conda env: `moe-assistant`

Core libraries:

- `torch`
- `transformers`
- `datasets`
- `pillow`
- `accelerate`
- `sentencepiece`

## Practical Notes

- local Mac runs are fine for lighter experiments, but `DocVQA` is much better suited to Colab
- notebook `03` contains the most practical end-to-end single-task Colab path
- notebook `04` is the dense multitask baseline
- notebook `05` is the current MoE prototype
- the code is intentionally minimal and easy to change rather than heavily abstracted

## Research Direction

The near-term plan this repo supports is:

1. validate data and training loops
2. collect single-task and dense multitask baselines
3. test question-conditioned routing over multiple visual encoders
4. inspect whether different tasks lean toward different experts
5. iterate toward a stronger multimodal MoE design

## References

This repo is loosely inspired by:

- Switch Transformer
- Mixtral
- MoME: Mixture of Multimodal Experts for Generalist Multimodal Large Language Models

The goal here is not exact reproduction. The goal is fast research iteration on multimodal routing ideas.
