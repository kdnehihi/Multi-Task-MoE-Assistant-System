# Adaptive Sparse LLM (Work in Progress)

This repository contains an ongoing project focused on building a cost-aware Mixture-of-Experts (MoE) based language model system.

## Motivation

Modern large language models are computationally expensive because every token passes through the full dense network. However, many real-world requests may not require the full model capacity.

This project explores whether a sparse Mixture-of-Experts architecture can reduce compute usage while maintaining task performance.

## Current Focus

The initial phase of this project aims to:

- Implement a Transformer backbone
- Integrate a Mixture-of-Experts (MoE) feed-forward layer
- Implement top-k routing
- Add auxiliary load balancing loss
- Establish a clean training and evaluation pipeline

## Goals (Early Stage)

- Compare dense vs sparse forward pass compute cost
- Track expert utilization statistics
- Analyze routing behavior across different input types

## Status

🚧 Work in progress.  
The repository is currently under active development.

## Planned Structure
