# Multi-Task MoE Assistant (Work in Progress)

This project explores building a multi-task AI assistant using a Mixture-of-Experts (MoE) architecture.

Instead of relying on a single dense feed-forward network for all tasks, the model routes inputs to specialized experts designed to handle different types of tasks.

## Motivation

Modern language models are typically dense: every input passes through the same parameters regardless of task type.

This project investigates whether task-specialized experts within a shared Transformer backbone can:

- Encourage functional specialization
- Improve modularity
- Provide clearer separation between different task behaviors

## Initial Scope

The first phase focuses on three tasks:

- Summarization
- Question Answering
- Email / Text Rewriting

Each task will be trained within a shared model that contains multiple experts.  
A routing mechanism determines which expert processes a given input.

## Architecture Overview

- Shared Transformer backbone
- Mixture-of-Experts feed-forward layer
- Top-k routing mechanism
- Auxiliary load-balancing objective

The goal is to observe and analyze how experts specialize across tasks.
## References

This project is inspired by and built upon ideas from:

- Switch Transformer (Fedus et al., 2021)
- Mixtral 8x7B Technical Report (Mistral AI, 2023)
- Literature on multi-task learning and parameter sharing

The goal of this repository is not to reproduce these works,
but to implement a simplified, educational version of
task-specialized Mixture-of-Experts models.
## Current Status

🚧 Early development stage.

Planned steps:

1. Implement core MoE layer
2. Integrate into Transformer block
3. Train on multi-task dataset
4. Track expert utilization across tasks

## Repository Structure
