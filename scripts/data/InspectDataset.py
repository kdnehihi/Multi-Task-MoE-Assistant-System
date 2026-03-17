from datasets import load_dataset

docvqa = load_dataset("lmms-lab/DocVQA", "DocVQA")
chartqa = load_dataset("HuggingFaceM4/ChartQA")

print("DocVQA columns:", docvqa["validation"].column_names)
print(docvqa["validation"][0])

print("\nChartQA columns:", chartqa["train"].column_names)
print(chartqa["train"][0])
