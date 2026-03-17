from datasets import load_dataset
print("Downloading DocVQA...")
docvqa = load_dataset("lmms-lab/DocVQA", "DocVQA")

print("Downloading ChartQA...")
chartqa = load_dataset("HuggingFaceM4/ChartQA")

print("Download finished!")

print(docvqa)
print(chartqa)
