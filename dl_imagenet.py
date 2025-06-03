from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("timm/imagenet-1k-wds", cache_dir="data/imagenet")

for item in ds["train"]:
    print(item["jpg"].shape, item['json']['label'])
    break  # Remove this line to process the entire dataset