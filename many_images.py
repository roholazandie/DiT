import os
import random
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Load dataset
dataset = load_dataset("yandex/alchemist", split="train")

# Choose 100 random samples
random.seed(42)
samples = random.sample(list(dataset), 100)

# Image processing parameters
IMAGE_SIZE = (256, 256)

# Accumulator for averaging
accumulator = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.float64)
count = 0

# Download and process images
for sample in tqdm(samples):
    url = sample['url']
    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_np = np.array(image, dtype=np.float64)
        accumulator += image_np
        count += 1
    except Exception as e:
        print(f"Failed to load {url}: {e}")

# Compute average
if count > 0:
    average_image = (accumulator / count).astype(np.uint8)
    average_pil = Image.fromarray(average_image)
    # average_pil.show()
    # save the average image
    average_pil.save("average_image.png")
else:
    print("No images were successfully downloaded.")

