import PIL.Image
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

from train_wandb import center_crop_arr

# -------------------------------------------------------------------
# (Optional) monkey-patch PIL to swallow bad EXIF decodes:
# -------------------------------------------------------------------
_orig_getexif = PIL.Image.Image.getexif
def _safe_getexif(self):
    try:
        return _orig_getexif(self)
    except (UnicodeDecodeError, AttributeError, OSError) as e:
        print(f"⚠️ Warning: EXIF decode failed for {getattr(self,'filename',None)} → {e}")
        return {}
PIL.Image.Image.getexif = _safe_getexif

# -------------------------------------------------------------------
# Your existing transform + dataset
# -------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Lambda(lambda img: center_crop_arr(img, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3, inplace=True)
])

class HFDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf = hf_dataset
        self.tf = transform

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, idx):
        # wrap the access in try/except if you want to pinpoint bad indices
        try:
            ex = self.hf[idx]
        except UnicodeDecodeError as e:
            print(f"❌ UnicodeDecodeError at index {idx}: {e}")
            raise
        img = ex["jpg"]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        x = self.tf(img)
        y = ex["json"]["label"]
        return x, y

# -------------------------------------------------------------------
# Build loader
# -------------------------------------------------------------------
ds = load_dataset("timm/imagenet-1k-wds", cache_dir="data/imagenet")
train_hf = ds["train"]

global_batch_size = 512
num_workers = 35

dataset = HFDataset(train_hf, transform)
sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=0)
loader = DataLoader(
    dataset,
    batch_size=global_batch_size,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)

# -------------------------------------------------------------------
# Smoke-test loop
# -------------------------------------------------------------------
try:
    for x, y in tqdm(loader, desc="Loading batches", total=len(loader)):
        # nothing special—just iterating is enough to trigger HF's EXIF path
        _ = x.shape
    print("✅ DataLoader completed without UnicodeDecodeError.")
except RuntimeError as e:
    print("❌ Caught RuntimeError during DataLoader iteration:", e)
    cause = e.__cause__
    if cause:
        print(f"Underlying exception → {type(cause).__name__}: {cause}")
    else:
        print("No nested exception info available.")
