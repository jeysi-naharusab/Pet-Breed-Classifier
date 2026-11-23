import tarfile
import urllib.request
from pathlib import Path

from torchvision.datasets import OxfordIIITPet

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

# 1. Oxford Pets (37 breeds, includes 12 cats)
OxfordIIITPet(root=RAW, split="trainval", download=True)
OxfordIIITPet(root=RAW, split="test", download=True)
print("Oxford Pets done")

# 2. Stanford Dogs (120 dog breeds, about 780 MB)
url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
tar_path = RAW / "stanford_images.tar"
out_dir = RAW / "stanford_dogs"

def show_progress(block, block_size, total):
    percent = min(100, block * block_size * 100 // total)
    print(f"\rStanford Dogs download: {percent}%", end="")

if not out_dir.exists():
    urllib.request.urlretrieve(url, tar_path, show_progress)
    print("\nExtracting...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(out_dir, filter="data")
    tar_path.unlink()  # delete the .tar to save disk space
print("Stanford Dogs done")