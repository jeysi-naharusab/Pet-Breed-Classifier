import random
import shutil 
from pathlib import Path

from torchvision.datasets import OxfordIIITPet

RAW = Path("data/raw")
OUT = Path("data")
SEED = 42
VAL_FRACTION = 0.2  # 20% of each breed goes to validation

# final class name -> (Stanford names, Oxford names)
# Dalmatian, Dachshund and Jack_Russell_Terrier are in neither dataset.
CLASSES = {
    # small and toy dogs
    "Shih_Tzu": (["Shih-Tzu"], []),
    "Chihuahua": (["Chihuahua"], ["Chihuahua"]),
    "Pomeranian": (["Pomeranian"], ["Pomeranian"]),
    "Pug": (["pug"], ["Pug"]),
    "Poodle": (["toy_poodle", "miniature_poodle", "standard_poodle"], []),
    "Maltese": (["Maltese_dog"], []),
    "Yorkshire_Terrier": (["Yorkshire_terrier"], ["Yorkshire Terrier"]),
    "Pekingese": (["Pekinese"], []),
    "Miniature_Pinscher": (["miniature_pinscher"], ["Miniature Pinscher"]),
    "French_Bulldog": (["French_bulldog"], []),
    "Shiba_Inu": ([], ["Shiba Inu"]),
    # medium dogs
    "Beagle": (["beagle"], ["Beagle"]),
    "Pembroke_Welsh_Corgi": (["Pembroke"], []),
    "Siberian_Husky": (["Siberian_husky"], []),
    "Chow_Chow": (["chow"], []),
    "Border_Collie": (["Border_collie"], []),
    "Samoyed": (["Samoyed"], ["Samoyed"]),
    # large dogs
    "Labrador_Retriever": (["Labrador_retriever"], []),
    "Golden_Retriever": (["golden_retriever"], []),
    "German_Shepherd": (["German_shepherd"], []),
    "Rottweiler": (["Rottweiler"], []),
    "Doberman": (["Doberman"], []),
    "Great_Dane": (["Great_Dane"], []),
    "Boxer": (["boxer"], ["Boxer"]),
    "Saint_Bernard": (["Saint_Bernard"], ["Saint Bernard"]),
    "Alaskan_Malamute": (["malamute"], []),
    "Belgian_Malinois": (["malinois"], []),
    # cats (Oxford only)
    "Persian": ([], ["Persian"]),
    "British_Shorthair": ([], ["British Shorthair"]),
    "Siamese": ([], ["Siamese"]),
    "Maine_Coon": ([], ["Maine Coon"]),
    "Ragdoll": ([], ["Ragdoll"]),
    "Bengal": ([], ["Bengal"]),
    "Russian_Blue": ([], ["Russian Blue"]),
    "Sphynx": ([], ["Sphynx"]),
    "Birman": ([], ["Birman"]),
    "Abyssinian": ([], ["Abyssinian"]),
}


def index_stanford():
    """Map breed name (without the n0208... code) -> its folder."""
    index = {}
    for folder in (RAW / "stanford_dogs" / "Images").iterdir():
        if folder.is_dir():
            index[folder.name.split("-", 1)[1]] = folder
    return index


def index_oxford():
    """Map Oxford breed name -> list of image paths (both splits pooled)."""
    index = {}
    for split in ("trainval", "test"):
        ds = OxfordIIITPet(root=RAW, split=split)
        for path, label in zip(ds._images, ds._labels):
            index.setdefault(ds.classes[label], []).append(Path(path))
    return index


stanford = index_stanford()
oxford = index_oxford()
rng = random.Random(SEED)

print(f"{'class':<24}{'train':>6}{'val':>6}")
for cls, (stanford_names, oxford_names) in CLASSES.items():
    files = []
    for name in stanford_names:
        files += [(p, "stanford") for p in sorted(stanford[name].glob("*.jpg"))]
    for name in oxford_names:
        files += [(p, "oxford") for p in sorted(oxford[name])]

    rng.shuffle(files)
    n_val = int(len(files) * VAL_FRACTION)
    splits = {"val": files[:n_val], "train": files[n_val:]}

    for split, items in splits.items():
        dest = OUT / split / cls
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for path, source in items:
            shutil.copy2(path, dest / f"{source}_{path.name}")

    flag = "  <-- few images" if len(files) < 150 else ""
    print(f"{cls:<24}{len(splits['train']):>6}{len(splits['val']):>6}{flag}")