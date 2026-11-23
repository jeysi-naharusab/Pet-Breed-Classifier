import json
from pathlib import Path

import timm
import torch
from timm.data import create_transform, resolve_data_config
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

DATA = Path("data")
CKPT = Path("checkpoints")
BATCH_SIZE = 32       # lower to 16 if you get "CUDA out of memory"
HEAD_EPOCHS = 3       # phase 1: train only the new final layer
FINETUNE_EPOCHS = 6   # phase 2: fine-tune the whole network


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print("Training on:", device)
    CKPT.mkdir(exist_ok=True)

    # 1. data
    train_ds = ImageFolder(DATA / "train")
    classes = train_ds.classes
    json.dump(classes, open(CKPT / "classes.json", "w"))

    # 2. model, and the preprocessing it was pretrained with
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(classes))
    cfg = resolve_data_config({}, model=model)
    train_ds.transform = create_transform(**cfg, is_training=True)
    val_ds = ImageFolder(DATA / "val", transform=create_transform(**cfg))
    assert val_ds.classes == classes, "train and val folders don't match"
    model.to(device)

    # 3. show small classes more often
    counts = torch.bincount(torch.tensor(train_ds.targets))
    weights = (1.0 / counts.float())[train_ds.targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=64, num_workers=4)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best = 0.0

    def run(epochs, lr, head_only):
        nonlocal best
        for p in model.parameters():
            p.requires_grad = not head_only
        for p in model.get_classifier().parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

        for ep in range(epochs):
            model.train()
            total = 0.0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with torch.autocast(device_type=device, enabled=use_amp):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item()

            model.eval()
            correct = 0
            with torch.no_grad():
                for x, y in val_dl:
                    with torch.autocast(device_type=device, enabled=use_amp):
                        pred = model(x.to(device)).argmax(1).cpu()
                    correct += (pred == y).sum().item()
            acc = correct / len(val_ds)
            print(f"epoch {ep + 1}/{epochs}  loss {total / len(train_dl):.3f}  val acc {acc:.3f}")

            torch.save(model.state_dict(), CKPT / "last.pt")
            if acc > best:
                best = acc
                torch.save(model.state_dict(), CKPT / "best.pt")

    print("Phase 1: training the new head")
    run(HEAD_EPOCHS, 1e-3, head_only=True)
    print("Phase 2: fine-tuning everything")
    run(FINETUNE_EPOCHS, 1e-4, head_only=False)
    print(f"Done. Best val acc {best:.3f}. Saved to {CKPT / 'best.pt'}")


if __name__ == "__main__":
    main()