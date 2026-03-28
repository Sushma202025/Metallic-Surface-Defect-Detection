"""
train.py - Two-phase training for CNN + ViT on NEU Metal Surface Defect dataset

PHASE 1 (fast, ~10-15 min on CPU):
  - ViT frozen, only CNN + classifier head trained
  - Full 224x224 resolution
  - Gets model to ~70-80% accuracy quickly

PHASE 2 (fine-tune, ~20-30 min on CPU):
  - Unfreeze entire model
  - Lower learning rate
  - Gets model to ~85-95% accuracy

Total expected time on CPU: 30-45 minutes
This is worth it - your current model is barely learning.
"""

import os
import copy
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from model import CNN_ViT

# ---- Config ------------------------------------------------------------------
DATASET_DIR = "dataset/train"
MODEL_OUT   = "cnn_vit_metal_defect.pth"
SEED        = 42

# Phase 1 - frozen ViT, fast warmup
P1_EPOCHS     = 8
P1_BATCH_SIZE = 16
P1_LR         = 5e-4

# Phase 2 - full fine-tune
P2_EPOCHS     = 15
P2_BATCH_SIZE = 8    # smaller batch for stability
P2_LR         = 5e-5 # much lower LR when unfreezing ViT

PATIENCE      = 5
NUM_WORKERS   = 0    # must be 0 on Windows
IMAGE_SIZE    = 224  # full resolution - critical for NEU texture defects

# ------------------------------------------------------------------------------

if __name__ == "__main__":

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  MetalInspect - Two-Phase Training")
    print(f"  Device     : {device}")
    print(f"  Image size : {IMAGE_SIZE}x{IMAGE_SIZE}  (full resolution)")
    print(f"  Phase 1    : {P1_EPOCHS} epochs, ViT FROZEN")
    print(f"  Phase 2    : {P2_EPOCHS} epochs, full fine-tune")
    print("=" * 60)

    # ---- Transforms ----------------------------------------------------------
    # NEU defects are texture-based - augmentation must preserve texture
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        # Mild colour jitter only - NEU is greyscale-ish, heavy colour hurts
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ---- Dataset -------------------------------------------------------------
    full_dataset = ImageFolder(DATASET_DIR, transform=train_tf)
    num_classes  = len(full_dataset.classes)
    print(f"\n  Classes ({num_classes}): {full_dataset.classes}")
    print(f"  Total images: {len(full_dataset)}")

    # Show class distribution
    from collections import Counter
    label_counts = Counter(full_dataset.targets)
    for idx, cls in enumerate(full_dataset.classes):
        print(f"    {cls:<12} {label_counts[idx]} images")

    with open("classes.txt", "w") as f:
        f.write("\n".join(full_dataset.classes))

    val_size   = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    val_ds.dataset           = copy.deepcopy(val_ds.dataset)
    val_ds.dataset.transform = val_tf

    print(f"\n  Train: {train_size}  |  Val: {val_size}")

    # ---- Model ---------------------------------------------------------------
    model = CNN_ViT(num_classes=num_classes, pretrained=True).to(device)

    # ========== PHASE 1: Train CNN + head only, ViT frozen ===================
    print(f"\n{'=' * 60}")
    print(f"  PHASE 1: Warming up CNN + classifier (ViT frozen)")
    print(f"  Estimated time: {P1_EPOCHS * 3}-{P1_EPOCHS * 5} min on CPU")
    print(f"{'=' * 60}")

    model.freeze_vit_backbone()
    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total        = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_p1:,} / {total:,}")

    train_loader = DataLoader(train_ds, batch_size=P1_BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=P1_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=P1_LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=P1_EPOCHS, eta_min=1e-6)

    best_val_loss = float("inf")
    best_weights  = None
    patience_ctr  = 0

    def run_epoch(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()
        total_loss, correct, total = 0.0, 0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for i, (images, labels) in enumerate(loader):
                images, labels = images.to(device), labels.to(device)
                if train:
                    optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                total_loss += loss.item() * images.size(0)
                correct    += (outputs.argmax(1) == labels).sum().item()
                total      += images.size(0)
                if train and ((i+1) % 20 == 0 or (i+1) == len(loader)):
                    print(f"\r    step {i+1}/{len(loader)}", end="", flush=True)
        if train:
            print("\r", end="")
        return total_loss / total, 100 * correct / total

    print(f"\n  {'Ep':<5} {'TrLoss':<10} {'TrAcc':<10} {'VaLoss':<10} {'VaAcc':<10} Time")
    print(f"  {'-'*55}")

    for epoch in range(1, P1_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        scheduler.step()
        va_loss, va_acc = run_epoch(val_loader,   train=False)
        elapsed = time.time() - t0
        saved   = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_weights  = copy.deepcopy(model.state_dict())
            torch.save(best_weights, MODEL_OUT)
            patience_ctr  = 0
            saved = " SAVED"
        else:
            patience_ctr += 1
        print(f"  {epoch:<5} {tr_loss:<10.4f} {tr_acc:<10.1f} {va_loss:<10.4f} {va_acc:<10.1f} {elapsed:.0f}s{saved}")
        if patience_ctr >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    # ========== PHASE 2: Unfreeze all, fine-tune =============================
    print(f"\n{'=' * 60}")
    print(f"  PHASE 2: Fine-tuning full model (ViT unfrozen)")
    print(f"  Estimated time: {P2_EPOCHS * 4}-{P2_EPOCHS * 7} min on CPU")
    print(f"{'=' * 60}")

    # Load best phase 1 weights before unfreezing
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    model.unfreeze_all()
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Rebuild loader with smaller batch for phase 2
    train_loader2 = DataLoader(train_ds, batch_size=P2_BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader2   = DataLoader(val_ds,   batch_size=P2_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    # Lower LR and use different LR for ViT vs CNN
    optimizer2 = torch.optim.AdamW([
        {"params": model.cnn.parameters(),     "lr": P2_LR * 5},   # CNN can learn faster
        {"params": model.adapter.parameters(), "lr": P2_LR * 5},
        {"params": model.vit.parameters(),     "lr": P2_LR},        # ViT very low LR
    ], weight_decay=1e-4)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=P2_EPOCHS, eta_min=1e-7)

    best_val_loss2 = float("inf")
    patience_ctr2  = 0

    print(f"\n  {'Ep':<5} {'TrLoss':<10} {'TrAcc':<10} {'VaLoss':<10} {'VaAcc':<10} Time")
    print(f"  {'-'*55}")

    for epoch in range(1, P2_EPOCHS + 1):
        t0 = time.time()
        # Inline train loop for phase 2 (different optimizer)
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for i, (images, labels) in enumerate(train_loader2):
            images, labels = images.to(device), labels.to(device)
            optimizer2.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer2.step()
            tl += loss.item() * images.size(0)
            tc += (outputs.argmax(1) == labels).sum().item()
            tt += images.size(0)
            if (i+1) % 20 == 0 or (i+1) == len(train_loader2):
                print(f"\r    step {i+1}/{len(train_loader2)}", end="", flush=True)
        print("\r", end="")
        scheduler2.step()

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader2:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                vl += criterion(outputs, labels).item() * images.size(0)
                vc += (outputs.argmax(1) == labels).sum().item()
                vt += images.size(0)

        tr_loss, tr_acc = tl/tt, 100*tc/tt
        va_loss, va_acc = vl/vt, 100*vc/vt
        elapsed = time.time() - t0
        saved   = ""
        if va_loss < best_val_loss2:
            best_val_loss2 = va_loss
            torch.save(copy.deepcopy(model.state_dict()), MODEL_OUT)
            patience_ctr2 = 0
            saved = " SAVED"
        else:
            patience_ctr2 += 1
        print(f"  {epoch:<5} {tr_loss:<10.4f} {tr_acc:<10.1f} {va_loss:<10.4f} {va_acc:<10.1f} {elapsed:.0f}s{saved}")
        if patience_ctr2 >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    # ---- Final per-class accuracy -------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Loading best checkpoint for evaluation...")
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    model.eval()

    class_correct = [0] * num_classes
    class_total   = [0] * num_classes
    with torch.no_grad():
        for images, labels in val_loader2:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            for p, l in zip(preds, labels):
                class_total[l.item()]   += 1
                class_correct[l.item()] += int(p == l)

    print("\n  Per-class validation accuracy:")
    for i, cls in enumerate(full_dataset.classes):
        if class_total[i]:
            acc    = 100 * class_correct[i] / class_total[i]
            filled = int(acc / 5)
            bar    = "█" * filled + "░" * (20 - filled)
            print(f"    {cls:<12} {bar}  {acc:.1f}%  ({class_correct[i]}/{class_total[i]})")

    overall = 100 * sum(class_correct) / sum(class_total)
    print(f"\n  Overall accuracy : {overall:.1f}%")
    print(f"  Model saved to   : {MODEL_OUT}")
    print(f"\n  Next step: python app.py")
    print("=" * 60)
