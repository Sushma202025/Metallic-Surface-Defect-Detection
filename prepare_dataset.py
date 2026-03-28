"""
prepare_dataset.py
──────────────────
Converts the Kaggle NEU Metal Surface Defect dataset into
ImageFolder format required by train.py.

Run from your project folder (C:\\Users\\sushm\\Desktop\\MP):
    python prepare_dataset.py

It will auto-detect the NEU folder even if the name has spaces.
You can also pass paths manually:
    python prepare_dataset.py --src "NEU Metal Surface Defects Data" --dst dataset
"""

import os
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

# ── NEU filename prefix → class name ──────────────────────────────────────────
PREFIX_MAP = {
    "Cr": "Crazing",
    "In": "Inclusion",
    "Pa": "Patches",
    "PS": "Pitted",
    "RS": "Rolled",
    "Sc": "Scratches",
}

FOLDER_MAP = {
    "crazing":   "Crazing",
    "inclusion": "Inclusion",
    "patches":   "Patches",
    "pitted":    "Pitted",
    "rolled":    "Rolled",
    "scratches": "Scratches",
    "normal":    "Normal",
    "cr": "Crazing",
    "in": "Inclusion",
    "pa": "Patches",
    "ps": "Pitted",
    "rs": "Rolled",
    "sc": "Scratches",
}

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}


def auto_find_neu_folder(base: Path) -> Path | None:
    """
    Look for the NEU dataset folder automatically.
    Handles names like:
      - NEU-DET
      - NEU Metal Surface Defects Data
      - NEU_Metal_Surface_Defects
      - any folder containing 'NEU' (case-insensitive)
    """
    candidates = [p for p in base.iterdir()
                  if p.is_dir() and "neu" in p.name.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Prefer the one that actually contains image files
        for c in candidates:
            if any(c.rglob("*.bmp")) or any(c.rglob("*.jpg")):
                return c
    return None


def class_from_filename(name: str) -> str | None:
    stem   = Path(name).stem
    prefix = stem.split("_")[0]
    return PREFIX_MAP.get(prefix)


def find_images(root: Path) -> list[tuple[Path, str]]:
    results = []
    for img_path in sorted(root.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        # Try parent folder name first
        cls = FOLDER_MAP.get(img_path.parent.name.lower())
        # Fall back to filename prefix
        if cls is None:
            cls = class_from_filename(img_path.name)
        if cls:
            results.append((img_path, cls))
        else:
            print(f"  [skip] Cannot determine class for: {img_path.name}")
    return results


def copy_split(src_root: Path, dst_root: Path, split: str) -> int:
    """Copy images from src_root into dst_root/split/ClassName/."""
    # Try both with and without a split subfolder
    src_split = src_root / split
    search_root = src_split if src_split.exists() else src_root

    pairs = find_images(search_root)
    if not pairs:
        print(f"  [warn] No labelled images found under: {search_root}")
        return 0

    counts: defaultdict[str, int] = defaultdict(int)
    for img_path, cls in pairs:
        out_dir = dst_root / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)
        counts[cls] += 1

    print(f"\n  {split}/ split:")
    for cls in sorted(counts):
        print(f"    {cls:<14} {counts[cls]} images")
    print(f"    {'TOTAL':<14} {sum(counts.values())} images")
    return sum(counts.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None,
                        help="Path to extracted NEU folder (auto-detected if omitted)")
    parser.add_argument("--dst", default="dataset",
                        help="Output folder (default: dataset)")
    args = parser.parse_args()

    here = Path.cwd()
    dst  = Path(args.dst)

    # ── Resolve source ──────────────────────────────────────────────────────
    if args.src:
        src = Path(args.src)
        if not src.exists():
            # Try treating it as relative to cwd
            src = here / args.src
    else:
        print("[prepare_dataset] --src not given, searching current folder…")
        src = auto_find_neu_folder(here)
        if src is None:
            print("\n[ERROR] Could not auto-detect NEU dataset folder.")
            print("  Make sure the extracted NEU folder is in:", here)
            print("  Or run:  python prepare_dataset.py --src \"NEU Metal Surface Defects Data\"")
            raise SystemExit(1)
        print(f"[prepare_dataset] Auto-detected: {src.name}")

    if not src.exists():
        print(f"\n[ERROR] Folder not found: {src}")
        print(f"  Current directory: {here}")
        print("  Folders here:")
        for p in here.iterdir():
            if p.is_dir():
                print(f"    {p.name}")
        raise SystemExit(1)

    print(f"\n[prepare_dataset] Source : {src.resolve()}")
    print(f"[prepare_dataset] Output : {dst.resolve()}\n")

    # ── Try train + test splits, else copy everything into train ───────────
    total = 0
    has_train = (src / "train").exists()
    has_test  = (src / "test").exists()

    if has_train or has_test:
        if has_train:
            total += copy_split(src, dst, "train")
        if has_test:
            total += copy_split(src, dst, "test")
    else:
        # Flat structure — put all images into train/
        print("  No train/test sub-folders found. Copying all images into train/")
        total += copy_split(src, dst, "train")

    if total == 0:
        print("\n[ERROR] No images were copied. Check that your NEU folder contains .bmp or .jpg files.")
        raise SystemExit(1)

    # ── Write classes.txt so app.py picks up classes automatically ─────────
    class_dirs = sorted(p.name for p in (dst / "train").iterdir() if p.is_dir())
    with open("classes.txt", "w") as f:
        f.write("\n".join(class_dirs))
    print(f"\n  classes.txt written: {class_dirs}")

    print(f"\n✅ Done! {total} images organised into {dst}/")
    print("   Next step: python train.py")


if __name__ == "__main__":
    main()
