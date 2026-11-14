"""
ml/src/data_utils.py  (Phase 1: dataset inspection and EDA)

Path-agnostic helpers: every function receives the paths it needs, so the same
file works on Colab and on a local machine. Nothing here assumes a class count,
class names or image size; all numbers come from the files you point it at.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------
# 1. Directory structure
# --------------------------------------------------------------------------
def _scandir(path):
    with os.scandir(path) as it:
        return sorted(it, key=lambda e: e.name)


def print_tree(root, max_depth: int = 3) -> None:
    """Print folder structure with counts of sub-folders and direct files."""

    def walk(path, depth):
        entries = _scandir(path)
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        print(f"{'  ' * depth}{os.path.basename(path) or path}/  "
              f"subfolders={len(dirs)}  files={len(files)}")
        if depth < max_depth:
            for d in dirs:
                walk(d.path, depth + 1)

    walk(str(root), 0)


def _has_images(path) -> bool:
    return any(e.is_file() and Path(e.name).suffix.lower() in IMAGE_EXTS
               for e in _scandir(path))


def find_class_root(base, max_depth: int = 4):
    """
    Return the shallowest folder that has >= 2 sub-folders which directly contain
    images (that is the folder whose children are the class folders).
    Returns None if nothing matches. Breadth-first, so nested duplicates like
    base/PlantVillage/PlantVillage/<classes> resolve to the real class folder.
    """
    queue = [(str(base), 0)]
    while queue:
        path, depth = queue.pop(0)
        subdirs = [e for e in _scandir(path) if e.is_dir()]
        with_images = [d for d in subdirs if _has_images(d.path)]
        if len(with_images) >= 2:
            return Path(path)
        if depth < max_depth:
            queue.extend((d.path, depth + 1) for d in subdirs)
    return None


def list_class_dirs(class_root):
    """Return (class_names sorted, skipped_folders) under class_root."""
    names, skipped = [], []
    for e in _scandir(class_root):
        if not e.is_dir():
            continue
        (names if _has_images(e.path) else skipped).append(e.name)
    return sorted(names), skipped


def parse_class_name(raw: str) -> dict:
    """
    Best-effort split of a class folder name into plant and disease.
    Review the printed table: naming differs between PlantVillage versions.
    Plant = text before the first underscore; disease = the rest, cleaned.
    """
    first = raw.split("_")[0]
    plant = re.sub(r"[^A-Za-z0-9 ]", "", first).strip() or raw
    rest = raw.split("___", 1)[1] if "___" in raw else raw[len(first):]
    disease = re.sub(r"[()]", "", rest)
    disease = re.sub(r"_+", " ", disease).strip(" ,")
    return {"raw": raw, "plant": plant, "disease": disease,
            "is_healthy": "healthy" in raw.lower()}


# --------------------------------------------------------------------------
# 2. Per-file inspection
# --------------------------------------------------------------------------
def md5_of_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def inspect_image(path) -> dict:
    """Open with PIL (verify, then reopen and fully decode) and record properties."""
    rec = {"filepath": str(path), "size_bytes": None, "md5": None, "status": "ok",
           "error": "", "width": None, "height": None, "mode": None, "format": None}
    try:
        rec["size_bytes"] = os.path.getsize(path)
        rec["md5"] = md5_of_file(path)
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:  # verify() invalidates the handle, so reopen
            im.load()
            rec.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
    except Exception as e:  # any failure counts as unreadable
        rec["status"] = "corrupt"
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def scan_dataset(class_root, class_names, workers: int = 8):
    """
    Inspect every file under each class folder (sub-folders included).
    Returns (images_df, other_files_df). Does NOT modify or delete anything.
    """
    image_items, other = [], []
    for cls in class_names:
        for dirpath, _, files in os.walk(Path(class_root) / cls):
            for fname in files:
                p = Path(dirpath) / fname
                if p.suffix.lower() in IMAGE_EXTS:
                    image_items.append((cls, p))
                else:
                    other.append({"class": cls, "filepath": str(p), "ext": p.suffix.lower()})

    with ThreadPoolExecutor(max_workers=workers) as ex:
        records = list(tqdm(ex.map(lambda cp: inspect_image(cp[1]), image_items),
                            total=len(image_items), desc="Scanning images"))

    df = pd.DataFrame(records)
    df.insert(0, "class", [c for c, _ in image_items])
    df["ext"] = [p.suffix.lower() for _, p in image_items]
    return df, pd.DataFrame(other, columns=["class", "filepath", "ext"])


# --------------------------------------------------------------------------
# 3. Counts, imbalance, duplicates
# --------------------------------------------------------------------------
def class_counts(df, parsed) -> pd.DataFrame:
    """Per-class counts (all image files and readable ones) joined with plant."""
    plant_of = {p["raw"]: p["plant"] for p in parsed}
    out = (df.groupby("class")
             .agg(n_files=("filepath", "size"), n_valid=("status", lambda s: (s == "ok").sum()))
             .reset_index())
    out["plant"] = out["class"].map(plant_of)
    return out.sort_values("n_valid", ascending=False).reset_index(drop=True)


def imbalance_ratio(counts: pd.DataFrame) -> float:
    return float(counts["n_valid"].max() / counts["n_valid"].min())


def plot_class_distribution(counts: pd.DataFrame, out_path=None):
    data = counts.sort_values("n_valid")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(data))))
    ax.barh(data["class"], data["n_valid"])
    for y, v in enumerate(data["n_valid"]):
        ax.text(v, y, f" {v}", va="center", fontsize=8)
    ax.set_xlabel("Readable images")
    ax.set_title("Class distribution (sorted)")
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
    plt.show()


def find_exact_duplicates(df):
    """
    Exact duplicates by MD5 across the whole dataset.
    Returns (duplicate_files_df, summary_dict).
    Groups spanning more than one class are label conflicts, which matter most.
    """
    ok = df[df["md5"].notna()]
    dup = ok[ok.duplicated("md5", keep=False)].copy()
    if dup.empty:
        return dup, {"n_duplicate_groups": 0, "n_files_in_duplicate_groups": 0,
                     "n_redundant_copies": 0, "groups_within_one_class": 0,
                     "groups_across_classes": 0}
    g = dup.groupby("md5").agg(n=("filepath", "size"), n_classes=("class", "nunique"))
    dup = dup.merge(g, left_on="md5", right_index=True)
    return dup.sort_values(["md5", "filepath"]), {
        "n_duplicate_groups": int(len(g)),
        "n_files_in_duplicate_groups": int(g["n"].sum()),
        "n_redundant_copies": int((g["n"] - 1).sum()),
        "groups_within_one_class": int((g["n_classes"] == 1).sum()),
        "groups_across_classes": int((g["n_classes"] > 1).sum()),
    }


# --------------------------------------------------------------------------
# 4. Visual inspection
# --------------------------------------------------------------------------
def show_sample_grid(df, per_class: int = 3, seed: int = 42,
                     classes_per_figure: int = 8, out_dir=None):
    """Random (seeded) sample of readable images, one row per class."""
    rng = random.Random(seed)
    ok = df[df["status"] == "ok"]
    classes = sorted(ok["class"].unique())
    for start in range(0, len(classes), classes_per_figure):
        chunk = classes[start:start + classes_per_figure]
        fig, axes = plt.subplots(len(chunk), per_class,
                                 figsize=(3 * per_class, 2.6 * len(chunk)), squeeze=False)
        for r, cls in enumerate(chunk):
            paths = ok.loc[ok["class"] == cls, "filepath"].tolist()
            picks = rng.sample(paths, min(per_class, len(paths)))
            for c in range(per_class):
                ax = axes[r][c]
                ax.axis("off")
                if c < len(picks):
                    with Image.open(picks[c]) as im:
                        ax.imshow(im.convert("RGB"))
            axes[r][0].set_title(cls, fontsize=8, loc="left")
        fig.tight_layout()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, f"sample_grid_{start // classes_per_figure + 1:02d}.png"),
                        dpi=120)
        plt.show()


# --------------------------------------------------------------------------
# 5. Summary
# --------------------------------------------------------------------------
def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def save_json(obj, path) -> None:
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def build_summary(dataset_root, class_root, class_names, parsed, df, other_df,
                  counts, dup_summary, test_fraction: float = 0.15) -> dict:
    ok = df[df["status"] == "ok"]
    plant_counts = counts.groupby("plant")["n_valid"].sum().sort_values(ascending=False)
    size_counts = ok.groupby(["width", "height"]).size().sort_values(ascending=False)
    smallest = counts.sort_values("n_valid").head(5)
    return {
        "dataset_root": str(dataset_root),
        "class_root": str(class_root),
        "n_classes": len(class_names),
        "class_names": class_names,
        "plants": sorted({p["plant"] for p in parsed}),
        "n_plants": len({p["plant"] for p in parsed}),
        "n_image_files": int(len(df)),
        "n_readable": int(len(ok)),
        "n_corrupt": int((df["status"] == "corrupt").sum()),
        "corrupt_files": df.loc[df["status"] == "corrupt", ["filepath", "error"]]
                           .head(100).to_dict("records"),
        "n_non_image_files": int(len(other_df)),
        "non_image_extensions": other_df["ext"].value_counts().to_dict(),
        "file_extensions": df["ext"].value_counts().to_dict(),
        "formats": ok["format"].value_counts().to_dict(),
        "modes": ok["mode"].value_counts().to_dict(),
        "n_non_rgb": int((ok["mode"] != "RGB").sum()),
        "n_distinct_sizes": int(len(size_counts)),
        "top_sizes": {f"{int(w)}x{int(h)}": int(n) for (w, h), n in size_counts.head(10).items()},
        "per_class_counts": dict(zip(counts["class"], counts["n_valid"].astype(int))),
        "per_plant_counts": plant_counts.astype(int).to_dict(),
        "max_class": {"name": counts.iloc[0]["class"], "n": int(counts.iloc[0]["n_valid"])},
        "min_class": {"name": counts.iloc[-1]["class"], "n": int(counts.iloc[-1]["n_valid"])},
        "imbalance_ratio_max_over_min": round(imbalance_ratio(counts), 3),
        "smallest_classes_with_expected_test_size": [
            {"class": r["class"], "n_valid": int(r["n_valid"]),
             f"expected_test_images_at_{int(test_fraction * 100)}pct": math.floor(r["n_valid"] * test_fraction)}
            for _, r in smallest.iterrows()],
        "exact_duplicates": dup_summary,
    }
