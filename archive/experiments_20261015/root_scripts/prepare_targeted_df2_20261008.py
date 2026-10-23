#!/usr/bin/env python3
"""Prepare a separate DF2 coverage experiment; never train or edit frozen GT.

Select 50 new source images each for short-sleeve outerwear, long-sleeve
outerwear and shorts. Exclude existing train/development/Core paths and exact
file/pixel duplicates. Preserve every usable eight-class target on each image.
Near duplicates are not certified by this exact-duplicate check.
"""
import csv
import hashlib
import io
import json
import os
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["FASHION_PROJECT_ROOT"] = str(ROOT)
DATA = Path(os.environ.get("FASHION_DATA_ROOT", "/workspace/fashion_data")).resolve()
from PIL import Image, ImageDraw
from fashion_multimodal_analysis.analysis import diagnose_prd31_v4 as diag
from fashion_multimodal_analysis.segmentation.training import run_prd31_v5_full_targets as v5

SEED = 20261008
LIMIT = 50
GROUPS = {3: "short_sleeve_outerwear", 4: "long_sleeve_outerwear", 7: "shorts"}
TAG = "df2_targeted150_20261008"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fingerprints(path, image):
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    return file_hash, pixel_hash


def main():
    inputs = [ROOT / p for p in (
        "configs/prd_8class_train_v5_full_targets.csv",
        "configs/prd_8class_val_v2_full_targets.csv",
        "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv",
    )]
    identities = {str(p.relative_to(ROOT)): diag.digest(p) for p in inputs}
    base = read_csv(inputs[0])
    excluded = {diag.resolve(r["source_image"], ROOT, DATA).resolve()
                for p in inputs for r in read_csv(p)}
    file_hashes, pixel_hashes = set(), set()
    print("Checking existing train/development/Core image identities...", flush=True)
    for path in sorted(excluded):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        file_hash, pixel_hash = fingerprints(path, image)
        file_hashes.add(file_hash)
        pixel_hashes.add(pixel_hash)

    raw = DATA / "raw/train/train"
    pools = {category: [] for category in GROUPS}
    print("Scanning candidate annotations...", flush=True)
    for index, path in enumerate(sorted((raw / "annos").glob("*.json")), 1):
        image_path = raw / "image" / (path.stem + ".jpg")
        if image_path.resolve() in excluded or not image_path.is_file():
            continue
        source = json.loads(path.read_text(encoding="utf-8-sig"))
        found = {}
        for key, ann in sorted(source.items()):
            if not key.startswith("item") or not isinstance(ann, dict):
                continue
            category = int(ann.get("category_id", -1))
            if category in GROUPS and diag.usable_annotation(ann, "DeepFashion2"):
                found.setdefault(category, key)
        for category, key in found.items():
            pools[category].append((path, image_path, key))
        if index % 20000 == 0:
            print("Scanned:", index, flush=True)

    rng = random.Random(SEED)
    selected, skipped, additions, previews = [], [], [], []
    taken_paths = set()
    for category, group in GROUPS.items():
        rng.shuffle(pools[category])
        count = 0
        for annotation_path, image_path, target_key in pools[category]:
            if image_path.resolve() in taken_paths:
                continue
            try:
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                file_hash, pixel_hash = fingerprints(image_path, image)
                if file_hash in file_hashes or pixel_hash in pixel_hashes:
                    skipped.append({"image": str(image_path), "reason": "exact duplicate"})
                    continue
                source = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
                targets = []
                for key, ann in sorted(source.items()):
                    if not key.startswith("item") or not isinstance(ann, dict):
                        continue
                    native = int(ann.get("category_id", -1))
                    if native not in diag.DF2_CLASS or not diag.usable_annotation(ann, "DeepFashion2"):
                        continue
                    mask = v5.decode_mask(ann, *image.size)
                    targets.append((key, ann, mask))
                if target_key not in {t[0] for t in targets}:
                    raise ValueError("Selected target absent after mask validation")
            except (OSError, ValueError, KeyError, TypeError) as error:
                skipped.append({"image": str(image_path), "reason": str(error)})
                continue

            for key, ann, mask in targets:
                native = int(ann["category_id"])
                fine = diag.DF2_FINE[native - 1]
                x1, y1, x2, y2 = mask.getbbox()
                relative = Path("processed") / TAG / "train" / (image_path.stem + "_" + key + ".png")
                buffer = io.BytesIO()
                mask.save(buffer, format="PNG")
                v5.same_or_write(DATA / relative, buffer.getvalue())
                additions.append(dict(
                    record_id=f"{TAG}_{len(additions)+1:05d}",
                    source_dataset="DeepFashion2",
                    source_image="../fashion_data/raw/train/train/image/" + image_path.name,
                    garment_id=key + "_" + fine,
                    garment_category=diag.DF2_CLASS[native], fine_category=fine,
                    source_category_id=native, annotation_ref=key,
                    mask_path="../fashion_data/" + relative.as_posix(),
                    bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                    bbox_width=x2-x1, bbox_height=y2-y1,
                    image_width=image.width, image_height=image.height,
                    source_manifest=TAG,
                ))
                if key == target_key and count < 6:
                    pad = max(20, int(max(x2-x1, y2-y1) * 0.2))
                    crop = (max(0,x1-pad), max(0,y1-pad), min(image.width,x2+pad), min(image.height,y2+pad))
                    whole = image.copy()
                    ImageDraw.Draw(whole).rectangle((x1,y1,x2,y2), outline="lime", width=4)
                    tinted = Image.blend(image, Image.new("RGB", image.size, "lime"), 0.35)
                    overlay = Image.composite(tinted, image, mask)
                    for label, picture in (("whole_gt_box", whole), ("target_original",image.crop(crop)), ("target_gt_mask",overlay.crop(crop))):
                        picture.thumbnail((1200,1200))
                        jpg = io.BytesIO()
                        picture.save(jpg, format="JPEG", quality=90)
                        previews.append((f"{group}/{image_path.stem}_{key}_{label}.jpg", jpg.getvalue()))
            selected.append(dict(group=group, source_image=str(image_path), annotation_ref=target_key,
                                 annotation_sha256=diag.digest(annotation_path), file_sha256=file_hash,
                                 rgb_pixel_sha256=pixel_hash))
            taken_paths.add(image_path.resolve())
            file_hashes.add(file_hash)
            pixel_hashes.add(pixel_hash)
            count += 1
            if count == LIMIT:
                break
        if count != LIMIT:
            raise RuntimeError(f"Not enough validated unique images for {group}: {count}/{LIMIT}")
        print(group, "selected:", count, flush=True)

    for path in inputs:
        if diag.digest(path) != identities[str(path.relative_to(ROOT))]:
            raise RuntimeError("An input manifest changed during preparation")
    combined = base + additions
    manifest = ROOT / "configs/prd_8class_train_v5_targeted150_20261008.csv"
    v5.same_or_write(manifest, v5.csv_bytes(combined))
    report_dir = ROOT / "reports/reruns/recovery_20261008/targeted150_preparation"
    report = dict(status="PREPARED_NOT_TRAINED", seed=SEED,
                  selected_images=len(selected), added_instances=len(additions),
                  total_instances=len(combined),
                  train_images=len({r["source_image"] for r in combined}),
                  input_sha256=identities, output_sha256=diag.digest(manifest),
                  added_classes=dict(Counter(r["garment_category"] for r in additions)),
                  selected=selected, skipped=skipped,
                  duplicate_check="Source paths, exact file bytes and exact RGB pixels; near duplicates not certified")
    encoded = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
    v5.same_or_write(report_dir / "preparation.json", encoded)
    review = ROOT.parent / "v5_targeted150_candidate_review.zip"
    with zipfile.ZipFile(review, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("preparation.json", encoded)
        for name, content in previews:
            archive.writestr(name, content)
    print("New manifest:", manifest)
    print("New images:", len(selected), "added targets:", len(additions))
    print("Upload review ZIP:", review)
    print("Review ZIP MiB:", round(review.stat().st_size / 1024**2, 2))
    print("No training started. Original V5 and frozen evaluation files preserved.")


if __name__ == "__main__":
    main()
