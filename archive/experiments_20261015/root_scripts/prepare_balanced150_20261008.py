#!/usr/bin/env python3
"""Prepare the second 150-image coverage comparison; do not train.

Reuse 25 short-sleeve outerwear, 25 long-sleeve outerwear and all 50 shorts
images from targeted150. Add 50 distinct dress images (13/12/12/13 across
native DF2 dress categories 10/11/12/13). Retain every usable eight-class
instance, with independent manifests, masks and receipts. Exact duplicate
screening does not certify near-duplicate or product-level independence.
Requires prepare_targeted_df2_20261008.py beside this file.
"""
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import prepare_targeted_df2_20261008 as helper
from PIL import Image, ImageDraw

ROOT, DATA = helper.ROOT, helper.DATA
diag, v5 = helper.diag, helper.v5
TAG = "df2_balanced150_20261008"
SEED = 20261008
QUOTAS = {10: 13, 11: 12, 12: 12, 13: 13}


def main():
    inputs = [ROOT / p for p in (
        "configs/prd_8class_train_v5_full_targets.csv",
        "configs/prd_8class_train_v5_targeted150_20261008.csv",
        "configs/prd_8class_val_v2_full_targets.csv",
        "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv",
        "reports/reruns/recovery_20261008/targeted150_preparation/preparation.json",
    )]
    identities = {str(p.relative_to(ROOT)): diag.digest(p) for p in inputs}
    receipt = json.loads(inputs[-1].read_text(encoding="utf-8"))
    if diag.digest(inputs[1]) != receipt["output_sha256"]:
        raise ValueError("First targeted150 manifest differs from its receipt")
    base = helper.read_csv(inputs[0])
    original = helper.read_csv(inputs[1])
    if original[:len(base)] != base:
        raise ValueError("First targeted150 did not preserve baseline rows")
    keep, groups = [], Counter()
    limits = {"short_sleeve_outerwear": 25, "long_sleeve_outerwear": 25, "shorts": 50}
    for entry in receipt["selected"]:
        group = entry["group"]
        if groups[group] < limits[group]:
            keep.append(entry)
            groups[group] += 1
    if dict(groups) != limits:
        raise ValueError("First selection receipt has unexpected image groups")
    kept_paths = {Path(entry["source_image"]).resolve() for entry in keep}
    retained = [row for row in original[len(base):]
                if diag.resolve(row["source_image"], ROOT, DATA).resolve() in kept_paths]
    if len({row["source_image"] for row in retained}) != 100:
        raise ValueError("Retained image count is not 100")

    excluded = {diag.resolve(row["source_image"], ROOT, DATA).resolve()
                for path in inputs[:4] for row in helper.read_csv(path)}
    file_hashes, pixel_hashes = set(), set()
    print("Checking existing image identities...", flush=True)
    for path in sorted(excluded):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        a, b = helper.fingerprints(path, image)
        file_hashes.add(a)
        pixel_hashes.add(b)

    raw = DATA / "raw/train/train"
    pools = {category: [] for category in QUOTAS}
    print("Scanning additional dress candidates...", flush=True)
    for index, path in enumerate(sorted((raw / "annos").glob("*.json")), 1):
        image_path = raw / "image" / (path.stem + ".jpg")
        if image_path.resolve() in excluded or not image_path.is_file():
            continue
        source = json.loads(path.read_text(encoding="utf-8-sig"))
        found = {}
        for key, ann in sorted(source.items()):
            if key.startswith("item") and isinstance(ann, dict):
                category = int(ann.get("category_id", -1))
                if category in QUOTAS and diag.usable_annotation(ann, "DeepFashion2"):
                    found.setdefault(category, key)
        for category, key in found.items():
            pools[category].append((path, image_path, key))
        if index % 20000 == 0:
            print("Scanned:", index, flush=True)

    rng = random.Random(SEED)
    selected, skipped, added, previews = [], [], [], []
    taken = set()
    for category, limit in QUOTAS.items():
        rng.shuffle(pools[category])
        count = 0
        for annotation_path, image_path, target_key in pools[category]:
            if image_path.resolve() in taken:
                continue
            try:
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                file_hash, pixel_hash = helper.fingerprints(image_path, image)
                if file_hash in file_hashes or pixel_hash in pixel_hashes:
                    skipped.append({"image": str(image_path), "reason": "exact duplicate"})
                    continue
                source = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
                targets = []
                for key, ann in sorted(source.items()):
                    if not key.startswith("item") or not isinstance(ann, dict):
                        continue
                    native = int(ann.get("category_id", -1))
                    if native in diag.DF2_CLASS and diag.usable_annotation(ann, "DeepFashion2"):
                        targets.append((key, ann, v5.decode_mask(ann, *image.size)))
                if target_key not in {t[0] for t in targets}:
                    raise ValueError("Selected dress absent after mask validation")
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
                added.append(dict(
                    record_id=f"{TAG}_{len(added)+1:05d}", source_dataset="DeepFashion2",
                    source_image="../fashion_data/raw/train/train/image/" + image_path.name,
                    garment_id=key + "_" + fine, garment_category=diag.DF2_CLASS[native],
                    fine_category=fine, source_category_id=native, annotation_ref=key,
                    mask_path="../fashion_data/" + relative.as_posix(),
                    bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                    bbox_width=x2-x1, bbox_height=y2-y1,
                    image_width=image.width, image_height=image.height, source_manifest=TAG,
                ))
                if key == target_key and count < 3:
                    group = diag.DF2_FINE[category - 1]
                    pad = max(20, int(max(x2-x1,y2-y1)*0.2))
                    crop = (max(0,x1-pad),max(0,y1-pad),min(image.width,x2+pad),min(image.height,y2+pad))
                    whole = image.copy()
                    ImageDraw.Draw(whole).rectangle((x1,y1,x2,y2),outline="lime",width=4)
                    tinted = Image.blend(image,Image.new("RGB",image.size,"lime"),0.35)
                    overlay = Image.composite(tinted,image,mask)
                    for label,picture in (("whole_gt_box",whole),("target_original",image.crop(crop)),("target_gt_mask",overlay.crop(crop))):
                        picture.thumbnail((1200,1200))
                        jpg = io.BytesIO()
                        picture.save(jpg,format="JPEG",quality=90)
                        previews.append((f"{group}/{image_path.stem}_{key}_{label}.jpg",jpg.getvalue()))
            selected.append(dict(group=diag.DF2_FINE[category - 1],source_image=str(image_path),
                                 annotation_ref=target_key,annotation_sha256=diag.digest(annotation_path),
                                 file_sha256=file_hash,rgb_pixel_sha256=pixel_hash))
            taken.add(image_path.resolve())
            file_hashes.add(file_hash)
            pixel_hashes.add(pixel_hash)
            count += 1
            if count == limit:
                break
        if count != limit:
            raise RuntimeError(f"Not enough validated unique dress images: {category}: {count}/{limit}")
        print(diag.DF2_FINE[category-1],"selected:",count,flush=True)

    combined = base + retained + added
    if len({r["record_id"] for r in combined}) != len(combined):
        raise ValueError("Duplicate record IDs")
    if len({r["source_image"] for r in combined}) != len({r["source_image"] for r in base}) + 150:
        raise ValueError("Final image count differs from baseline + 150")
    for path in inputs:
        if diag.digest(path) != identities[str(path.relative_to(ROOT))]:
            raise RuntimeError("An input changed during preparation")
    manifest = ROOT / "configs/prd_8class_train_v5_balanced150_20261008.csv"
    v5.same_or_write(manifest,v5.csv_bytes(combined))
    report = dict(status="PREPARED_NOT_TRAINED",seed=SEED,retained_images=100,new_dress_images=50,
                  train_images=len({r["source_image"] for r in combined}),total_instances=len(combined),
                  classes=dict(Counter(r["garment_category"] for r in combined)),
                  input_sha256=identities,output_sha256=diag.digest(manifest),
                  retained_selection=keep,dress_selection=selected,skipped=skipped,
                  duplicate_check="Source paths, exact bytes and RGB pixels; near duplicates not certified")
    encoded = (json.dumps(report,ensure_ascii=False,indent=2)+"\n").encode()
    v5.same_or_write(ROOT/"reports/reruns/recovery_20261008/balanced150_preparation/preparation.json",encoded)
    review = ROOT.parent/"v5_balanced150_candidate_review.zip"
    with zipfile.ZipFile(review,"w",zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("preparation.json",encoded)
        for name,content in previews:
            archive.writestr(name,content)
    print("New manifest:",manifest)
    print("Training images:",report["train_images"],"targets:",len(combined))
    print("Class counts:",report["classes"])
    print("Upload review ZIP:",review)
    print("No training started. Both previous runs and evaluation manifests preserved.")


if __name__ == "__main__":
    main()
