#!/usr/bin/env python3
"""
Perception2YOLO.py

Preserved behavior:
- Scans all JSONs in two passes to infer labelId → labelName when available.
- Normalizes label strings (trim, lowercase, basic plural → singular conversion).
- Forces the canonical class order (1:bottle, 2:cup, 3:book, 4:laptop, 5:chair) and writes 0-based indices.
- Falls back to numeric labelId (with a 1-based → 0-based assumption) when necessary.
- Emits YOLO-style label files with indices in range 0..4.
- Creates the same outputs: images/{train,val}, labels/{train,val}, classes.txt, dataset.yaml

Run: python Perception2YOLO.py -v
"""
from pathlib import Path
import json
import random
import shutil
import os
import argparse
from typing import Any, List, Tuple, Dict, Optional
from PIL import Image

# directories
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "yolo_synthetic_dataset"

# canonical order the user requested (human 1..5 -> we use 0..4)
PREFERRED_ORDER = ["bottle", "cup", "book", "laptop", "chair"]
NAME_TO_IDX = {n: i for i, n in enumerate(PREFERRED_ORDER)}


def list_class_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')])


def list_sequences(class_dir: Path) -> List[Path]:
    return sorted([p for p in class_dir.iterdir() if p.is_dir() and not p.name.startswith('.')])


def locate_pair(sequence_dir: Path) -> Optional[Tuple[Path, Path]]:
    # prefer "*.camera.png" then "*.png"; prefer "*frame_data.json" then other jsons
    imgs = list(sequence_dir.glob("*.camera.png")) or list(sequence_dir.glob("*.png"))
    jsn = (list(sequence_dir.glob("*frame_data.json"))
           or list(sequence_dir.glob("*.frame_data.json"))
           or list(sequence_dir.glob("*.json")))
    if not imgs or not jsn:
        return None
    return (imgs[0], jsn[0])


def read_json(path: Path) -> Any:
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def normalize_label(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip().lower()
    if not s:
        return None
    # naive plural -> singular: "books" -> "book", "batteries" -> "battery"
    if s.endswith('ies'):
        s = s[:-3] + 'y'
    elif s.endswith('s') and not s.endswith('ss'):
        s = s[:-1]
    s = s.replace(' ', '_')
    return s


def extract_annotations(parsed: Any, img_size: Tuple[int, int]) -> List[Tuple[Any, Any, float, float, float, float]]:
    out: List[Tuple[Any, Any, float, float, float, float]] = []
    if not isinstance(parsed, dict):
        return out
    captures = parsed.get('captures')
    if not isinstance(captures, list):
        return out
    w, h = img_size
    for cap in captures:
        if not isinstance(cap, dict):
            continue
        anns = cap.get('annotations')
        if not isinstance(anns, list):
            continue
        for ann in anns:
            vals = ann.get('values')
            if not isinstance(vals, list):
                continue
            for v in vals:
                if not isinstance(v, dict):
                    continue
                # Accept both labelId / label_id and labelName / label
                lid = v.get('labelId') if 'labelId' in v else v.get('label_id')
                lname = v.get('labelName') or v.get('label') or None
                origin = v.get('origin')
                dimension = v.get('dimension')
                if not (isinstance(origin, list) and isinstance(dimension, list) and len(origin) >= 2 and len(dimension) >= 2):
                    continue
                left = float(origin[0])
                top = float(origin[1])
                width_px = float(dimension[0])
                height_px = float(dimension[1])
                cx = (left + width_px / 2.0) / w
                cy = (top + height_px / 2.0) / h
                nw = width_px / w
                nh = height_px / h
                out.append((lid, lname, cx, cy, nw, nh))
    return out


def ensure_output_dirs(root: Path) -> None:
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        (root / sub).mkdir(parents=True, exist_ok=True)


def transfer_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'copy':
        shutil.copy2(src, dst)
        return
    if mode == 'symlink':
        try:
            if dst.exists():
                dst.unlink()
            os.symlink(src.resolve(), dst)
            return
        except Exception:
            shutil.copy2(src, dst)
            return
    raise ValueError('mode must be copy or symlink')


def dump_yolo_label(path: Path, annots: List[Tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for cls_idx, x, y, w, h in annots:
        lines.append(f"{cls_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    with open(path, 'w', encoding='utf-8') as fh:
        if lines:
            fh.write('\n'.join(lines) + '\n')
        else:
            fh.write('')


def main(val_ratio: float, mode: str, include_empty: bool, verbose: bool) -> None:
    # discover class directories residing next to the script
    discovered = list_class_dirs(BASE_DIR)
    if not discovered:
        print('No class folders found next to the script. Put your class folders (cup, book, etc.) next to this script and run again.')
        return

    discovered_names = [p.name for p in discovered]
    # final classes: keep requested order first, then append any extras found
    classes: List[str] = [n for n in PREFERRED_ORDER if n in discovered_names]
    classes += [n for n in discovered_names if n not in classes]
    class_to_index = {n: i for i, n in enumerate(classes)}
    print('Detected classes (final order):', classes)

    # aggregate image/json pairs
    items: List[Tuple[Path, Path, str]] = []
    for cls in classes:
        class_dir = BASE_DIR / cls
        if not class_dir.exists() or not class_dir.is_dir():
            if verbose:
                print(f'skipping missing class folder: {cls}')
            continue
        seqs = list_sequences(class_dir)
        for seq in seqs:
            pair = locate_pair(seq)
            if pair is None:
                if verbose:
                    print(f'skipping sequence (no pair): {seq}')
                continue
            img_p, json_p = pair
            items.append((img_p, json_p, cls))

    if not items:
        print('No image+json pairs found in any sequences.')
        return

    # pass 1: build labelId -> canonical label name map
    id_to_name: Dict[int, str] = {}
    for img_p, json_p, _ in items:
        try:
            parsed = read_json(json_p)
        except Exception:
            continue
        captures = parsed.get('captures') if isinstance(parsed, dict) else None
        if not captures or not isinstance(captures, list):
            continue
        for cap in captures:
            anns = cap.get('annotations') if isinstance(cap, dict) else None
            if not anns or not isinstance(anns, list):
                continue
            for ann in anns:
                vals = ann.get('values') if isinstance(ann, dict) else None
                if not vals or not isinstance(vals, list):
                    continue
                for v in vals:
                    if not isinstance(v, dict):
                        continue
                    lid = v.get('labelId') if 'labelId' in v else v.get('label_id')
                    lname = v.get('labelName') or v.get('label') or None
                    if lid is not None and lname is not None:
                        try:
                            lid_int = int(lid)
                            cname = normalize_label(lname)
                            if cname:
                                id_to_name[lid_int] = cname
                        except Exception:
                            pass
    if verbose:
        print('Discovered labelId->name mapping (from JSONs):', id_to_name)

    # make folders and classes list file
    ensure_output_dirs(OUT_DIR)
    with open(OUT_DIR / 'classes.txt', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(classes) + '\n')

    random.shuffle(items)
    split_idx = int(len(items) * (1.0 - val_ratio))

    for i, (img_p, json_p, seq_class) in enumerate(items):
        split = 'train' if i < split_idx else 'val'
        try:
            with Image.open(img_p) as im:
                iw, ih = im.size
        except Exception as e:
            if verbose:
                print(f'failed to open image {img_p}: {e}')
            continue

        try:
            parsed = read_json(json_p)
        except Exception as e:
            if verbose:
                print(f'failed to load json {json_p}: {e}')
            parsed = {}

        parsed_annots = extract_annotations(parsed, (iw, ih))

        final_annots: List[Tuple[int, float, float, float, float]] = []
        for lid, lname, cx, cy, nw, nh in parsed_annots:
            chosen_idx: Optional[int] = None
            # prefer label name canonicalization -> requested mapping
            c_name = normalize_label(lname)
            if c_name and c_name in NAME_TO_IDX:
                chosen_idx = NAME_TO_IDX[c_name]
            # else if numeric id present, consult discovered id->name map or assume 1-based mapping
            if chosen_idx is None and lid is not None:
                try:
                    lid_int = int(lid)
                    if lid_int in id_to_name:
                        candidate = id_to_name[lid_int]
                        if candidate in NAME_TO_IDX:
                            chosen_idx = NAME_TO_IDX[candidate]
                    else:
                        # fallback: assume labelId is 1-based index into PREFERRED_ORDER
                        if 1 <= lid_int <= len(PREFERRED_ORDER):
                            chosen_idx = lid_int - 1
                except Exception:
                    pass
            # fallback: use sequence folder's class index
            if chosen_idx is None:
                chosen_idx = class_to_index.get(seq_class, 0)

            # sanity checks: only allow indices inside requested range
            if chosen_idx < 0:
                chosen_idx = 0
            if chosen_idx >= len(PREFERRED_ORDER):
                if verbose:
                    print(f'skipping annotation with mapped_idx={chosen_idx} for {img_p}')
                continue

            final_annots.append((chosen_idx, cx, cy, nw, nh))

        if not final_annots and not include_empty:
            if verbose:
                print(f'skipping {img_p} (no annotations found)')
            continue

        uniq = str(i)
        dst_img = OUT_DIR / f"images/{split}/{uniq}{img_p.suffix}"
        dst_lbl = OUT_DIR / f"labels/{split}/{uniq}.txt"

        transfer_file(img_p, dst_img, mode)
        dump_yolo_label(dst_lbl, final_annots)

        if verbose:
            print(f'[{split}] {img_p} -> {dst_img} labels={len(final_annots)}')

    # dataset.yaml with the requested order
    yaml_path = OUT_DIR / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as yf:
        yf.write("names:\n")
        for nm in PREFERRED_ORDER:
            yf.write(f"  - {nm}\n")
        yf.write(f"\nnc: {len(PREFERRED_ORDER)}\n")
        yf.write(f"\ntrain: {OUT_DIR}/images/train\n")
        yf.write(f"val: {OUT_DIR}/images/val\n")

    print('Done. Output written to', OUT_DIR)
    print('Classes file:', OUT_DIR / 'classes.txt')
    print('YAML file:', yaml_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto Unity Perception -> YOLO dataset (no path args required)')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Fraction reserved for validation')
    parser.add_argument('--mode', choices=('copy','symlink'), default='copy', help='How to transfer images (default copy)')
    parser.add_argument('--include-empty', action='store_true', help='Keep images without annotations')
    parser.add_argument('-v','--verbose', action='store_true')
    args = parser.parse_args()
    main(args.val_ratio, args.mode, args.include_empty, args.verbose)
