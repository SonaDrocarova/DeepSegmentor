import os
import numpy as np
import torch
import cv2
from collections import defaultdict
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import argparse  

from skimage.morphology import skeletonize

from split_sperm_parts import split_head_mid_tail, colorize_parts, save_intensity_profile_plot, skeleton_to_polylines
import json
import math
from skimage.measure import label





def stitch_predictions(patch_preds, patch_coords, img_shape, patch_size):
    h, w = img_shape
    output_sum = np.zeros((h, w), dtype=np.float32)
    count_sum = np.zeros((h, w), dtype=np.float32)

    for pred, (x, y) in zip(patch_preds, patch_coords):
        output_sum[y:y+patch_size, x:x+patch_size] += pred
        count_sum[y:y+patch_size, x:x+patch_size] += 1

    count_sum[count_sum == 0] = 1
    return output_sum / count_sum

def apply_mask_overlay(original_img, binary_mask, alpha=0.4):
    mask_colored = np.zeros_like(original_img)
    mask_colored[:, :, 2] = binary_mask  # red channel

    overlay = cv2.addWeighted(original_img, 1.0, mask_colored, alpha, 0)
    return overlay

def postprocess_predictions(predictions, object_size_threshold=370, kernel_size=3, iterations=2):
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # predictions = cv2.morphologyEx(predictions, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    # Remove small objects
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(predictions, connectivity=8)
    sizes = stats[1:, -1]
    img2 = np.zeros_like(predictions)
    for i in range(0, num_labels - 1):
        if sizes[i] >= object_size_threshold:
            img2[labels == i + 1] = 255
    return img2

#----------writing to cvat----
def write_cvat_xml_dataset(items, out_xml_path):
    """
    items: list of dicts:
      {'name': <image filename>,
       'width': int, 'height': int,
       'label_to_polys': Dict[str, List[List[Tuple[int,int]]]]}
    """
    import xml.etree.ElementTree as ET

    ann = ET.Element('annotations')
    ET.SubElement(ann, 'version').text = '1.1'

    # Labels (union over all images)
    label_set = set()
    for it in items:
        label_set.update(it.get('label_to_polys', {}).keys())

    meta = ET.SubElement(ann, 'meta')
    task = ET.SubElement(meta, 'task')
    labels_el = ET.SubElement(task, 'labels')
    for lbl in sorted(label_set):
        lbl_el = ET.SubElement(labels_el, 'label')
        ET.SubElement(lbl_el, 'name').text = lbl
        ET.SubElement(lbl_el, 'attributes')  # empty placeholder

    # Images and their polylines
    for idx, it in enumerate(items):
        img_el = ET.SubElement(ann, 'image', {
            'id': str(idx),
            'name': it['name'],
            'width': str(it['width']),
            'height': str(it['height']),
        })
        for lbl, polys in it.get('label_to_polys', {}).items():
            for poly in polys:
                pts = ';'.join(f'{x},{y}' for (x, y) in poly)
                ET.SubElement(img_el, 'polyline', {
                    'label': lbl,
                    'points': pts,
                    'occluded': '0',
                    'z_order': '0',
                    'group_id': '0',
                    'source': 'auto',
                })

    os.makedirs(os.path.dirname(out_xml_path), exist_ok=True)
    ET.ElementTree(ann).write(out_xml_path, encoding='utf-8', xml_declaration=True)

import json, os

def load_global_um_per_px(calib_filename: str = "scale_calibration.json") -> float:
    """
    Reads scale_calibration.json from the same directory as this script and
    returns um_per_px from the FIRST entry.

    Matches JSON like:
    {
      "some_image.jpg": { "um_per_px": 0.0862, ... }
    }
    """
    calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), calib_filename)

    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Missing calibration file: {calib_path}")

    with open(calib_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Calibration JSON is empty/invalid: {calib_path}")

    first_key = next(iter(data.keys()))
    entry = data[first_key]

    if not isinstance(entry, dict) or "um_per_px" not in entry:
        raise ValueError(
            f"Expected first entry '{first_key}' to contain 'um_per_px'. "
            f"Got keys: {list(entry.keys()) if isinstance(entry, dict) else type(entry)}"
        )

    return float(entry["um_per_px"])

def skeleton_length_px(skel_mask: np.ndarray) -> float:
    """
    Compute length of a 1-pixel skeleton using 8-neighborhood step weights.
    Counts each undirected edge once.
    """
    skel = skel_mask.astype(bool)
    ys, xs = np.where(skel)
    if ys.size == 0:
        return 0.0

    pts = set(zip(ys.tolist(), xs.tolist()))
    diag = math.sqrt(2)

    # (dy, dx, weight)
    nbrs = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag),
    ]

    length = 0.0
    for (y, x) in pts:
        for dy, dx, w in nbrs:
            ny, nx = y + dy, x + dx
            if (ny, nx) in pts:
                # count each edge once
                if (ny, nx) > (y, x):
                    length += w
    return float(length)


def annotate_part_lengths_on_overlay(
    parts_overlay_bgr: np.ndarray,
    part_mask: np.ndarray,
    um_per_px: float,
    font_scale: float = 0.65,
    thickness: int = 2,
) -> np.ndarray:
    """
    Annotate the parts overlay with per-part lengths (in µm) for each connected object.

    Inputs:
      - parts_overlay_bgr: image you want to draw text on (BGR)
      - part_mask: uint8 image with values:
            0 = background
            1 = head
            2 = midpiece
            3 = tail
        IMPORTANT: part_mask should be present only on skeleton pixels (as in your pipeline).
      - um_per_px: conversion (µm per pixel), loaded once from scale_calibration.json :contentReference[oaicite:1]{index=1}

    Behavior:
      - Finds connected components on (part_mask > 0) => individual sperm objects
      - For each object, measures each part length along skeleton
      - Writes "H:xx.xµm", "M:xx.xµm", "T:xx.xµm" near that part in matching color
    """
    out = parts_overlay_bgr.copy()

    # Match your color scheme (BGR): head red, mid green, tail blue
    part_info = {
        1: ("H", (0, 0, 255)),   # red
        2: ("M", (0, 255, 0)),   # green
        3: ("T", (255, 0, 0)),   # blue
    }

    # Label connected components across the whole skeleton (any part)
    comp = label(part_mask > 0, connectivity=2)
    n_comp = int(comp.max())
    if n_comp == 0:
        return out

    H, W = out.shape[:2]

    for lab_id in range(1, n_comp + 1):
        obj_mask = (comp == lab_id)

        for part_val, (abbr, color) in part_info.items():
            pm = obj_mask & (part_mask == part_val)
            if not np.any(pm):
                continue

            # length in px and µm
            length_px = skeleton_length_px(pm)
            length_um = length_px * um_per_px

            # Place text near the part: use centroid of that part's pixels
            ys, xs = np.where(pm)
            cy = int(np.round(np.mean(ys)))
            cx = int(np.round(np.mean(xs)))

            # Offset a bit so the text doesn't sit exactly on the line
            tx = int(np.clip(cx + 10, 0, W - 1))
            ty = int(np.clip(cy - 10, 0, H - 1))

            text = f"{abbr}:{length_um:.1f}µm"

            # Optional readability: draw a thin dark outline behind text
            cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness, cv2.LINE_AA)

    return out


if __name__ == '__main__':
    # --------------------------- Custom arguments ---------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for morphological operations')
    parser.add_argument('--iterations', type=int, default=2, help='Number of iterations for morphological operations')
    parser.add_argument('--object_size_threshold', type=int, default=2500, help='Threshold for small object removal')
    parser.add_argument('--mask_threshold', type=float, default=0.38, help='Threshold for mask creation')
    parser.add_argument('--save_results', type=bool, default=True, help='Save results')

    # Profile plot switch (use proper boolean flags)
    parser.add_argument('--save_profiles', dest='save_profiles', action='store_true',
                        help='Save smoothed intensity profile plots with split markers')
    parser.add_argument('--no-save_profiles', dest='save_profiles', action='store_false',
                        help='Do not save intensity profile plots')
    parser.set_defaults(save_profiles=True)

    # CVAT export switches (single XML for whole dataset)
    parser.add_argument('--save_cvat', dest='save_cvat', action='store_true',
                        help='Export polylines to a single CVAT XML for the whole dataset')
    parser.add_argument('--no-save_cvat', dest='save_cvat', action='store_false',
                        help='Do not export CVAT XML')
    parser.set_defaults(save_cvat=True)

    parser.add_argument('--cvat_parts', dest='cvat_parts', action='store_true',
                        help='Use separate labels head/midpiece/tail')
    parser.add_argument('--no-cvat_parts', dest='cvat_parts', action='store_false',
                        help='Use a single label for all polylines')
    parser.set_defaults(cvat_parts=True)

    parser.add_argument('--cvat_label', type=str, default='sperm',
                        help='Label name when not splitting into parts')
    parser.add_argument('--cvat_xml_path', type=str, default='cvat_annotations.xml',
                        help='Output path for the dataset-level CVAT XML')

    custom_args, unknown = parser.parse_known_args()
    # -----------------------------------------------------------------------

    # ---------------------------- Existing options -------------------------
    opt = TestOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.use_augment = False
    opt.display_id = -1
    patch_size = 256

    # Collect dataset-wide CVAT items here
    cvat_items = []

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    output_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    overlay_dir = os.path.join(output_dir, 'final_thresholded')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    grouped_preds = defaultdict(list)
    grouped_coords = defaultdict(list)
    grouped_shapes = {}
    # -----------------------------------------------------------------------

    print("Running inference on patches...")
    um_per_px = load_global_um_per_px("scale_calibration.json")
    print(f"[Calibration] Using um_per_px={um_per_px:.8f} from scale_calibration.json")

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        img_path = data['A_paths'][0]
        coord = data['coord']
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        pred_logit = visuals['fused'].squeeze().cpu()
        pred_prob  = torch.sigmoid(pred_logit).numpy()  # now in [0,1]
        grouped_preds[img_path].append(pred_prob)
        grouped_coords[img_path].append(coord)

        if img_path not in grouped_shapes:
            original_img = cv2.imread(img_path)
            grouped_shapes[img_path] = original_img.shape[:2]

    print("Stitching predictions and saving results...")

    for img_path in grouped_preds:
        name = os.path.splitext(os.path.basename(img_path))[0]
        pred_patches = grouped_preds[img_path]
        coords = grouped_coords[img_path]
        img_shape = grouped_shapes[img_path]

        full_pred = stitch_predictions(pred_patches, coords, img_shape, patch_size)
        full_pred = np.clip(full_pred, 0.0, 1.0)
        print("full_pred min/max:", float(full_pred.min()), float(full_pred.max()))

        binary_mask = (full_pred > custom_args.mask_threshold).astype(np.uint8) * 255
        binary_mask = postprocess_predictions(
            binary_mask,
            object_size_threshold=custom_args.object_size_threshold,
            kernel_size=custom_args.kernel_size,
            iterations=custom_args.iterations
        )
        grayscale_pred = (full_pred * 255).astype(np.uint8)

        original_img = cv2.imread(img_path)
        overlay_img = apply_mask_overlay(original_img, binary_mask)

        # --- Split into head / mid / tail on the stitched mask ---
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        skel_bool = skeletonize((binary_mask > 0).astype(bool))
        skeleton = (skel_bool.astype(np.uint8)) * 255

        if custom_args.save_profiles:
            part_mask, profiles = split_head_mid_tail(gray_img, skeleton, return_profiles=True)
        else:
            part_mask = split_head_mid_tail(gray_img, skeleton)
            profiles = None

        parts_color = colorize_parts(part_mask)
        parts_overlay = cv2.addWeighted(original_img, 1.0, parts_color, 0.7, 0)

        parts_overlay_with_lengths = annotate_part_lengths_on_overlay(
            parts_overlay_bgr=parts_overlay,
            part_mask=part_mask,
            um_per_px=um_per_px
            )          


        # Save intensity profiles (if enabled)
        if custom_args.save_profiles and profiles is not None:
            for k, p in enumerate(profiles):
                sm = p.get('smoothed', None)
                head = p.get('head', -1)
                tail = p.get('tail', -1)
                if sm is not None and len(sm) > 0:
                    out_plot = os.path.join(overlay_dir, f'{name}_profile_{k:02d}.png')
                    save_intensity_profile_plot(sm, head, tail, out_plot)
                    print(f"Saved intensity profile plot: {out_plot}")

        # --- Build polylines (for CVAT) ---
        label_to_polys = {}
        if custom_args.cvat_parts:
            for val, lbl in [(1, 'Head'), (2, 'Midpiece'), (3, 'Tail')]:
                pmask = (part_mask == val).astype(np.uint8) * 255
                polys = skeleton_to_polylines(pmask, min_points=5)
                if polys:
                    label_to_polys[lbl] = polys
        else:
            polys = skeleton_to_polylines(skeleton, min_points=5)
            if polys:
                label_to_polys[custom_args.cvat_label] = polys

        print(f"[CVAT] {name}: labels={list(label_to_polys.keys())}, counts={[len(v) for v in label_to_polys.values()]}")

        # Accumulate for dataset-level export
        if custom_args.save_cvat:
            h, w = original_img.shape[:2]
            cvat_items.append({
                'name': os.path.basename(img_path),
                'width': w,
                'height': h,
                'label_to_polys': label_to_polys
            })

        # --- Save per-image outputs ---
        cv2.imwrite(os.path.join(overlay_dir, f'{name}_parts_overlay.png'), parts_overlay)
        cv2.imwrite(os.path.join(output_dir, f'{name}.png'), original_img)
        cv2.imwrite(os.path.join(output_dir, f'{name}_label_viz.png'), binary_mask)
        cv2.imwrite(os.path.join(output_dir, f'{name}_fused.png'), grayscale_pred)
        cv2.imwrite(os.path.join(overlay_dir, f'{name}_overlay.png'), overlay_img)
        cv2.imwrite(os.path.join(overlay_dir, f"{name}_parts_overlay_lengths.png"), parts_overlay_with_lengths)


        print(f"Saved: {name}_label_viz.png, {name}_fused.png, {name}_overlay.png")

    # ----------------- Write one CVAT XML after the loop --------------------
    if custom_args.save_cvat:
        if len(cvat_items) == 0:
            print("[CVAT] No items to export (no polylines found).")
        else:
            out_xml = custom_args.cvat_xml_path
            if not os.path.isabs(out_xml):
                out_xml = os.path.join(overlay_dir, custom_args.cvat_xml_path)
            write_cvat_xml_dataset(cvat_items, out_xml)
            print(f"[CVAT] Saved dataset annotations: {out_xml}")

    print("✅ All done.")
