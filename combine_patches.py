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



if __name__ == '__main__':
    # Custom arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for morphological operations')
    parser.add_argument('--iterations', type=int, default=2, help='Number of iterations for morphological operations')
    parser.add_argument('--object_size_threshold', type=int, default=2500, help='Threshold for small object removal')
    parser.add_argument('--mask_threshold', type=float, default=0.38, help='Threshold for mask creation')
    parser.add_argument('--save_results', type=bool, default=True, help='Save results')
    # ADD to your argparse block (keep style consistent with your other bools)
    parser.add_argument('--save_profiles', type=bool, default=True,
                        help='Save smoothed intensity profile plots with split markers')
    # parser.add_argument('--cvat_parts', type=bool, default=True,
                        help='Save CVAT XML with head/midpiece/tail polygons')

    custom_args, unknown = parser.parse_known_args()
    # print(f"[debug] save_profiles = {custom_args.save_profiles}")

    # Existing options
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



    print("Running inference on patches...")

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
        # map logits -> probabilities in [0,1]
        # full_pred = 1.0 / (1.0 + np.exp(-full_pred))

        full_pred = np.clip(full_pred, 0.0, 1.0)
        print("full_pred min/max:", float(full_pred.min()), float(full_pred.max()))


        # print("full_pred min/max after remap:", full_pred.min(), full_pred.max())

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
        
        # --- NEW: split into head / mid / tail on the stitched mask ---
        # Prepare grayscale for intensity sampling
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        # Make a 1-pixel skeleton from the binary mask
        # skimage.skeletonize expects boolean; result is bool -> cast to uint8*255
        skel_bool = skeletonize((binary_mask > 0).astype(bool))
        skeleton = (skel_bool.astype(np.uint8)) * 255

        # WITH this block:
        if custom_args.save_profiles:
            part_mask, profiles = split_head_mid_tail(gray_img, skeleton, return_profiles=True)
        else:
            part_mask = split_head_mid_tail(gray_img, skeleton)
            profiles = None

        # Pretty visualization of parts
        parts_color = colorize_parts(part_mask)

        # Optional: overlay colored parts on original image
        parts_overlay = cv2.addWeighted(original_img, 1.0, parts_color, 0.7, 0)

        # Save intensity profiles (if enabled)
        if custom_args.save_profiles and profiles is not None:
            for k, p in enumerate(profiles):
                sm = p.get('smoothed', None)
                head = p.get('head', -1)
                tail = p.get('tail', -1)
                # Save even if head/tail are -1; skip only if smoothed is empty
                if sm is not None and len(sm) > 0:
                    out_plot = os.path.join(overlay_dir, f'{name}_profile_{k:02d}.png')
                    save_intensity_profile_plot(sm, head, tail, out_plot)
                    print(f"Saved intensity profile plot: {out_plot}")
        # # --- NEW: build polylines and (optionally) export CVAT XML ---
        # label_to_polys = {}
        # if custom_args.cvat_parts:
        #     # separate polylines for head/mid/tail using part_mask
        #     for val, lbl in [(1, 'head'), (2, 'midpiece'), (3, 'tail')]:
        #         pmask = (part_mask == val).astype(np.uint8) * 255
        #         polys = skeleton_to_polylines(pmask, min_points=5)
        #         if polys:
        #             label_to_polys[lbl] = polys
        # else:
        #     # single label from whole skeleton
        #     polys = skeleton_to_polylines(skeleton, min_points=5)
        #     if polys:
        #         label_to_polys[custom_args.cvat_label] = polys

        # # Accumulate for dataset-level export
        # if custom_args.save_cvat:
        #     h, w = original_img.shape[:2]
        #     cvat_items.append({
        #         'name': os.path.basename(img_path),
        #         'width': w,
        #         'height': h,
        #         'label_to_polys': label_to_polys
        #     })
        # # After all images are processed, write a single CVAT XML if requested
        # if custom_args.save_cvat:
        #     if len(cvat_items) == 0:
        #         print("[CVAT] No items to export (no polylines found).")
        #     else:
        #         # If cvat_xml_path is relative, put it next to your overlays of the last run
        #         out_xml = custom_args.cvat_xml_path
        #         if not os.path.isabs(out_xml):
        #             # default to the same directory where overlays are saved for this run
        #             out_xml = os.path.join(overlay_dir if 'overlay_dir' in locals() else os.getcwd(),
        #                                 custom_args.cvat_xml_path)
        #         write_cvat_xml_dataset(cvat_items, out_xml)
        #         print(f"[CVAT] Saved dataset annotations: {out_xml}")




        # cv2.imwrite(os.path.join(output_dir, f'{name}_skeleton.png'), skeleton)
        # cv2.imwrite(os.path.join(output_dir, f'{name}_parts_mask.png'), part_mask)
        # cv2.imwrite(os.path.join(output_dir, f'{name}_parts_color.png'), parts_color)
        cv2.imwrite(os.path.join(overlay_dir, f'{name}_parts_overlay.png'), parts_overlay)

        # Save outputs
        cv2.imwrite(os.path.join(output_dir, f'{name}.png'), original_img)
        cv2.imwrite(os.path.join(output_dir, f'{name}_label_viz.png'), binary_mask)
        cv2.imwrite(os.path.join(output_dir, f'{name}_fused.png'), grayscale_pred)
        cv2.imwrite(os.path.join(overlay_dir, f'{name}_overlay.png'), overlay_img)

        print(f"Saved: {name}_label_viz.png, {name}_fused.png, {name}_overlay.png")

    print("✅ All done.")
