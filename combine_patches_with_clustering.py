# !/usr/bin/env python3
import os
import sys
import argparse
from collections import defaultdict

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from options.test_options import TestOptions
from data import create_dataset
from models import create_model

matplotlib.use("Agg")

# ------------------------- Utilities (stitching/overlay/postproc) -------------------------
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



# ------------------------- New: feature computation, clustering, viz -------------------------
def compute_features(binary_mask):
    """
    Returns:
      df: DataFrame with per-object features (label_id, skeleton_length, thickness, branchpoints, endpoints, etc.)
      labels_img: labeled image (0 background, 1..N components)
    """
    labels_img = label(binary_mask > 0, connectivity=2)
    props = regionprops(labels_img)
    rows = []

    for p in props:
        area = float(p.area)
        perim = float(p.perimeter) if p.perimeter > 0 else 1.0
        comp_mask = (labels_img == p.label).astype(np.uint8)

        # --- replace your "Skeleton-based metrics" block with this ---
        # Skeleton-based metrics (+ tortuosity)
        skel = skeletonize(comp_mask > 0)
        skel_len = float(skel.sum())

        # Neighborhood count to get endpoints/branchpoints
        nbrs = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
        endpoints = np.logical_and(skel, nbrs == 2)     # self + 1 neighbor
        branchpoints = np.logical_and(skel, nbrs >= 4)  # self + >=3 neighbors
        n_endpoints = int(endpoints.sum())
        n_branchpoints = int(branchpoints.sum())

        thickness = area / max(skel_len, 1.0)

        # --- NEW: tortuosity ---
        # tortuosity = skeleton_length / straight-line distance between the two farthest endpoints
        # (if fewer than 2 endpoints, fall back to 1.0 to avoid division by zero)
        tortuosity = 1.0
        ys, xs = np.where(endpoints)
        if len(xs) >= 2:
            # print(len(xs))
            coords = np.stack([ys, xs], axis=1)
            # farthest pair by Euclidean distance
            d2 = ((coords[None, :, :] - coords[:, None, :]) ** 2).sum(-1)
            i, j = np.unravel_index(np.argmax(d2), d2.shape)
            euclid = float(np.sqrt(d2[i, j]))
            tortuosity = skel_len / max(euclid, 1.0)


        # Extra shape cues (not used in current clustering but saved to CSV)
        minr, minc, maxr, maxc = p.bbox
        h, w = (maxr - minr), (maxc - minc)
        bbox_area = max(h * w, 1)
        rectangularity = area / bbox_area
        aspect_ratio = (max(h, w) / max(min(h, w), 1))
        eccentricity = float(getattr(p, 'eccentricity', 0.0))
        convex_area = max(float(getattr(p, 'convex_area', area)), 1.0)
        solidity = area / convex_area
        circularity = 4.0 * np.pi * area / (perim ** 2)

        rows.append(dict(
            label_id=int(p.label),
            area=area,
            perimeter=perim,
            rectangularity=float(rectangularity),
            aspect_ratio=float(aspect_ratio),
            eccentricity=float(eccentricity),
            solidity=float(solidity),
            circularity=float(circularity),
            skeleton_length=float(skel_len),
            thickness=float(thickness),
            branchpoints=float(n_branchpoints),
            endpoints=float(n_endpoints),
            tortuosity=float(tortuosity),   

        ))

    df = pd.DataFrame(rows)
    return df, labels_img


# ------------------------- Main pipeline -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for morphological operations')
    parser.add_argument('--iterations', type=int, default=2, help='Number of iterations for morphological operations')
    parser.add_argument('--object_size_threshold', type=int, default=3000, help='Threshold for small object removal')
    parser.add_argument('--mask_threshold', type=float, default=0.52, help='Threshold for mask creation')
    parser.add_argument('--save_results', type=bool, default=True, help='Save results')
    custom_args, unknown = parser.parse_known_args()
    chosen_features=['tortuosity', 'area', 'thickness']
    num_clusters = 2  # <--- Set your desired number of clusters here
    # Existing options
    opt = TestOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.use_augment = False
    opt.display_id = -1
    patch_size = 256

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

        grayscale_pred = visuals['fused'].squeeze().cpu().numpy()
        grouped_preds[img_path].append(grayscale_pred)
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
        full_pred = 1 + full_pred

        print("full_pred min/max after remap:", full_pred.min(), full_pred.max())

        binary_mask = (full_pred > custom_args.mask_threshold).astype(np.uint8) * 255
        df, labels_img = compute_features(binary_mask)
        print(f"Computed features for {len(df)} objects.")

        # --- Remove small objects based on area threshold ---
        min_area = custom_args.object_size_threshold
        keep_labels = df[df['area'] >= min_area]['label_id'].values
        mask_filtered = np.isin(labels_img, keep_labels).astype(np.uint8) * 255
        # Recompute features on filtered mask
        df, labels_img = compute_features(mask_filtered)
        print(f"After filtering, {len(df)} objects remain (area >= {min_area}).")

        #perform clustering on the extracted features
        if len(df) >= num_clusters:
            feature_data = df[chosen_features].values
            scaler = StandardScaler()
            feature_data_scaled = scaler.fit_transform(feature_data)

            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(feature_data_scaled)

            #plot 3D clustering in one figure with binary mask with colored objects
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122)
            scatter = ax1.scatter(
                df[chosen_features[0]], df[chosen_features[1]], df[chosen_features[2]],
                c=df['cluster'], cmap='viridis', s=50
            )
            ax1.set_xlabel('Tortuosity')
            ax1.set_ylabel('Area')
            ax1.set_zlabel('Thickness')
            ax1.set_title('3D Clustering of Objects')
            legend1 = ax1.legend(*scatter.legend_elements(), title="Clusters")
            ax1.add_artist(legend1) 
            #colored mask with cluster ids
            colored_mask = np.zeros((*labels_img.shape, 3), dtype=np.uint8)
            # Generate colors for clusters
            colors = plt.cm.get_cmap('viridis', num_clusters)
            for cluster_id in range(num_clusters):
                color = tuple((np.array(colors(cluster_id)[:3]) * 255).astype(int))
                for label_id in df[df['cluster'] == cluster_id]['label_id']:
                    colored_mask[labels_img == label_id] = color
            ax2.imshow(colored_mask)
            ax2.set_title('Clustered Objects Overlay')
            ax2.axis('off')
            plt.tight_layout()
            cluster_viz_path = os.path.join(output_dir, f'{name}_clusters.png')
            plt.savefig(cluster_viz_path)
            plt.close()

            # target cluster 0
            target_cluster  = 0
            target_objects = df[(df['cluster'] == target_cluster)]['label_id']
            refined_mask = np.isin(labels_img, target_objects).astype(np.uint8) * 255

            #plot the mask and save
            plt.figure(figsize=(6, 6))
            plt.imshow(refined_mask, cmap='gray')
            plt.title(f'Refined Mask: Cluster {target_cluster} with 0 Branchpoints')
            plt.axis('off')
            refined_mask_path = os.path.join(output_dir, f'{name}_refined_mask.png')
            plt.savefig(refined_mask_path)
            plt.close()
            print(f"Saved clustering viz: {cluster_viz_path} and refined mask: {refined_mask_path}")