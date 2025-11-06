# split_sperm_parts.py
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from skimage.measure import label
from collections import deque
# ADD near the top imports (no need to duplicate if already present)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_intensity_profile_plot(smoothed, head_idx, tail_idx, out_path):
    """
    Save the smoothed intensity profile with vertical markers at the detected
    head/mid (head_idx) and mid/tail (tail_idx) split points (if valid).
    """
    if smoothed is None or len(smoothed) == 0:
        return
    xs = np.arange(len(smoothed))
    plt.figure()
    plt.plot(xs, smoothed, linewidth=2)
    if head_idx is not None and 0 <= head_idx < len(smoothed):
        plt.axvline(head_idx, linestyle="--", linewidth=1.5, label="head/mid")
    if tail_idx is not None and 0 <= tail_idx < len(smoothed):
        plt.axvline(tail_idx, linestyle="--", linewidth=1.5, label="mid/tail")
    if ((head_idx is not None and 0 <= head_idx < len(smoothed)) or
        (tail_idx is not None and 0 <= tail_idx < len(smoothed))):
        plt.legend(loc="best")
    plt.xlabel("Path index")
    plt.ylabel("Smoothed intensity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()




def _find_endpoints(skel):
    """Return list of (y, x) endpoints in a binary skeleton (8-neighborhood)."""
    ys, xs = np.where(skel > 0)
    endpoints = []
    H, W = skel.shape
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y-1), min(H-1, y+1)
        x0, x1 = max(0, x-1), min(W-1, x+1)
        nb = np.sum(skel[y0:y1+1, x0:x1+1]) - skel[y, x]
        if nb == 255:  # exactly one neighbor (since skeleton is 0/255)
            endpoints.append((y, x))
    return endpoints

def _trace_path(skel, start, end):
    """
    Trace a single path between start and end through the skeleton (0/255 image).
    Uses BFS predecessors to recover the shortest 8-connected path.
    Returns list of (y, x) points in order from start to end.
    """
    H, W = skel.shape
    start, end = tuple(start), tuple(end)
    q = deque([start])
    seen = {start: None}
    nbrs8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while q:
        y, x = q.popleft()
        if (y, x) == end:
            # reconstruct path
            path = []
            cur = end
            while cur is not None:
                path.append(cur)
                cur = seen[cur]
            path.reverse()
            return path
        for dy, dx in nbrs8:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] > 0:
                nxt = (ny, nx)
                if nxt not in seen:
                    seen[nxt] = (y, x)
                    q.append(nxt)
    return []  # no path

def _find_sperm_split_points(smoothed_signal):
    """
    Adapted from your head_middle_tail.py::find_sperm_split_points.
    Returns (head_idx, tail_idx, reversed_flag) or (-1, -1, False) on failure.
    """
    if len(smoothed_signal) == 0:
        return -1, -1, False

    length = len(smoothed_signal)
    reverse = False

    # If the first value is lower than the first peak, reverse
    if np.argmin(smoothed_signal) >= length // 2:
        smoothed_signal = smoothed_signal[::-1]
        reverse = True

    first_peak = int(np.argmax(smoothed_signal[:length // 2]))
    second_peak = int(np.argmax(smoothed_signal[length // 2:]) + length // 2)

    first_derivative = np.diff(smoothed_signal[:length // 2])
    if len(first_derivative) == 0:
        return -1, -1, reverse
    if len(first_derivative[first_peak:]) == 0:
        return -1, -1, reverse
    head = int(first_peak + np.argmax(-first_derivative[first_peak:]) + 10)

    second_derivative = np.diff(smoothed_signal, n=2)
    tail = int(
        second_peak - (length // 3)
        + np.argmax(second_derivative[second_peak - (length // 3):second_peak]) + 10
    )
    if tail > length - 1:
        tail = int(
            second_peak - (length // 3)
            + np.argsort(second_derivative[second_peak - (length // 3):second_peak - 30])[-2] + 10
        )

    if reverse:
        head = length - 1 - head
        tail = length - 1 - tail

    if head <= 0 or tail <= 0:
        return -1, -1, reverse
    return head, tail, reverse

def split_head_mid_tail(original_gray, skeleton_mask, return_profiles: bool = False):
    """
    Parameters:
        original_gray (H,W) uint8
        skeleton_mask (H,W) uint8 in {0,255}
        return_profiles (bool): if True, also return a list of dicts with
            {'smoothed': np.ndarray, 'head': int or -1, 'tail': int or -1}
    Returns:
        part_mask (H,W) uint8 in {0,1,2,3}  (0=bg, 1=head, 2=midpiece, 3=tail)
        (and optionally) profiles: List[Dict]
    """
    from skimage.measure import label

    labeled = label(skeleton_mask > 0, connectivity=2)
    part_mask = np.zeros_like(skeleton_mask, dtype=np.uint8)
    profiles = [] if return_profiles else None

    max_lab = labeled.max()
    if max_lab == 0:
        return (part_mask, profiles) if return_profiles else part_mask

    for lab in range(1, max_lab + 1):
        skel = (labeled == lab).astype(np.uint8) * 255
        endpoints = _find_endpoints(skel)
        if len(endpoints) != 2:
            # still record an empty profile for bookkeeping if requested
            if return_profiles:
                profiles.append({'smoothed': np.array([]), 'head': -1, 'tail': -1})
            continue

        start, end = endpoints[0], endpoints[1]
        path = _trace_path(skel, start, end)
        if len(path) < 5:
            if return_profiles:
                profiles.append({'smoothed': np.array([]), 'head': -1, 'tail': -1})
            continue

        intens = np.array([original_gray[y, x] for (y, x) in path], dtype=np.float32)

        # Smooth + spline
        smooth = gaussian_filter1d(intens, sigma=15)
        x = np.arange(len(smooth))
        spline = UnivariateSpline(x, smooth, s=len(smooth))
        smoothed = spline(x)

        head, tail, rev = _find_sperm_split_points(smoothed)

        # record profile even if split points failed
        if return_profiles:
            profiles.append({'smoothed': smoothed, 'head': head if head != -1 else -1, 'tail': tail if tail != -1 else -1})

        if head == -1 or tail == -1:
            continue

        # Assign labels along the path: 1=head, 2=midpiece, 3=tail
        idxs = [0, head, tail, len(path) - 1]
        if not rev:
            idxs.sort()
            seg_val = 0
            cursor = 0
            for i in range(0, len(path)):
                if i == idxs[cursor]:
                    seg_val += 1
                    cursor = min(cursor + 1, 3)
                y, x = path[i]
                part_mask[y, x] = seg_val
        else:
            idxs.sort(reverse=True)
            seg_val = 0
            cursor = 0
            for i in range(len(path) - 1, -1, -1):
                if i == idxs[cursor]:
                    seg_val += 1
                    cursor = min(cursor + 1, 3)
                y, x = path[i]
                part_mask[y, x] = seg_val

    return (part_mask, profiles) if return_profiles else part_mask


def colorize_parts(part_mask):
    """
    Returns a BGR color image visualizing the parts: head=red, mid=green, tail=blue.
    with thickened lines for better visibility.
    """
    h, w = part_mask.shape
    # Create a thickened version using dilation
    kernel = np.ones((3,3), np.uint8)
    thick_mask = cv2.dilate(part_mask, kernel, iterations=1)
    
    color = np.zeros((h, w, 3), dtype=np.uint8)
    # Head (1): red channel
    color[thick_mask == 1, 2] = 255
    # Midpiece (2): green channel  
    color[thick_mask == 2, 1] = 255
    # Tail (3): blue channel
    color[thick_mask == 3, 0] = 255
    return color
    # h, w = part_mask.shape
    # color = np.zeros((h, w, 3), dtype=np.uint8)
    # # Head (1): red channel
    # color[part_mask == 1, 2] = 255
    # # Midpiece (2): green channel
    # color[part_mask == 2, 1] = 255
    # # Tail (3): blue channel
    # color[part_mask == 3, 0] = 255
    return color


# --- ADD: convert skeleton mask -> ordered polylines (list of [(x,y), ...]) ---
def skeleton_to_polylines(skeleton_mask, min_points=5):
    """
    Parameters:
        skeleton_mask (H,W) uint8 in {0,255}
    Returns:
        List[List[Tuple[int,int]]] : each polyline is a list of (x,y) points
                                     ordered along the skeleton path.
    """
    from skimage.measure import label
    import numpy as np

    labeled = label(skeleton_mask > 0, connectivity=2)
    polylines = []
    for lab in range(1, labeled.max() + 1):
        skel = (labeled == lab).astype(np.uint8) * 255
        endpoints = _find_endpoints(skel)
        if len(endpoints) != 2:
            # skip branched/ambiguous components for CVAT polylines
            continue
        path = _trace_path(skel, endpoints[0], endpoints[1])
        if len(path) >= min_points:
            # convert (y,x) -> (x,y)
            poly = [(int(x), int(y)) for (y, x) in path]
            polylines.append(poly)
    return polylines
