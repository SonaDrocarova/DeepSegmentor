import argparse
import json
import math
import os
from datetime import datetime

import cv2
import numpy as np


def euclid(p1, p2) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


class ZoomPanClick:
    """
    OpenCV viewer with:
      - mouse wheel zoom (centered at cursor)
      - pan with middle-button drag OR space+left drag
      - left click to pick points (in image coordinates)
    """

    def __init__(self, image_bgr, window="Measure scale bar (zoom/pan)"):
        self.img0 = image_bgr
        self.H, self.W = image_bgr.shape[:2]
        self.window = window

        # View transform: screen = (img - offset) / zoom  (implemented via crop+resize)
        self.zoom = 1.0  # >1 means zoom in
        self.cx = self.W / 2.0  # view center in image coords
        self.cy = self.H / 2.0

        # Interaction state
        self.points = []  # list of (x_img, y_img)
        self.done = False

        self._dragging = False
        self._drag_start = (0, 0)
        self._center_start = (self.cx, self.cy)
        self._space_down = False

        # Last rendered view bookkeeping
        self._view = None
        self._view_rect = None  # (x0,y0,x1,y1) in image coords
        self._view_size = None  # (vw,vh) in screen pixels

    def reset_points(self):
        self.points = []
        self.done = False

    def reset_view(self):
        self.zoom = 1.0
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0

    def clamp_center(self, vw_img, vh_img):
        half_w = vw_img / 2.0
        half_h = vh_img / 2.0
        self.cx = float(np.clip(self.cx, half_w, self.W - half_w))
        self.cy = float(np.clip(self.cy, half_h, self.H - half_h))

    def render(self, win_w=1400, win_h=900):
        # compute view size in image coords based on zoom
        vw_img = self.W / self.zoom
        vh_img = self.H / self.zoom
        vw_img = min(vw_img, self.W)
        vh_img = min(vh_img, self.H)

        self.clamp_center(vw_img, vh_img)

        x0 = int(round(self.cx - vw_img / 2.0))
        y0 = int(round(self.cy - vh_img / 2.0))
        x1 = int(round(self.cx + vw_img / 2.0))
        y1 = int(round(self.cy + vh_img / 2.0))

        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(self.W, x1); y1 = min(self.H, y1)

        crop = self.img0[y0:y1, x0:x1].copy()

        # Draw points/line in the crop coordinate system
        def img_to_crop(pt):
            return (int(round(pt[0] - x0)), int(round(pt[1] - y0)))

        for p in self.points:
            pc = img_to_crop(p)
            cv2.circle(crop, pc, 5, (0, 255, 0), -1)

        if len(self.points) == 2:
            p0c = img_to_crop(self.points[0])
            p1c = img_to_crop(self.points[1])
            cv2.line(crop, p0c, p1c, (0, 255, 0), 2)
            dist = euclid(self.points[0], self.points[1])
            txt = f"{dist:.2f}px"
            tx = min(p0c[0], p1c[0])
            ty = max(20, min(p0c[1], p1c[1]) - 10)
            cv2.putText(crop, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # HUD
        hud = f"zoom={self.zoom:.2f} | points={len(self.points)}/2 | wheel=zoom | MMB drag or Space+LMB=pan | r=reset points | 0=reset view | s=save"
        cv2.putText(crop, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Resize crop to window display size while keeping aspect
        ch, cw = crop.shape[:2]
        scale = min(win_w / cw, win_h / ch)
        disp_w = max(1, int(cw * scale))
        disp_h = max(1, int(ch * scale))
        disp = cv2.resize(crop, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

        self._view = disp
        self._view_rect = (x0, y0, x1, y1)
        self._view_size = (disp_w, disp_h)
        return disp

    def screen_to_image(self, sx, sy):
        # Map screen coords in the displayed view back into image coords
        if self._view_rect is None or self._view_size is None:
            return None
        x0, y0, x1, y1 = self._view_rect
        disp_w, disp_h = self._view_size

        # If click is outside the displayed image (possible if window bigger), ignore
        if sx < 0 or sy < 0 or sx >= disp_w or sy >= disp_h:
            return None

        vw = (x1 - x0)
        vh = (y1 - y0)
        ix = x0 + (sx / disp_w) * vw
        iy = y0 + (sy / disp_h) * vh
        return (float(ix), float(iy))

    def zoom_at(self, factor, sx, sy):
        """Zoom around cursor position (in screen coords)."""
        before = self.screen_to_image(sx, sy)
        if before is None:
            # zoom around center
            before = (self.cx, self.cy)

        # Update zoom with limits
        new_zoom = float(np.clip(self.zoom * factor, 1.0, 50.0))
        if abs(new_zoom - self.zoom) < 1e-6:
            return

        self.zoom = new_zoom

        # After zoom, adjust center so the same image point stays under cursor
        after = before  # desired
        # compute current view size in image coords
        vw_img = self.W / self.zoom
        vh_img = self.H / self.zoom
        vw_img = min(vw_img, self.W)
        vh_img = min(vh_img, self.H)

        # determine where the cursor lies within the view as fractions
        x0, y0, x1, y1 = self._view_rect if self._view_rect else (0, 0, self.W, self.H)
        disp_w, disp_h = self._view_size if self._view_size else (self.W, self.H)
        fx = sx / max(1, disp_w)
        fy = sy / max(1, disp_h)

        # Set center so that the desired point lands at that fractional position
        self.cx = after[0] - (fx - 0.5) * vw_img
        self.cy = after[1] - (fy - 0.5) * vh_img
        self.clamp_center(vw_img, vh_img)

    def on_mouse(self, event, x, y, flags, param):
        # Mouse wheel zoom
        if event == cv2.EVENT_MOUSEWHEEL:
            # flags > 0: forward (away) typically zoom in
            if flags > 0:
                self.zoom_at(1.25, x, y)
            else:
                self.zoom_at(1 / 1.25, x, y)
            return

        # Start pan: middle button OR space+left
        if event == cv2.EVENT_MBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and self._space_down):
            self._dragging = True
            self._drag_start = (x, y)
            self._center_start = (self.cx, self.cy)
            return

        if event == cv2.EVENT_MOUSEMOVE and self._dragging:
            dx = x - self._drag_start[0]
            dy = y - self._drag_start[1]

            # Convert screen delta to image delta based on current view size
            if self._view_rect and self._view_size:
                x0, y0, x1, y1 = self._view_rect
                disp_w, disp_h = self._view_size
                vw = (x1 - x0)
                vh = (y1 - y0)
                # dragging right should move view right -> center moves left in image coords
                self.cx = self._center_start[0] - (dx / max(1, disp_w)) * vw
                self.cy = self._center_start[1] - (dy / max(1, disp_h)) * vh

                vw_img = self.W / self.zoom
                vh_img = self.H / self.zoom
                vw_img = min(vw_img, self.W)
                vh_img = min(vh_img, self.H)
                self.clamp_center(vw_img, vh_img)
            return

        # End pan
        if event == cv2.EVENT_MBUTTONUP or event == cv2.EVENT_LBUTTONUP:
            self._dragging = False

        # Left click to place points (only if not panning)
        if event == cv2.EVENT_LBUTTONDOWN and not self._space_down:
            if self.done:
                return

            pt = self.screen_to_image(x, y)
            if pt is None:
                return

            # snap to int pixel coords
            pt_i = (int(round(pt[0])), int(round(pt[1])))
            self.points.append(pt_i)

            if len(self.points) == 2:
                self.done = True
def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="Click two points on the scale bar with zoom/pan support.")
    ap.add_argument("--image", required=True, help="Path to an image that contains the scale bar.")
    ap.add_argument("--bar_um", type=float, default=20.0, help="Scale bar length in micrometers (default: 20).")
    ap.add_argument("--out", default=None, help="Output JSON path. Default: <image_dir>/scale_calibration.json")
    ap.add_argument("--key", default=None, help="Optional key in JSON (default: image basename).")
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    if args.out is not None:
        out_path = args.out
    else:
        out_path = os.path.join(script_dir(), "scale_calibration.json")

    key = args.key if args.key is not None else os.path.basename(args.image)

    viewer = ZoomPanClick(img)
    cv2.namedWindow(viewer.window, cv2.WINDOW_NORMAL)

    # Optional: set a reasonable initial window size (user can still resize)
    cv2.resizeWindow(viewer.window, 1400, 900)
    cv2.setMouseCallback(viewer.window, viewer.on_mouse)

    print("Controls:")
    print("- Mouse wheel: zoom in/out (at cursor)")
    print("- Pan: middle-mouse drag OR hold Space and left-drag")
    print("- Left click: pick 2 points (start/end of scale bar)")
    print("- r: reset points   |  0: reset view")
    print("- s: save calibration JSON   |  q / ESC: quit without saving")

    while True:
        disp = viewer.render(win_w=1600, win_h=1000)
        cv2.imshow(viewer.window, disp)

        k = cv2.waitKey(20) & 0xFF

        if k in (27, ord("q")):
            cv2.destroyAllWindows()
            print("Quit without saving.")
            return

        if k == ord(" "):
            # Some OS/window managers don't reliably give keyup; we handle "sticky"
            # via toggling on keypress. But better: treat space as "held" when pressed
            # by using getWindowProperty isn't possible. We'll implement as toggle:
            viewer._space_down = not viewer._space_down

        if k == ord("r"):
            viewer.reset_points()

        if k == ord("0"):
            viewer.reset_view()

        if k == ord("s"):
            if len(viewer.points) != 2:
                print("You need to click exactly 2 points before saving.")
                continue

            bar_pixels = euclid(viewer.points[0], viewer.points[1])
            um_per_px = args.bar_um / bar_pixels

            data = {}
            if os.path.exists(out_path):
                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}

            data[key] = {
                "image_path": os.path.abspath(args.image),
                "bar_um": float(args.bar_um),
                "bar_pixels": float(bar_pixels),
                "um_per_px": float(um_per_px),
                "points_xy": [list(viewer.points[0]), list(viewer.points[1])],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            cv2.destroyAllWindows()
            print(f"Saved calibration to: {out_path}")
            print(f"bar_pixels = {bar_pixels:.3f}")
            print(f"um_per_px   = {um_per_px:.6f}")
            return


if __name__ == "__main__":
    main()
