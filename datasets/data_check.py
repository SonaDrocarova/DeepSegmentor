import os, glob, cv2, textwrap

# ↩︎ 1) point this to the folder that contains the images you THINK you’re loading
folder = r"C:\Users\sonic\OneDrive\Documents\DeepSegmentor_new\datasets\new\test_lab"

print("\n--- What files are actually there? ---")
for ext in (".png", ".jpg", ".jpeg"):
    files = glob.glob(os.path.join(folder, f"*{ext}"))
    print(f"{ext:6} {len(files)} files")

# ↩︎ 2) take one name that OpenCV just complained about
stem = "BS25-035_1"        # <== change to any of the stems in the warning
for ext in (".png", ".jpg", ".jpeg"):
    p = os.path.join(folder, stem + ext)
    print(f"\nChecking {p}")
    print("  exists:", os.path.exists(p))
    if os.path.exists(p):
        try:
            img = cv2.imread(p)
            print("  OpenCV read OK:", img is not None, "" if img is None else f"shape={img.shape}")
        except Exception as e:
            print("  OpenCV raised error:", e)
