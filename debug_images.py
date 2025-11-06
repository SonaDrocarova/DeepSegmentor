import cv2
from pathlib import Path

mask_dir = Path("C:/Users/sonic/OneDrive/Documents/Spermie/split_dataset/train_lab")
bad_files = []

for file in mask_dir.glob("*.png"):
    img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
    if img is None:
        bad_files.append(file.name)

print("Unreadable masks:", bad_files)
