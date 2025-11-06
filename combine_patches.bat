@echo off
REM Activate your conda or virtual environment
CALL "C:\Users\sonic\anaconda3\condabin\conda.bat" activate deepsegmentor_new

REM --- PATCH-BASED INFERENCE WITH STITCHING ---
python combine_patches.py ^
  --model deepcrack ^
  --dataset_mode sperm ^
  --dataroot C:\Users\sonic\OneDrive\Documents\DeepSegmentor_new\datasets\new^
  --use_augment False ^
  --phase test ^
  --eval ^
  --name final_deepsperm_split ^
  --load_iter 0 ^
  --epoch fallen-sweep-16^
  --gpu_ids 0 ^
  --num_test 999999

pause
