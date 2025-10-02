@echo off
REM Activate your conda or virtual environment
CALL "C:\Users\sonic\anaconda3\condabin\conda.bat" activate deepsegmentor_new

REM --- PATCH-BASED INFERENCE WITH STITCHING + CLUSTERING POSTPROCESS ---
python combine_patches_with_clustering.py ^
  --model deepcrack ^
  --dataset_mode sperm ^
  --dataroot C:\Users\sonic\OneDrive\Documents\DeepSegmentor_new\datasets\single_img^
  --use_augment False ^
  --phase test ^
  --eval ^
  --name final_deepsperm_split ^
  --load_iter 0 ^
  --epoch playful-sweep-24^
  --gpu_ids 0 ^
  --num_test 999
pause
