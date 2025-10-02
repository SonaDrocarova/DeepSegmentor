@echo off
REM Activate your conda or virtual environment
CALL "C:\Users\sonic\anaconda3\condabin\conda.bat" activate deepsegmentor_new


REM --- TRAINING ---
python train.py ^
  --model deepcrack ^
  --dataset_mode sperm ^
  --batch_size 8 ^
  --lr 0.0001 ^
  --loss_mode focal ^
  --use_augment False^
  --niter 20 ^
  --niter_decay 20 ^
  --name final_deepsperm_split ^
  --gpu_ids 0^
  --lr_policy linear

pause
