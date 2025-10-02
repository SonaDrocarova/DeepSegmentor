@echo off
REM Activate your conda or virtual environment
CALL "C:\Users\sonic\anaconda3\condabin\conda.bat" activate deepsegmentor_new


REM --- TRAINING ---
python train_sweep.py ^
  --model deepcrack ^
  --dataset_mode deepcrack ^
  --gpu_ids 0^
  --display_id 1^
  --no_html 

pause
