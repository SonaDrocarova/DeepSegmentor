@echo off
REM Activate your conda or virtual environment
CALL "C:\Users\sonic\anaconda3\condabin\conda.bat" activate deepsegmentor_new


REM --- TESTING ---
python test.py ^
  --model deepcrack ^
  --dataset_mode sperm ^
  --use_augment False^
  --phase test ^
  --eval ^
  --name final_deepsperm_split ^
  --load_iter 0 ^
  --epoch playful-sweep-24 ^
  --gpu_ids 0^
  --num_test 99999

pause