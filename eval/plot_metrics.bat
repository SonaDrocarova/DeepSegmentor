@echo off
REM Activate your conda or virtual environment
CALL "C:\Users\sonic\anaconda3\condabin\conda.bat" activate deepsegmentor_new


REM --- TRAINING ---
python plt_metrics.py ^
  --metric_mode sem ^
  --results_dir "C:\Users\sonic\OneDrive\Documents\DeepSegmentor_new\results\final_deepsperm_split\test_playful-sweep-24" 


pause