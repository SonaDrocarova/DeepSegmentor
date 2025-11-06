import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--metric_mode', type=str, default='sem', help='[prf | sem]')
parser.add_argument('--model_name', type=str, default='deepcrack')
parser.add_argument('--results_dir', type=str, default='C:\\Users\\sonic\\OneDrive\\Documents\\DeepSegmentor_new\\results\\final_deepcrack\\test_eternal-dream-52_best')


if __name__==  "__main__":
    # Load the dataset
    metric_mode = parser.parse_args().metric_mode
    results_dir = parser.parse_args().results_dir
    model_name = parser.parse_args().model_name
    
    # df = pd.read_csv(results_dir+'\\metrics.csv')
    data = pd.read_csv(f'{results_dir}/metrics.txt', delim_whitespace=True, header=None)
    data.columns = ['Threshold', 'GlobalAccuracy', 'MeanAccuracy', 'MeanIoU']
    plt.plot(data.Threshold, data.GlobalAccuracy, label='GlobalAccuracy')
    plt.plot(data.Threshold, data.MeanAccuracy, label='MeanAccuracy')
    plt.plot(data.Threshold, data.MeanIoU, label='MeanIoU')
    plt.legend()
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.savefig(f'{results_dir}/metrics.png')