#!/usr/bin/env python3

import random
import numpy as np
import math
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
import xgboost as xgb
import os
from data_analysis import rounded_predictions
from hk import hk_plotting, hk_plotting_and_grain_boundery_elimination
import matplotlib.pyplot as plt
from neighborhood_extraction import extract_2D_neighborhood_with_error, extract_2D_full_data_with_error_range
from prediction import apply_smoothing_2D


def xbg_predict_full_image_serial(model, arr, windowSize):
    rows, cols = arr.shape

    # Generate all data windows for prediction
    #data = generate_data(arr, windowSize)

    for i in range(rows):
        for j in range(cols):
            data = extract_2D_neighborhood_with_error(i, j, arr, windowSize,0)
            predictions = model.get_booster().inplace_predict([data])
            arr[i, j] = np.round(predictions[0])

    return arr

def xbgpredict_full_image(model, arr, windowSize):
    rows, cols = arr.shape
    new_arr = arr.copy()

    # Generate all data windows for prediction
    data = extract_2D_full_data_with_error_range(arr, windowSize, 0)
    predictions = model.predict(data)
    k = 0
    for i in range(rows):
        for j in range(cols):
            new_arr[i, j] = predictions[k] 
            k += 1

    return new_arr

def make_grid_binary(arr):
    side_length = int(math.sqrt(arr.size))
    pix = np.zeros((side_length, side_length), dtype=int)

    for i in range(side_length):
        for j in range(side_length):
            pix[i, j] = int(arr[i, j] / 255)

    return pix

def save_fig(arr, path):
    
    return 0

# Specify the number of iterations
iterations = 20
hk = True
max_error = 15
ws = [13]




for windowSize in ws:
    print(f"Processing window size {windowSize}, error {max_error}")
    model = xgb.XGBClassifier()
    model.load_model('model_xgb.json')

    # create subfolder per model
    #output_dir = f'./dat/smoothing_models/equiaxed/equiaxed_ws{windowSize}_error{max_error}/raw_output'
    output_dir = f'./dat/testxbg/output'
    ref = 'dat/0_original_structures/addalloy_512.png'
    smoothing_input_name = 'dat/0_original_structures/addalloy_binary_recon_512.npy'
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    #smoothing_input_name = 'dat/0_original_structures/3D/last_frame_128_z0.png'
    
    if smoothing_input_name.endswith('.npy'):
        smoothing_input = np.load(smoothing_input_name).astype(int)
    else:
        smoothing_input = make_grid_binary(np.array(Image.open(smoothing_input_name)))
    
    print(f"Shape of smoothing_input {smoothing_input.shape}")
    
    output_path = os.path.join(output_dir, '0.png')
    Image.fromarray((smoothing_input * 255).astype(np.uint8)).save(output_path)
    
    for i in range(iterations):
        print(f"Iteration {i + 1} of {iterations}")

        #output_image = predict_full_image(model, smoothing_input, windowSize)
        output_image = apply_smoothing_2D(model, smoothing_input, windowSize)
        
        smoothing_input = output_image

        # Save the output image
        output_path = os.path.join(output_dir, f'{i + 1}.png')
        Image.fromarray((output_image * 255).astype(np.uint8)).save(output_path)
        #hk_plotting(output_path, windowSize, ref, os.path.join(output_dir, f'{i + 1}hk.png'))
        #hk_plotting_and_grain_boundery_elimination(output_path, windowSize, ref, os.path.join(output_dir, f'{i + 1}hkNoGB.png'))
        print(f"Iteration {i + 1} complete.")

    print("Processing complete.")
    