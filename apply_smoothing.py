

import argparse
import random as random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import os.path
from prediction import apply_smoothing_2D, apply_smoothing_3D
from conversions_and_misc import make_grid_binary
#from hk import hk_plotting, hk_plotting_and_grain_boundery_elimination
from tensorflow.python.keras.saving import hdf5_format
import h5py
import json
#import train_xgb
import time
import xgboost as xgb
#log file: dokumentiert nochmal alle parameter

# collect Timestamp
start_time = time.time()


# collecting the arguments from the command line
parser = argparse.ArgumentParser("apply_smoothing")
parser.add_argument("iterations", help="defines the number of sequential smoothings applied to the input", type=int)
parser.add_argument("smoothing_input_path", help="must be a path to a 2 or 3 dimensional binary matrix .npy or .png file (only 2D for obvious reasons)", type=str)
parser.add_argument("model_path", help="must be a path to a .h5 file containing the model", type=str)
parser.add_argument("filename", help="defines the name of the output subfolder", type=str)
args = parser.parse_args()
print('hello')
print(args.model_path)


#TODO: implement print statements for the file attributes
# Load the model and extract the window size and error range from file attributes

if args.model_path.endswith('.json'):
    print(('works'))
    with open(args.model_path, 'r') as f:
        model_data = json.load(f)
        window_size = model_data['window_size']
        error_range = model_data['error_range']
        train_picture = model_data['train_arr']
        model = xgb.XGBClassifier()
        print(f"Window size: {window_size}")
        print(f"Error range: {error_range}")
        print(f"Training input: {train_picture}")
        model.load_model('model_xgb.json')
else:
    try:
        with tf.device('/CPU:0'):
            with h5py.File(args.model_path, 'r') as f:
                window_size = f.attrs['window_size'] # this is the window size used in training
                error_range = f.attrs['error_range'] # this is the error range used in training
                train_picture = f.attrs['training_input'] # this is the training input image
                print(f"Window size: {window_size}")
                print(f"Error range: {error_range}")
                print(f"Training input: {train_picture}")
                model = tf.keras.models.load_model(f)
    except:
        "no metadata in file available"

    # Check if smoothing_input is a .png or .npy file
    if args.smoothing_input_path.endswith('.png'):
        input_im = Image.open(args.smoothing_input_path)
        # Convert the input image to an array
        input_arr = np.array(list(input_im.getdata(0)))
        # Make the input image binary (255-> 1, 0 -> 0)
        input_picture = make_grid_binary(input_arr)
    if args.smoothing_input_path.endswith('.npy'):
        # Load npy file into an array
        input_picture = np.load(args.smoothing_input)
    else:
        print('unsupported file type')
        exit(1)

    # Check if the input is 2D or 3D
    resolution = len(input_picture)
    if input_picture.ndim == 3:
        input_is_3d = True
        dim=3
    else:
        input_is_3d = False
        dim=2



    # inp_ = 'dat/0_original_structures/addalloy_binary_recon_512.png'
    # ref = 'dat/0_original_structures/addalloy_binary_recon_512.png'
    #inp_ = 'dat/0_original_structures/3D/last_frame_128_z0.png'
    # ref = 'dat/0_original_structures/3D/last_frame_128_z0.png'
    # inp_ = 'dat/0_original_structures/3D/addalloy_128.png'
    #ref =    np.load('dat/0_original_structures/3D/addalloy_128.png')
    # inp_ = './dat/0_original_structures/addalloy_506.png'
    # ref = './dat/0_original_structures/addalloy_506.png'
    # inp_ = 'dat/0_original_structures/random_matrix.png'
    # ref = 'dat/0_original_structures/random_matrix.png'
    # inp_ = 'dat/0_original_structures/random_matrix_506.png'
    # ref = 'dat/0_original_structures/random_matrix_506.png'

    #3D
    #smoothing_input_ = 'dat/0_original_structures/3D/last_frame.npy'
    #smoothing_input_ = reconstructed_volume_128 = np.random.choice([0, 1], size=(resolution, resolution, resolution))

        # Check if target folder already exists and create if not
    output_dir = f'./output/{start_time}{args.filename}_{resolution}_window_size{window_size}_{dim}D'
    for sub_dir in ['npy', 'png']:
        dir_path = os.path.join(output_dir, sub_dir)
        os.makedirs(dir_path, exist_ok=True)

        # Run the model for the specified number of iterations
    for i in range(args.iterations):
        print(f"Iteration {i + 1} of {args.iterations}")
        if input_is_3d:
            output_image = apply_smoothing_3D(model, input_picture, window_size)
        else:
            output_image = apply_smoothing_2D(model, input_picture, window_size)

        # Save the output image
        output_image_path = f'./output/{start_time}{args.filename}_{resolution}_window_size{window_size}_{dim}D/png/{i + 1}.png'
        Image.fromarray((output_image*255).astype(np.uint8)).save(output_image_path)

        # Save the output image as a .npy file
        output_npy_path = f'./output/{start_time}{args.filename}_{resolution}_window_size{window_size}_{dim}D/npy/{i + 1}.npy'
        np.save(output_npy_path, output_image)

        #if hk:
        #    hk_output_image_path_ = f'./dat/smoothing_models/{filename}_{resolution}/range/{filename}_{resolution}_ws{window_size}_error{error_range}/1.{i + 1}_hk_without_grain_boundaries_{filename}_{resolution}_ws{window_size}_error{error_range}.png'
        #    hk_output_image_path = f'./dat/smoothing_models/{filename}_{resolution}/range/{filename}_{resolution}_ws{window_size}_error{error_range}/2.{i + 1}_hk_{filename}_{resolution}_ws{window_size}_error{error_range}.png'
        #    # hk_plotting_and_grain_boundery_elimination(output_image_path, windowSize, image_for_comp, hk_output_image_path_)
        #    hk_plotting(output_image_path, window_size, ref, hk_output_image_path)
        # Set the output of this iteration as the input for the next
        smoothing_input = output_image_path
        print(f"Iteration {i + 1} complete.")

    print("Processing complete.")

