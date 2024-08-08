import argparse
import json
import os.path
import time
import numpy as np
import xgboost as xgb
from PIL import Image
from conversions_and_misc import generate_log_file, make_grid_binary, convert_np_to_vti
from prediction import apply_smoothing_2D, apply_smoothing_3D

#log file: dokumentiert nochmal alle parameter

# collect Timestamp
start_time = time.strftime("%Y%m%d_%H%M%S")
sequential=False
window_size = 0
error_range = 0
train_picture = 0
model=0

# collecting the arguments from the command line
parser = argparse.ArgumentParser("apply_smoothing")
parser.add_argument("iterations", help="defines the number of sequential smoothings applied to the input", type=int)
parser.add_argument("threshold", help="defines the treshold for deciding between grain and boundary. Must be between 0 and 3. treshold > 1.5 means more boundary predictions will be made ", type=float)
parser.add_argument("smoothing_input_path", help="must be a path to a 2 or 3 dimensional binary matrix .npy or .png file (only 2D for obvious reasons)", type=str)
parser.add_argument("model_path", help="must be a .json file containing the xgb model", type=str)
parser.add_argument("filename", help="defines the name of the output subfolder", type=str)
args = parser.parse_args()
print(args.smoothing_input_path)
# Load the model and extract the window size and error range from file attributes


with open(args.model_path, 'r') as f:
    model_data = json.load(f)
    window_size = model_data['window_size']
    error_range = model_data['error_range']
    train_picture = model_data['train_arr']
    multiplicator = model_data['multiplicator']
    model = xgb.XGBClassifier()
    model.load_model(args.model_path)



# Check if smoothing_input is a .png or .npy file
if args.smoothing_input_path.endswith('.png'):
    print('works')
    input_im = Image.open(args.smoothing_input_path)
    # Convert the input image to an array
    input_arr = np.array(list(input_im.getdata(0)))
    # Make the input image binary (255-> 1, 0 -> 0)
    input_picture = make_grid_binary(input_arr)
elif args.smoothing_input_path.endswith('.npy'):
    # Load npy file into an array
    input_picture = np.load(args.smoothing_input_path)
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
for sub_dir in ['npy', 'png','vtk']:
    dir_path = os.path.join(output_dir, sub_dir)
    os.makedirs(dir_path, exist_ok=True)
convert_np_to_vti(input_picture,
                      f'./output/{start_time}{args.filename}_{resolution}_window_size{window_size}_{dim}D/vtk/output_0000.vti')

model_params = {
    'window_size': window_size,
    'error_range': error_range,
    #'train_picture': train_picture,
    'multiplicator': multiplicator
}

# Call the method to generate the log file
generate_log_file(output_dir, model_params, args)

for i in range(args.iterations):
    print(f"Iteration {i + 1} of {args.iterations}")
    if input_is_3d:
        print('yes')
        output_image = apply_smoothing_3D(model, input_picture, window_size, treshold=args.threshold)
        print(output_image)
        convert_np_to_vti(output_image,f'./output/{start_time}{args.filename}_{resolution}_window_size{window_size}_{dim}D/vtk/output_{i+1:04d}.vti')
    else:
        output_image = apply_smoothing_2D(model, input_picture, window_size)
        output_image_path = f'./output/{start_time}{args.filename}_{resolution}_window_size{window_size}_{dim}D/png/{i + 1}.png'
        Image.fromarray((output_image * 255).astype(np.uint8)).save(output_image_path)

    # Save the output image as a .npy file
    output_npy_path = f'./output/{start_time}{args.filename}_{resolution}_window_size{window_size}_{dim}D/npy/{i + 1}.npy'
    np.save(output_npy_path, output_image)

    #if hk:
    #    hk_output_image_path_ = f'./dat/smoothing_models/{filename}_{resolution}/range/{filename}_{resolution}_ws{window_size}_error{error_range}/1.{i + 1}_hk_without_grain_boundaries_{filename}_{resolution}_ws{window_size}_error{error_range}.png'
    #    hk_output_image_path = f'./dat/smoothing_models/{filename}_{resolution}/range/{filename}_{resolution}_ws{window_size}_error{error_range}/2.{i + 1}_hk_{filename}_{resolution}_ws{window_size}_error{error_range}.png'
    #    # hk_plotting_and_grain_boundery_elimination(output_image_path, windowSize, image_for_comp, hk_output_image_path_)
    #    hk_plotting(output_image_path, window_size, ref, hk_output_image_path)
    # Set the output of this iteration as the input for the next
    smoothing_input = output_image
    print(f"Iteration {i + 1} complete.")

print("Processing complete.")

