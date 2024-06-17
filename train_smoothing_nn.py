#!/usr/bin/env python3
import pickle
import time

import h5py
import numpy as np
import tensorflow as tf
from PIL import Image
import os.path

from conversions_and_misc import make_grid_binary
from neighborhood_extraction import extract_2D_neighborhood_with_error, extract_2D_full_data_with_error_range
from sklearn.model_selection import train_test_split
from keras import callbacks
#from hk import plot_and_save_losses

reconstructed_input_image_path_512 = 'dat/0_original_structures/addalloy_binary_recon_512.png'
reconstructed_input_image_path_128 = f'dat/0_original_structures/3D/last_frame_128_z0.png'
original_image_for_comp_128_path = f'dat/0_original_structures/3D/addalloy_128.png'
original_image_for_comp_506_path = './dat/0_original_structures/addalloy_506.png'
random_matrix_128_path = 'dat/0_original_structures/random_matrix.png'
random_matrix_506_path = 'dat/0_original_structures/random_matrix_506.png'


iterations = 10
hk = True
patience = 10
error_range = range(0, 20, 1)
ws_range = [13]
#für ordern und dateinamen
resolution = 128
filename = "addalloy128_bs_32_test_modeL_error_0to10"



with tf.device('/CPU:0'):

    # Extract training data from image
    train_image = original_image_for_comp_128_path
    train_im = Image.open(train_image)
    train_arr = list(train_im.getdata(0))
    #test_arr = [(x, x, x) for x in train_arr]
    train_picture = make_grid_binary(train_arr)
    #print(train_picture)



    for window_size in ws_range:

            #start_time = time.time()

            inputs, outputs = extract_2D_full_data_with_error_range(train_picture, window_size,error_range)

            number_of_inputs = np.shape(inputs)
            #print(f'shape:{number_of_inputs}')
            #inputs = np.reshape(inputs, (number_of_inputs[0], number_of_inputs[1], 1))

            train_data, test_data, train_labels, test_labels = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

            #print('opening file')
            #with open('train_test_dataset.pkl', 'rb') as file:
            #    train_data, test_data, train_labels, test_labels = pickle.load(file)
            #print('file opened')

            #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'./dat/smoothing_models/{filename}_ws{window_size}_error{error_range}/')
            [,[[]],]
            # Design the model (2 hidden layers with 24 neurons each and one output layer)
            model = tf.keras.Sequential([
                #tf.keras.layers.Flatten(),
                tf.keras.layers.Input(shape=(window_size*window_size-1,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                #tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax')

            ])


            #opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
            opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0005)

            model.compile(optimizer=opt,
                          loss='sparse_categorical_crossentropy',
                          metrics=['acc'])

            earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                                    mode="min", patience=patience,
                                                    restore_best_weights=True, verbose=1)
        # Train the model
            history = model.fit(x=train_data,
                                y=train_labels,
                                epochs=1000,
                                verbose=1,
                                batch_size=32,
                                validation_data=(test_data, test_labels),
                                callbacks=[earlystopping])

            bestEpoch = earlystopping.stopped_epoch-patience

            if not os.path.isdir(
                    f'./dat/smoothing_models/{filename}_ws{window_size}_error{error_range}/'):
                os.makedirs(
                                    f'./dat/smoothing_models/{filename}_ws{window_size}_error{error_range}/')

            # Save model
            with h5py.File(   f'./dat/smoothing_models/{filename}_ws{window_size}.h5', mode='w', track_order=True) as f:
                model.save(f, save_format='h5')
                f.attrs['window_size'] = window_size
                f.attrs['error_range'] = error_range
                f.attrs['training_input'] = train_picture

            #tf.keras.models.save_model(f)
            #model.save(f'./dat/smoothing_models/{filename}_ws{window_size}_error{error_range}/{filename}_ws{window_size}_error{error_range}.keras')


            # Plot the training and validation accuracy and loss at each epoch

            plot_and_save_losses(history, f'./dat/smoothing_models/{filename}_ws{window_size}_error{error_range}/0_train_vs_val_loss', bestEpoch)


            #end_time = time.time()
            #elapsed_time = end_time - start_time

            #print(elapsed_time)
