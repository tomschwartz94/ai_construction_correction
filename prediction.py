import time

import numpy as np
import xgboost as xgb
from joblib import Parallel, delayed

from neighborhood_extraction import extract_3D_neighborhood, extract_2D_neighborhood_with_error, extract_3D_neighborhoods

# todo: re

######################################################################################################################################################################
# __  __          _     _                   _            __                    ___    _____                                        _     _       _
# |  \/  |        | |   | |                 | |          / _|                  |__ \  |  __ \                                      | |   | |     (_)
# | \  / |   ___  | |_  | |__     ___     __| |  ___    | |_    ___    _ __       ) | | |  | |    ___   _ __ ___     ___     ___   | |_  | |__    _   _ __     __ _
# | |\/| |  / _ \ | __| | '_ \   / _ \   / _` | / __|   |  _|  / _ \  | '__|     / /  | |  | |   / __| | '_ ` _ \   / _ \   / _ \  | __| | '_ \  | | | '_ \   / _` |
# | |  | | |  __/ | |_  | | | | | (_) | | (_| | \__ \   | |   | (_) | | |       / /_  | |__| |   \__ \ | | | | | | | (_) | | (_) | | |_  | | | | | | | | | | | (_| |
# |_|  |_|  \___|  \__| |_| |_|  \___/   \__,_| |___/   |_|    \___/  |_|      |____| |_____/    |___/ |_| |_| |_|  \___/   \___/   \__| |_| |_| |_| |_| |_|  \__, |
#                                                                                                                                                             __/ |
#                                                                                                                                                            |___/
######################################################################################################################################################################
def apply_smoothing_2D(model, arr, windowSize):
    """
        Apply a smoothing operation to a 2D array using a given model.

        Parameters:
        model (callable): The model to use for smoothing.
        arr (numpy.ndarray): The 2D array to smooth.
        windowSize (int): The size of the window to use for smoothing.

        Returns:
        numpy.ndarray: The smoothed 2D array.
        """

    for i in range(len(arr)):
        for j in range(len(arr[0])):

            input = extract_2D_neighborhood_with_error(i, j, arr, windowSize, 0)

            predictions = model.predict([input])
            arr[i, j] = predictions[0]

    return arr



#########################################################################################################################################################################
# __  __          _     _                   _            __                    ____    _____                                        _     _       _
# |  \/  |        | |   | |                 | |          / _|                  |___ \  |  __ \                                      | |   | |     (_)
# | \  / |   ___  | |_  | |__     ___     __| |  ___    | |_    ___    _ __      __) | | |  | |    ___   _ __ ___     ___     ___   | |_  | |__    _   _ __     __ _
# | |\/| |  / _ \ | __| | '_ \   / _ \   / _` | / __|   |  _|  / _ \  | '__|    |__ <  | |  | |   / __| | '_ ` _ \   / _ \   / _ \  | __| | '_ \  | | | '_ \   / _` |
# | |  | | |  __/ | |_  | | | | | (_) | | (_| | \__ \   | |   | (_) | | |       ___) | | |__| |   \__ \ | | | | | | | (_) | | (_) | | |_  | | | | | | | | | | | (_| |
# |_|  |_|  \___|  \__| |_| |_|  \___/   \__,_| |___/   |_|    \___/  |_|      |____/  |_____/    |___/ |_| |_| |_|  \___/   \___/   \__| |_| |_| |_| |_| |_|  \__, |
#                                                                                                                                                              __/ |
#                                                                                                                                                             |___/
######################################################################################################################################################################
def apply_smoothing_3D(model, arr, windowSize, treshold):
    """
    Apply a smoothing operation to a 3D array using a given model.

    Parameters:
    model (callable): The model to use for smoothing.
    arr (numpy.ndarray): The 3D array to smooth.
    windowSize (int): The size of the window to use for smoothing.

    Returns:
    numpy.ndarray: The smoothed 3D array.
    """

 #   if sequential:
  #      for plane in range(1):
   #         print(f'{plane}')
    #        for i in range(len(arr)):
     #           for j in range(len(arr[0])):
      #              for k in range(len(arr[0][0])):
       #                 input = extract_3D_neighborhood(i, j, k, arr, windowSize, plane)
        #                predictions = model.get_booster().inplace_predict([input])
         #               arr[i, j, k] = 0 if predictions[0] < treshold else 1

    for i in range(len(arr)):
        for j in range(len(arr[0])):
            for k in range(len(arr[0][0])):
                input_1, input_2, input_3 = extract_3D_neighborhoods(i, j, k, arr, windowSize)
                predictions = sum(
                    model.get_booster().inplace_predict([input]) for input in [input_1, input_2, input_3])
                arr[i, j, k] = 0 if predictions[0] < treshold else 1


    return arr


def apply_smoothing_3D_xgb_array_output(model, arr, windowSize, treshold):
    """
    Apply a smoothing operation to a 3D array using a given model.

    Parameters:
    model (callable): The model to use for smoothing.
    arr (numpy.ndarray): The 3D array to smooth.
    windowSize (int): The size of the window to use for smoothing.

    Returns:
    numpy.ndarray: The smoothed 3D array.
    """

    depth = len(arr)
    height = len(arr[0])
    width = len(arr[0][0])

    input_1_list= np.ndarray([len(arr)**3,windowSize**2-1])
    input_2_list = np.ndarray([len(arr)**3,windowSize**2-1])
    input_3_list = np.ndarray([len(arr)**3,windowSize**2-1])

    res_arr=np.ndarray([len(arr),len(arr),len(arr)])
    start_time = time.time()
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            for k in range(len(arr[0][0])):
                index = i * (depth * height) + j * width + k
                input_1, input_2, input_3 = extract_3D_neighborhoods(i, j, k, arr, windowSize)
                input_1_list[index] = input_1
                input_2_list[index] = input_2
                input_3_list[index] = input_3
    end_time = time.time()
    print(f"Extraction Duration: {end_time - start_time} seconds")
    start_time = time.time()
    input_1_dmatrix = xgb.DMatrix(input_1_list)
    input_2_dmatrix = xgb.DMatrix(input_2_list)
    input_3_dmatrix = xgb.DMatrix(input_3_list)
    end_time = time.time()
    print(f"Conversion to DMatrix Duration: {end_time - start_time} seconds")

    start_time = time.time()
    predictions_1 = model.get_booster().predict(input_1_dmatrix)
    predictions_2 = model.get_booster().predict(input_2_dmatrix)
    predictions_3 = model.get_booster().predict(input_3_dmatrix)
    end_time = time.time()
    print(f"Prediction Duration: {end_time - start_time} seconds")
    start_time = time.time()


    for i in range(len(arr)):
        for j in range(len(arr[0])):
            for k in range(len(arr[0][0])):
                index = i * (depth * height) + j * width + k

                prediction = predictions_1[index] + predictions_2[index] + predictions_3[index]

                try:
                    res_arr[i, j, k] = 0 if prediction < treshold else 1
                except IndexError:
                    print(f"IndexError at index: {index}")

    end_time = time.time()
    print(f"Reconstruction Duration: {end_time - start_time} seconds")
    return res_arr




def apply_smoothing_3D_parallelized(model, arr, windowSize):
    """
        Apply a smoothing operation to a 3D array using a given model, in a parallelized manner.

        Parameters:
        model (callable): The model to use for smoothing.
        arr (numpy.ndarray): The 3D array to smooth.
        windowSize (int): The size of the window to use for smoothing.

        Returns:
        numpy.ndarray: The smoothed 3D array.
        """

    for plane in range(3):
        print(f'starting plane {plane}')
        start_time = time.time()
        Parallel(n_jobs=-1, backend='threading')(
            delayed(parallelized_loop_for_prediction)(arr, i, model, plane, windowSize) for i in range(len(arr)))
        end_time = time.time()
        print(f"Duration plane {plane}: {end_time - start_time} seconds")
    return arr


def parallelized_loop_for_prediction(arr, i, model, plane, windowSize):
    """
        Apply a model to a 3D array in a parallelized manner.

        Parameters:
        arr (numpy.ndarray): The 3D array to which to apply the model.
        i (int): The index of the current slice of the array.
        model (callable): The model to apply.
        plane (int): The plane to use for extraction.
        windowSize (int): The size of the window to use for extraction.
        """

    for j in range(len(arr[0])):
        for k in range(len(arr[0][0])):
            input = extract_3D_neighborhood(i, j, k, arr, windowSize, plane)
            predictions = model([input])
            # [(0.2, 0.8)]
            if predictions[0][0] < 0.5:
                # if predictions[0][1] > 0.5:
                arr[i][j][k] = 1
            else:
                arr[i][j][k] = 0
