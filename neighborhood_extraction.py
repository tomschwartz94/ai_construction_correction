# Description: This file contains the functions to extract 2D and 3D neighborhoods from a given array, with a specified error rate.

import numpy as np



def extract_2D_neighborhood_with_error(i, j, arr, windowSize, error):
    """
    Extract a 2D neighborhood from a given array, with a specified error rate.

    Parameters:
    i, j (int): The coordinates of the center of the neighborhood.
    arr (numpy.ndarray): The array from which to extract the neighborhood.
    windowSize (int): The size of the window to use for extraction.
    error (float): The error rate to use when extracting.

    Returns:
    numpy.ndarray: The extracted neighborhood.
    """
    inputs = np.empty(windowSize * windowSize-1, dtype=int)
    rows, cols = arr.shape
    central_pixel_was_ignored = False

    for x in range(i - windowSize // 2, i + windowSize // 2 + 1):
        for y in range(j - windowSize // 2, j + windowSize // 2 + 1):
            # Wrap around for both rows and columns
            x_periodic = x % rows
            y_periodic = y % cols

            if central_pixel_was_ignored:
                index = (x - (i - windowSize // 2)) * windowSize + (y - (j - windowSize // 2))-1
            else:
                index = (x - (i - windowSize // 2)) * windowSize + (y - (j - windowSize // 2))

            if x == i and y == j:
                central_pixel_was_ignored = True
                continue
            else:
                inputs[index] = arr[x_periodic][y_periodic]

    if error > 0:
        for index in range(len(inputs)):
            if np.random.uniform(0, 1) < error:
                inputs[index] = np.random.randint(0, 2)


    return inputs


#append richtige methode performancemäßig?
def extract_2D_full_data_with_error_range(arr, windowSize, error_range,multiplicator, rotation=False):
    inputs = list()
    outputs = list()


    for error in error_range:
        print(f'Error: {error}')
        for n in range(multiplicator):
            for i in range(windowSize // 2, len(arr) - windowSize // 2):
                for j in range(windowSize // 2, len(arr[0]) - windowSize // 2):
                    inputArr = extract_2D_neighborhood_with_error(i, j, arr, windowSize, error/100)
                    inputs.append(inputArr)
                    outputs.append(arr[i, j])
            if error==0:
                break

    if rotation:
        for _ in range(4):
            for error in error_range:
                print(f'Error: {error}')
                for i in range(windowSize // 2, len(arr) - windowSize // 2):
                    for j in range(windowSize // 2, len(arr[0]) - windowSize // 2):
                        inputArr = extract_2D_neighborhood_with_error(i, j, arr, windowSize, error / 100)
                        inputs.append(inputArr)
                        outputs.append(arr[i][j])
            arr = list(zip(*arr[::-1]))


    return np.asarray(inputs), np.asarray(outputs)


def extract_3D_neighborhood(i, j, k, arr, windowSize, plane):
    inputs = np.empty(windowSize * windowSize-1, dtype=int)

    central_pixel_was_ignored = False

    height, length, width = arr.shape
    z = k
    for x in range(i - windowSize // 2, i + windowSize // 2 + 1):
        for y in range(j - windowSize // 2, j + windowSize // 2 + 1):
            # Wrap around for both rows and columns
            i_periodic = x % height
            j_periodic = y % length

            if central_pixel_was_ignored:
                index = (x - (i - windowSize // 2)) * windowSize + (y - (j - windowSize // 2))-1
            else:
                index = (x - (i - windowSize // 2)) * windowSize + (y - (j - windowSize // 2))

            if x == i and y == j:
                central_pixel_was_ignored = True
                continue

            else:
                if plane == 0:
                    inputs[index] = arr[i_periodic][j_periodic][k]

                if plane == 1:
                    inputs[index] = arr[j_periodic][k][i_periodic]

                if plane == 2:
                    inputs[index] = arr[k][j_periodic][i_periodic]

    return inputs


