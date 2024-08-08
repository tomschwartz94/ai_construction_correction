#!/usr/bin/env python3
import os
import time
import numpy as np
np.bool = np.bool_
import math
import random
import vtk
from joblib import Parallel, delayed
from PIL import Image
from vtkmodules.util import numpy_support


def make_grid_RGBA(arr):
    pixels = list()
    for i in range(len(arr)):
        pixels.append(int(arr[i][0] / 255))
    pix = np.array(pixels, dtype=int)
    pix = np.reshape(pix, (int(math.sqrt(len(arr))), int(math.sqrt(len(arr)))))
    return pix

def make_grid_binary(arr):
    pixels = list()
    for i in range(len(arr)):
        pixels.append(int(arr[i]/ 255))
    pix = np.array(pixels, dtype=int)
    pix = np.reshape(pix, (int(math.sqrt(len(arr))), int(math.sqrt(len(arr)))))
    return pix

def reconstruct_image(arr):
    reconstruct_image = np.empty((len(arr), len(arr[0]), 3), dtype=np.uint8)
    for i in range(len(reconstruct_image)):
        for j in range(len(reconstruct_image[0])):
            reconstruct_image[i][j][0] = arr[i][j] * 255
            reconstruct_image[i][j][1] = arr[i][j] * 255
            reconstruct_image[i][j][2] = arr[i][j] * 255
    return reconstruct_image

def rgba_png_to_binary_numpy(file_path):
    # Open the image using Pillow
    img = Image.open(file_path).convert('RGBA')

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Extract the alpha channel
    alpha_channel = img_array[:, :, 2]

    # Threshold the alpha channel to create a binary mask (1 for values above threshold, 0 for others)
    binary_mask = (alpha_channel > 0).astype(np.uint8)

    return binary_mask

def convert_np_to_vti(volume, output_path):
    global vtk_data_array, vtk_image_data, writer
    vtk_data_array = numpy_support.numpy_to_vtk(num_array=volume.ravel(), deep=True, array_type=vtk.VTK_INT)
    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(volume.shape)
    vtk_image_data.GetPointData().SetScalars(vtk_data_array)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(
        output_path)
    writer.SetInputData(vtk_image_data)
    writer.Write()