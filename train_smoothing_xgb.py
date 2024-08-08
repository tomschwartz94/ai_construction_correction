#!/usr/bin/env python3
import json
import os
import time

import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
import xgboost as xgb
import math
import matplotlib.pyplot as plt

from conversions_and_misc import make_grid_binary
# Custom imports
from neighborhood_extraction import extract_2D_full_data_with_error_range
#from data_analysis import rounded_predictions
#from hk import plot_and_save_losses

# Paths and parameters
inp_path = 'dat/input_structures/training/addalloy_128.png'
#inp_path = './dat/0_original_structures/addalloy_512.png'
multiplicator = 5  # Each error value will be applied that many times to an input img
error_range = range(0, 15 + 1, 5)
window_size  = 13
# Load and preprocess image
train_image = inp_path
train_im = Image.open(train_image)
train_arr = list(train_im.getdata(0))
# test_arr = [(x, x, x) for x in train_arr]
train_picture = make_grid_binary(train_arr)

# Extract training data from image
inputs, outputs = extract_2D_full_data_with_error_range(train_picture, window_size, error_range, multiplicator)  # inputs.shape: (n_data, ws*ws-1)

print("Data amount in tf variables")
print(f"Shape of inputs: {inputs.shape}")
print(f"Shape of outputs/labels: {outputs.shape}")

# Prepare the datasets
train_data, test_data, train_labels, test_labels = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

# Reshape for XGBoost
#train_data_reshaped = train_data.reshape((train_data.shape[0], -1))
#test_data_reshaped = test_data.reshape((test_data.shape[0], -1))

# Define and train the XGBoost model with GPU support
eval_set = [(train_data, train_labels), (test_data, test_labels)]
model = xgb.XGBClassifier(
    n_estimators=5000,  # Increased number of estimators
    max_depth=10,
    learning_rate=0.05,  # Reduced learning rate
    random_state=42,
    tree_method='hist',  # Use CPU for training
    predictor='cpu_predictor',  # Use CPU for prediction
    reg_alpha=0.1,  # L1 regularization term
    reg_lambda=1  # L2 regularization term
)

model.fit(
    train_data,
    train_labels, 
    eval_metric="logloss", 
    eval_set=eval_set, 
    early_stopping_rounds=30,  # Stop if no improvement in 10 rounds
    verbose=False
)
# Save the model
model.save_model('model_xgb.json')

with open('model_xgb.json') as json_file:
    json_decoded = json.load(json_file)

json_decoded['window_size']= window_size
json_decoded['error_range']=[*error_range]
json_decoded['train_arr']= train_arr
json_decoded['multiplicator']= multiplicator

with open('model_xgb.json', 'w') as json_file:
    json.dump(json_decoded, json_file)
#model_dict = open('model_xgb.json', 'rb').read()

# Predict on the test set
y_pred = model.predict(test_data)

# Evaluate the model
accuracy = accuracy_score(test_labels, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Retrieve and print training and validation log loss
results = model.evals_result()
training_loss = results['validation_0']['logloss']
validation_loss = results['validation_1']['logloss']
plt.figure(figsize=(10, 6))
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Log Loss')
plt.title('Training and Validation Log Loss')
plt.legend()
plt.savefig('training_validation_loss.png')
plt.show()
print(f'Training log loss: {training_loss[-1]:.4f}')
print(f'Validation log loss: {validation_loss[-1]:.4f}')

y_pred_prob = model.predict_proba(test_data)[:, 1]  # Probability of the positive class
# Calculate ROC curve
fpr, tpr, _ = roc_curve(test_labels, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()
