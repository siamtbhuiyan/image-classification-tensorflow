# Imports
try: 
    import argparse
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import os
    import csv
    from keras.models import load_model
except Exception as error:
    print(f"{type(error).__name__}: {error}")
    if type(error).__name__ == 'ModuleNotFoundError':
       print("Please Check requirements.txt")
    raise SystemExit(0)

# Checking for GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print("WARNING: No GPU Found")
else:
    print('Found GPU at: {}'.format(device_name))

parser = argparse.ArgumentParser(description="Classification")

# Fixed Variables
img_height = 224
img_width = 224

# Define arguments
# Define arguments
parser.add_argument('--dataset_path', type=str, default='/content/drive/MyDrive/Github Repository/Classification/Lung Dataset Short', help='Path to the dataset')
parser.add_argument('--number_of_classes', type=int, default=5, help='Number of classes in the dataset')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone model to use')
parser.add_argument('--results_path', type=str, default='/content/drive/MyDrive/Github Repository/Classification/Results', help='Path to save the results')
parser.add_argument('--training_epoch', type=int, default=50, help='Number of training epochs')
parser.add_argument('--training_learning_rate', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--training_loss', type=str, default='SparseCategoricalCrossentropy', help='Loss function for training')
parser.add_argument('--training_lr_optimizer', type=str, default='ReduceLROnPlateau', help='Learning rate optimizer for training')

# Array arguments
parser.add_argument('--training_metrics', type=str, nargs='+', default=['accuracy'], help='Metrics for training (e.g., "accuracy", "precision")')
parser.add_argument('--finetuning_metrics', type=str, nargs='+', default=['accuracy'], help='Metrics for fine-tuning')
parser.add_argument('--classifier_list', type=str, nargs='+', default=["Dropout(0.2)", "Dense(1024, activation='relu')", "Flatten()", "Dense(number_of_classes, activation='softmax')"], help='List of classifier layers')

parser.add_argument('--finetune', type=bool, default=True, help='Whether to finetune the model')
parser.add_argument('--finetuning_epoch', type=int, default=50, help='Number of finetuning epochs')
parser.add_argument('--finetuning_learning_rate', type=float, default=0.0001, help='Learning rate for finetuning')
parser.add_argument('--finetuning_loss', type=str, default='SparseCategoricalCrossentropy', help='Loss function for finetuning')
parser.add_argument('--finetuning_lr_optimizer', type=str, default='ReduceLROnPlateau', help='Learning rate optimizer for finetuning')
parser.add_argument('--fine_tune_at', type=int, default=313, help='Layer number at which to begin fine-tuning')

# Parse arguments
args = parser.parse_args()

# Print parsed arguments
print("Dataset Path:", args.dataset_path)
print("Number of Classes:", args.number_of_classes)
print("Batch Size:", args.batch_size)
print("Backbone Model:", args.backbone)
print("Results Path:", args.results_path)
print("Training Epochs:", args.training_epoch)
print("Training Learning Rate:", args.training_learning_rate)
print("Training Loss:", args.training_loss)
print("Training Learning Rate Optimizer:", args.training_lr_optimizer)
print("Training Metrics:", args.training_metrics)
print("Finetuning:", args.finetune)
print("Finetuning Epochs:", args.finetuning_epoch)
print("Finetuning Learning Rate:", args.finetuning_learning_rate)
print("Finetuning Loss:", args.finetuning_loss)
print("Finetuning Learning Rate Optimizer:", args.finetuning_lr_optimizer)
print("Finetuning Metrics:", args.finetuning_metrics)
print("Fine Tune At Layer:", args.fine_tune_at)
print("Classifier List:", args.classifier_list)





