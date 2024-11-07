# Multiclass Image Classification with Transfer Learning

This project provides a Python script for training a multiclass classification model using transfer learning with a customizable backbone (e.g., ResNet50, DenseNet121, EfficientNetB3, etc.) and various classifiers. The script uses TensorFlow/Keras for model training and evaluation. It includes functionalities for:

- Loading and preprocessing the dataset.
- Selecting a pre-trained backbone model.
- Training the model with custom layers for classification.
- Saving the model and training history.
- Generating performance metrics, including confusion matrix and ROC curve.


## Project Setup

Clone this repository:

```bash
git clone https://github.com/siamtbhuiyan/image-classification-tensorflow.git
cd multiclass-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Script

The script can be run from the command line, and it accepts several arguments. Here's the basic syntax:

```bash
python train.py --dataset_path <path_to_dataset> --number_of_classes <num_classes> --batch_size <batch_size> --backbone <backbone_model> --results_path <results_directory> --training_epoch <epochs> --training_learning_rate <learning_rate> --training_optimizer <optimizer> --training_metrics <metrics> --classifier_list <classifier_layers>
```

## Arguments

- `--dataset_path`: Required. Path to the dataset directory containing images and labels.
- `--number_of_classes`: Required. Number of classes in your dataset.
- `--batch_size`: Optional. Batch size for training. Default is 16.
- `--backbone`: Optional. Pre-trained backbone model for transfer learning. Default is ResNet50. You can choose from models like DenseNet121, EfficientNetB0, etc. Backbones can be selected from [here](https://www.tensorflow.org/api_docs/python/tf/keras/applications).
- `--results_path`: Optional. Directory to save the trained model and results (confusion matrix, history, etc.). Default is `./Results`.
- `--training_epoch`: Optional. Number of epochs to train. Default is 50.
- `--training_learning_rate`: Optional. Learning rate for training. Default is 0.0001.
- `--training_optimizer`: Optional. Optimizer for training. Default is Adam. Optimizers can be selected from [here](https://keras.io/api/optimizers/).
- `--training_metrics`: Optional. Metrics to track during training (e.g., accuracy, precision, recall).
- `--classifier_list`: Optional. List of classifier layers to append after the backbone model for fine-tuning.

## Example Command

```bash
python train.py --dataset_path ./data --number_of_classes 10 --batch_size 32 --backbone ResNet50 --results_path ./output --training_epoch 100 --training_learning_rate 0.0001 --training_optimizer Adam --training_metrics accuracy --classifier_list "GlobalAveragePooling2D()", "Dense(1024, activation='relu')", "Dense(512, activation='relu')", "Dense(10, activation='softmax')"
```

This command will use the ResNet50 backbone, train for 100 epochs with a batch size of 32, and save the results in the `./output` directory.

## Dataset

Provide the dataset folder path as an argument in the script. The dataset must be in the following format.

```
dataset/
    ├── train/
    │   ├── class_1/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── class_2/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── ...
    ├── validation/
    │   ├── class_1/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── class_2/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── ...
    └── test/
        ├── class_1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── class_2/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── ...
```

## Results

After training, the following files will be saved in the `results_path`:

- `model.keras`: The trained model in Keras format.
- `history.csv`: A CSV file containing the training history (loss and accuracy values for each epoch).
- `history.png`: A plot of the training and validation loss/accuracy over epochs.
- `confusion_matrix.png`: A confusion matrix for the model's predictions.
- `roc_curve.png`: A plot of the ROC curve with AUC score for classification performance.


## Acknowledgements

This project uses TensorFlow for deep learning.

Special thanks to the authors of the various transfer learning models in TensorFlow/Keras.

