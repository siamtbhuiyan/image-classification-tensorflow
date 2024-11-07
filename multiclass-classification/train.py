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

# Argument Parser
parser = argparse.ArgumentParser(description="Classification")

# Fixed Variables
img_height = 224
img_width = 224

# Define arguments
parser.add_argument('--dataset_path', type=str, help='Path to the dataset', required=True)
parser.add_argument('--number_of_classes', type=int, required=True, help='Number of classes in the dataset')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--backbone', type=str, default='ResNet50', help='Backbone model to use')
parser.add_argument('--results_path', type=str, default='./Results', help='Path to save the results')
parser.add_argument('--training_epoch', type=int, default=50, help='Number of training epochs')
parser.add_argument('--training_learning_rate', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--training_optimizer', type=str, default='Adam', help='Optimizer for Training')

# Array arguments
parser.add_argument('--training_metrics', type=str, nargs='+', default=['accuracy'], help='Metrics for training (e.g., "accuracy", "precision")')
parser.add_argument('--classifier_list', type=str, nargs='+', default=["GlobalAveragePooling2D()", "Dense(1024, activation='relu')", "Dense(512, activation='relu')", "Dense(number_of_classes, activation='softmax')"], help='List of classifier layers')

# Parse arguments
args = parser.parse_args()

# Assuming args.results_path is the root path where you want to save all files
training_model_save_path = os.path.join(args.results_path, 'model.keras')
training_history_save_path = os.path.join(args.results_path, 'history.csv')
training_history_plot_save_path = os.path.join(args.results_path, 'history.png')
training_confusion_matrix_save_path = os.path.join(args.results_path, 'confusion_matrix.png')
training_roc_curve_save_path = os.path.join(args.results_path, 'roc_curve.png')


def get_backbone(args, img_height, img_width):
    """
    Get the backbone model based on user input.
    """
    print("Getting backbone model...")
    backbone_dict = {
        'ConvNeXtBase': tf.keras.applications.ConvNeXtBase,
        'ConvNeXtLarge': tf.keras.applications.ConvNeXtLarge,
        'ConvNeXtSmall': tf.keras.applications.ConvNeXtSmall,
        'ConvNeXtTiny': tf.keras.applications.ConvNeXtTiny,
        'ConvNeXtXLarge': tf.keras.applications.ConvNeXtXLarge,
        'DenseNet121': tf.keras.applications.DenseNet121,
        'DenseNet169': tf.keras.applications.DenseNet169,
        'DenseNet201': tf.keras.applications.DenseNet201,
        'EfficientNetB0': tf.keras.applications.EfficientNetB0,
        'EfficientNetB1': tf.keras.applications.EfficientNetB1,
        'EfficientNetB2': tf.keras.applications.EfficientNetB2,
        'EfficientNetB3': tf.keras.applications.EfficientNetB3,
        'EfficientNetB4': tf.keras.applications.EfficientNetB4,
        'EfficientNetB5': tf.keras.applications.EfficientNetB5,
        'EfficientNetB6': tf.keras.applications.EfficientNetB6,
        'EfficientNetB7': tf.keras.applications.EfficientNetB7,
        'EfficientNetV2B0': tf.keras.applications.EfficientNetV2B0,
        'EfficientNetV2B1': tf.keras.applications.EfficientNetV2B1,
        'EfficientNetV2B2': tf.keras.applications.EfficientNetV2B2,
        'EfficientNetV2B3': tf.keras.applications.EfficientNetV2B3,
        'EfficientNetV2L': tf.keras.applications.EfficientNetV2L,
        'EfficientNetV2M': tf.keras.applications.EfficientNetV2M,
        'EfficientNetV2S': tf.keras.applications.EfficientNetV2S,
        'InceptionResNetV2': tf.keras.applications.InceptionResNetV2,
        'InceptionV3': tf.keras.applications.InceptionV3,
        'MobileNet': tf.keras.applications.MobileNet,
        'MobileNetV2': tf.keras.applications.MobileNetV2,
        'MobileNetV3Large': tf.keras.applications.MobileNetV3Large,
        'MobileNetV3Small': tf.keras.applications.MobileNetV3Small,
        'NASNetLarge': tf.keras.applications.NASNetLarge,
        'NASNetMobile': tf.keras.applications.NASNetMobile,
        'ResNet101': tf.keras.applications.ResNet101,
        'ResNet101V2': tf.keras.applications.ResNet101V2,
        'ResNet152': tf.keras.applications.ResNet152,
        'ResNet152V2': tf.keras.applications.ResNet152V2,
        'ResNet50': tf.keras.applications.ResNet50,
        'ResNet50V2': tf.keras.applications.ResNet50V2,
        'VGG16': tf.keras.applications.VGG16,
        'VGG19': tf.keras.applications.VGG19,
        'Xception': tf.keras.applications.Xception,
    }
    
    print(f"Selected backbone: {args.backbone}")
    # Get the model class from the dictionary, or raise an error if not found
    backbone_class = backbone_dict.get(args.backbone)
    if not backbone_class:
        raise ValueError(f"Backbone {args.backbone} is not supported.")
    
    # Initialize and return the model with specified parameters
    return backbone_class(include_top=False, weights="imagenet", input_shape=(img_height, img_width, 3))


def get_optimizer(args):
    """
    Get the optimizer based on user input.
    """
    print("Getting optimizer...")
    optimizer_dict = {
        'SGD': tf.keras.optimizers.SGD,
        'RMSprop': tf.keras.optimizers.RMSprop,
        'Adam': tf.keras.optimizers.Adam,
        'AdamW': tf.keras.optimizers.AdamW,
        'Adadelta': tf.keras.optimizers.Adadelta,
        'Adagrad': tf.keras.optimizers.Adagrad,
        'Adamax': tf.keras.optimizers.Adamax,
        'Nadam': tf.keras.optimizers.Nadam,
        'Ftrl': tf.keras.optimizers.Ftrl
    }

    # Get the optimizer class from the dictionary, or raise an error if not found
    optimizer_class = optimizer_dict.get(args.training_optimizer)
    if not optimizer_class:
        raise ValueError(f"Optimizer {args.training_optimizer} is not supported.")
    
    # Initialize and return the optimizer with a specified learning rate
    return optimizer_class(learning_rate=args.training_learning_rate)


training_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

training_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(training_model_save_path, save_weights_only=False, save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(),
]

def generates_dataset(dataset_path, batch_size, img_height, img_width):
    """
    Generates TensorFlow datasets for training, validation, and testing from a directory.
    """
    print("Generating datasets from the directory...")
    # Create train dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=f'{dataset_path}/train',
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    # Create validation dataset
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=f'{dataset_path}/validation',
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    # Create test dataset
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=f'{dataset_path}/test',
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    return train_dataset, validation_dataset, test_dataset

def generate_model(pretrained_model, classifier_list, number_of_classes):
    """
    Generates a new model by adding custom classification layers on top of a pre-trained model.
    """
    print("Generating model with custom classifier layers...")
    # Freeze the base model layers
    for layer in pretrained_model.layers:
        layer.trainable = False

    # Add custom layers on top of the pre-trained model based on classifier_list
    x = pretrained_model.output
    for layer_str in classifier_list:
        # Replace 'number_of_classes' with the actual value in the string if present
        layer_str = layer_str.replace("number_of_classes", str(number_of_classes))
        # Convert the string into actual layer objects and apply
        x = eval(f"tf.keras.layers.{layer_str}")(x)
    
    # Final model with classifier layers
    model = tf.keras.Model(inputs=pretrained_model.input, outputs=x)

    return model

def plot_history(history, save_path):
    """
    Plots the training history of a TensorFlow or Keras model.

    Args:
    - history (tf.keras.callbacks.History): Training history containing loss and metrics.
    - save_path (str): File path to save the plot as an image.

    This function takes the training history of a TensorFlow or Keras model and plots the training and validation
    loss as well as the training and validation accuracy over epochs. The resulting plots provide insights into
    the model's performance and training progress. It also saves the plot as an image to the specified file path.
    """
    fig, (ax1, ax2) = plt.subplots(2)

    fig.set_size_inches(18.5, 10.5)

    # Plot loss
    ax1.set_title('Loss')
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='validation')
    ax1.set_ylabel('Loss')

    # Determine upper bound of y-axis
    max_loss = max(history.history['loss'] + history.history['val_loss'])

    ax1.set_ylim([0, np.ceil(max_loss)])
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'])

    # Plot accuracy
    ax2.set_title('Accuracy')
    ax2.plot(history.history['accuracy'], label='train')
    ax2.plot(history.history['val_accuracy'], label='validation')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])
    plt.savefig(save_path)

def plot_confusion_matrix(dataset, model, class_names, save_path):
    """
    Plots the confusion matrix for a TensorFlow or Keras model's predictions.

    Args:
    - dataset (tf.data.Dataset): TensorFlow dataset containing images and labels.
    - model (tf.keras.Model): TensorFlow or Keras model for making predictions.
    - class_names (list): List of class names corresponding to the labels.
    - save_path (str): File path to save the plot as an image.

    Returns:
    - cm (ndarray of shape (n_classes, n_classes)): Confusion matrix.

    This function takes a TensorFlow dataset containing images and labels, a trained TensorFlow or Keras model,
    and a list of class names, and plots the confusion matrix for the model's predictions. It computes the
    confusion matrix by comparing the actual labels from the dataset with the predicted labels obtained
    from the model's predictions. It also saves the plot as an image to the specified file path.
    """
    actual_labels = []
    predicted_labels = []

    for images, actual_labels_batch in dataset:
        predictions = model.predict(images)
        predicted_labels_batch = np.argmax(predictions, axis=1)
        actual_labels.extend(actual_labels_batch)
        predicted_labels.extend(predicted_labels_batch)

    actual_labels = np.array(actual_labels)
    predicted_labels = np.array(predicted_labels)

    cm = confusion_matrix(actual_labels, predicted_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('Actual labels')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    return cm

def plot_roc_curve(model, test_dataset, class_names, save_path):
    """
    Plots the ROC curve for a TensorFlow or Keras model's predictions.

    Args:
    - model (tf.keras.Model): TensorFlow or Keras model for making predictions.
    - test_dataset (tf.data.Dataset): TensorFlow dataset containing test images and labels.
    - class_names (list): List of class names corresponding to the labels.
    - save_path (str): File path to save the plot as an image.


    This function takes a TensorFlow or Keras model, a TensorFlow dataset containing test images and labels,
    and a list of class names, and plots the ROC curve for the model's predictions. It computes the ROC curve
    and ROC area for each class, providing insights into the model's performance across different classes.
    Also saves the plot as an image.
    """
    y_true = []
    y_scores = []

    for inputs, labels in test_dataset:
        outputs = model.predict(inputs)
        y_true.extend(labels.numpy())
        y_scores.extend(outputs)

    y_true = label_binarize(y_true, classes=range(len(class_names)))
    y_scores = np.array(y_scores)

    # Compute ROC curve and ROC area for each class
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        print(f"Class {name} ROC AUC: {roc_auc:.3f}")
        plt.plot(fpr, tpr, label=f'Class {name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)

# Main
def main():
    print("Starting the process...")

    # Load and print backbone model
    pretrained_model = get_backbone(args, img_height, img_width)
    
    # Generate datasets
    train_dataset, validation_dataset, test_dataset = generates_dataset(args.dataset_path, args.batch_size, img_height, img_width)

    # Generate model
    model = generate_model(pretrained_model, args.classifier_list, args.number_of_classes)

    # Compile model
    model.compile(
        optimizer=get_optimizer(args),
        loss=training_loss,
        metrics=args.training_metrics
    )

    # Train model
    print("Training the model...")
    history = model.fit(
        train_dataset,
        epochs=args.training_epoch,
        validation_data=validation_dataset,
        callbacks=training_callbacks
    )

    # Save training history to CSV
    print("Saving training history...")
    pd.DataFrame(history.history).to_csv(training_history_save_path)

    # Plot training history
    print("Plotting training history...")
    plot_history(history, training_history_plot_save_path)

    # Evaluate on test set
    print("Evaluating the model on the test set...")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Confusion Matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(test_dataset, model, class_names=train_dataset.class_names, save_path=training_confusion_matrix_save_path)

    # ROC Curve
    print("Generating ROC curve...")
    # You can implement the ROC curve here, using the previously shown code example.
    plot_roc_curve(model, test_dataset, train_dataset.class_names, training_roc_curve_save_path)

    print("Process completed.")

# Call the main function to start the process
if __name__ == '__main__':
    main()
