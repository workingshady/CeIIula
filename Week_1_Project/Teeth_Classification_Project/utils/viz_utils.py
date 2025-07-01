from typing import Dict, List, Tuple
import random
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-white')
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_class_distribution(distribution: Dict[str, Dict[str, Dict[str, float]]]):
    """
    Plot the class distribution (counts and percentages) for each split.
    Args:
        distribution: Dict from get_class_distribution().
    """
    for split, classes in distribution.items():
        class_names = list(classes.keys())
        counts = [classes[class_name]['count'] for class_name in class_names]
        percentages = [classes[class_name]['percentage'] for class_name in class_names]

        colors = [
    'skyblue',
    'mediumorchid',
    'lightgreen',
    'salmon',
    'lightyellow',
    'lightcoral',

    'orange',
    'sandybrown',
    'slategray'
]


        plt.figure(figsize=(12, 6))
        bars = plt.bar(class_names, counts, color=random.choice(colors), edgecolor='black')

        # Annotate each bar with count and percentage
        for idx, (bar, count, perc) in enumerate(zip(bars, counts, percentages)):
            plt.annotate(
                f"{count} ({perc}%)",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center',
                va='bottom',
                fontsize=10
            )

        plt.title(f"{split} Class Distribution", fontsize=14)
        plt.xlabel("Class", fontsize=12)
        plt.ylabel("Number of Samples", fontsize=12)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()




def plot_image_sample(images, labels):
    """
    Plot a grid of images with their labels.

    Args:
        images: List of image paths or numpy arrays (image data).
        labels: List of labels corresponding to each image.
    """
    import math
    import numpy as np

    num_images = len(images)
    cols = min(5, num_images)
    rows = math.ceil(num_images / cols)
    plt.figure(figsize=(4 * cols, 4 * rows))

    for idx, (img_or_path, label) in enumerate(zip(images, labels)):
        # Determine if input is a path or an image array
        if isinstance(img_or_path, str):
            img = plt.imread(img_or_path)
        elif isinstance(img_or_path, np.ndarray):
            img = img_or_path
        else:
            raise ValueError("Each image must be a file path or a numpy array.")

        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img)
        ax.set_title(label, fontsize=20)
        ax.axis('off')

    plt.tight_layout()
    plt.show()



def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy',color='blue')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy',color='red')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss',color='blue')
    ax2.plot(history.history['val_loss'], label='Validation Loss',color='red')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'visualizations/training_history_{history.model.name}.png')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'visualizations/confusion_matrix.png')
    plt.show()


