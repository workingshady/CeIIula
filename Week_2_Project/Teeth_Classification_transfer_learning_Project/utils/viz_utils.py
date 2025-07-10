from typing import Dict, List, Tuple
import random
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-white')
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os


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

        # Use different colors for each bar
        colors = [
            'skyblue', 'mediumorchid', 'lightgreen', 'salmon',
            'lightyellow', 'lightcoral', 'orange', 'sandybrown',
            'slategray'
        ]
        colors = (colors * ((len(class_names) // len(colors)) + 1))[:len(class_names)]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(class_names, counts, color=colors, edgecolor='black')

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

    num_images = len(images)
    cols = min(5, num_images)
    rows = math.ceil(num_images / cols)
    plt.figure(figsize=(4 * cols, 4 * rows))

    for idx, (img_or_path, label) in enumerate(zip(images, labels)):
        if isinstance(img_or_path, str):
            img = plt.imread(img_or_path)
        elif isinstance(img_or_path, np.ndarray):
            img = img_or_path
        else:
            raise ValueError("Each image must be a file path or a numpy array.")

        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img)
        ax.set_title(label, fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_history(history,model_name):
    os.makedirs('visualizations', exist_ok=True)

    hist = history.history  # ✅ Safe!

    acc = (
        hist.get('accuracy') or
        hist.get('acc') or
        hist.get('categorical_accuracy')
    )
    val_acc = (
        hist.get('val_accuracy') or
        hist.get('val_acc') or
        hist.get('val_categorical_accuracy')
    )

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    if acc:
        ax[0].plot(acc, "b", label='Training Accuracy')
        ax[0].plot(val_acc, "r-", label='Validation Accuracy')
        ax[0].set_title('Model Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()

    ax[1].plot(hist['loss'], "b", label='Training Loss')
    ax[1].plot(hist['val_loss'], "r-", label='Validation Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.tight_layout()
    save_path = f'visualizations/training_history_{model_name}.png'
    fig.savefig(save_path)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, model_name: str, normalize: bool = False):
    """
    Plot confusion matrix with optional normalization.
    """
    os.makedirs('visualizations', exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    save_path = f'visualizations/confusion_matrix_{model_name}.png'
    fig.savefig(save_path)
    plt.show()
