from typing import Dict, List, Tuple
import random
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-white')
import numpy as np


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
