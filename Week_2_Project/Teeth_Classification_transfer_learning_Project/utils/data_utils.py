from pathlib import Path
from typing import Dict, List, Iterator, Tuple

import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image_dataset_from_directory


def walk_through_dir(dir_path):

  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def get_image_data_loader(
    data_dir: str = "data/Teeth_Dataset",
    split: str = "Training",
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    class_mode: str = "categorical",
    shuffle: bool = True,
    seed: int = 42,
    augment: bool = False,
    ) -> tf.data.Dataset:
    """
    Get a data loader for the dataset.
    Returns a tf.data.Dataset for the given split.
    If augment is True, applies basic data augmentation.
    """
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    if augment:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.0
        )
    else:
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.0
        )

    generator = datagen.flow_from_directory(
        split_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle,
        seed=seed
    )

    return generator


def get_tf_dataset_loader(
    data_dir: str = "data/Teeth_Dataset",
    split: str = "Training",
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    class_mode: str = "categorical",
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Create a data loading pipeline for the dataset.
    """
    dataset = image_dataset_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode=class_mode,
        shuffle=shuffle,
        seed=seed
    )
    return dataset

def get_class_distribution(
    data_dir: str = "data/Teeth_Dataset",exclude_classes: List[str] = []
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Get the distribution of samples across different classes in each split.
    Returns:
        Dict with splits as keys, each containing class counts and percentages.
    """
    distribution: Dict[str, Dict[str, Dict[str, float]]] = {}
    splits = ["Training", "Validation", "Testing"]

    for split in splits:
        split_path = Path(data_dir) / split
        distribution[split] = {}

        if split_path.exists():
            class_counts = {}
            total_samples = 0

            for class_dir in split_path.iterdir():
                if class_dir.name in exclude_classes:
                    continue

                if class_dir.is_dir():
                    # Count files directly without storing the list
                    count = sum(
                        1 for f in class_dir.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    )
                    class_counts[class_dir.name] = count
                    total_samples += count

            for class_name, count in class_counts.items():
                percentage = (count / total_samples * 100) if total_samples > 0 else 0
                distribution[split][class_name] = {
                    "count": count,
                    "percentage": round(percentage, 2)
                }

    return distribution

def get_image_paths_and_labels(
    data_dir: str = "data/Teeth_Dataset", split: str = "Training"
) -> Iterator[Tuple[str, str]]:
    """
    Yield (image_path, label) pairs from a specific split.
    """
    split_path = Path(data_dir) / split

    if split_path.exists():
       for class_dir in split_path.iterdir():
          if class_dir.is_dir():
              class_name = class_dir.name
              for image_file in class_dir.iterdir():
                  if image_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                      yield str(image_file), class_name
