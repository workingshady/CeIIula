import os
import random
from pathlib import Path
from typing import Dict, List, Iterator, Tuple

import numpy as np


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
