import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
import tensorflow as tf
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WaterSegmentationDataLoader:
    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (128, 128),
        num_channels: int = 12,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.num_channels = num_channels
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        self.image_files: List[str] = []
        self.label_files: List[str] = []
        self._scan_files()

    def _scan_files(self):
        """
        Scan and match image and label files, ensuring only one-to-one matches:
        1.tif <-> 1.png, and both must have the same image_size.
        """
        logger.info("Scanning files for strict one-to-one matches...")

        image_files: List[Path] = sorted(self.images_dir.glob("*.tif"))
        label_files: List[Path] = sorted(self.labels_dir.glob("*.png"))

        # Map by stem (without extension)
        image_stems = {f.stem: f for f in image_files}
        label_stems = {f.stem: f for f in label_files}

        matched_pairs = []
        unmatched_images = []
        unmatched_labels = []

        for stem, image_file in image_stems.items():
            if stem in label_stems:
                label_file = label_stems[stem]
                try:
                    # Check both files are readable and have the correct shape
                    with rasterio.open(str(image_file)) as src_img:
                        img = src_img.read()
                        img = np.transpose(img, (1, 2, 0))
                        if img.shape[:2] != self.image_size:
                            logger.warning(f"Image {image_file.name} shape {img.shape[:2]} does not match required {self.image_size}")
                            continue
                    with rasterio.open(str(label_file)) as src_mask:
                        mask = src_mask.read(1)
                        if mask.shape != self.image_size:
                            logger.warning(f"Label {label_file.name} shape {mask.shape} does not match required {self.image_size}")
                            continue
                    matched_pairs.append((str(image_file), str(label_file)))
                except Exception as e:
                    logger.warning(
                        f"Skipping pair due to error reading files: {image_file}, {label_file}, error: {e}"
                    )
            else:
                unmatched_images.append(image_file.name)

        # Find labels that do not have a matching image
        for stem, label_file in label_stems.items():
            if stem not in image_stems:
                unmatched_labels.append(label_file.name)

        if not matched_pairs:
            raise ValueError("No matching image-label pairs found!")

        self.image_files, self.label_files = zip(*matched_pairs)
        logger.info(f"Found {len(self.image_files)} strict one-to-one image-label pairs")
        logger.info(f"Unmatched images: {len(unmatched_images)} ({unmatched_images[:5]})")
        logger.info(f"Unmatched labels: {len(unmatched_labels)} ({unmatched_labels[:5]})")

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load a 12-channel TIF image. No resizing is performed; image must be (128, 128).
        """
        with rasterio.open(image_path) as src:
            image = src.read()  # (bands, H, W)
            image = np.transpose(image, (1, 2, 0))  # (H, W, bands)

        if image.shape[:2] != self.image_size:
            raise ValueError(
                f"Image shape {image.shape[:2]} does not match required size {self.image_size}."
            )

        if image.shape[-1] < self.num_channels:
            padding = np.zeros(
                (image.shape[0], image.shape[1], self.num_channels - image.shape[2]),
                dtype=image.dtype,
            )
            image = np.concatenate([image, padding], axis=-1)
        elif image.shape[-1] > self.num_channels:
            image = image[:, :, : self.num_channels]

        return image.astype(np.float32)

    def load_mask(self, mask_path: str) -> np.ndarray:
        """
        Load a binary mask. No resizing is performed; mask must be (128, 128).
        """
        try:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)  # Read first band
            if mask is None:
                raise ValueError(f"Could not load mask: {mask_path}")

            if mask.shape != self.image_size:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match required size {self.image_size}."
                )

            mask = (mask > 0).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)
            return mask

        except Exception as e:
            logger.error(f"Error loading mask {mask_path}: {e}")
            raise

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image data using per-channel min-max normalization .
        """
        normalized = np.zeros_like(image, dtype=np.float32)
        for ch in range(image.shape[-1]):
            ch_data = image[:, :, ch]
            min_val = np.min(ch_data)
            max_val = np.max(ch_data)
            if max_val > min_val:
                normalized[:, :, ch] = (ch_data - min_val) / (max_val - min_val)
            else:
                normalized[:, :, ch] = 0.0 # review later
        return normalized

    def split_data(
        self, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
        """
        Split data into train, validation, and test sets.
        """
        train_val_images, test_images, train_val_labels, test_labels = train_test_split(
            self.image_files,
            self.label_files,
            test_size=test_size,
            random_state=random_state,

            shuffle=True,
        )

        train_images, val_images, train_labels, val_labels = train_test_split(
            train_val_images,
            train_val_labels,
            test_size=val_size,
            random_state=random_state,
            shuffle=True,
        )

        logger.info(
            f"Data split - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}"
        )

        return (
            train_images,
            train_labels,
            val_images,
            val_labels,
            test_images,
            test_labels,
        )

    def create_dataset(
        self,
        image_paths: List[str],
        label_paths: List[str],
        batch_size: int = 16,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset from image and label paths.
        """

        def load_and_preprocess(image_path, label_path):
            def py_load_image(path):
                path_str = path.numpy().decode("utf-8")
                image = self.load_image(path_str)
                image = self.normalize_image(image)
                return image.astype(np.float32)

            def py_load_mask(path):
                path_str = path.numpy().decode("utf-8")
                mask = self.load_mask(path_str)
                return mask.astype(np.float32)

            image = tf.py_function(py_load_image, [image_path], tf.float32)
            mask = tf.py_function(py_load_mask, [label_path], tf.float32)

            image.set_shape([self.image_size[0], self.image_size[1], self.num_channels])
            mask.set_shape([self.image_size[0], self.image_size[1], 1])

            return image, mask

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_data_statistics(self) -> Dict:
        """
        Get statistics about the dataset.
        """
        stats = {
            "num_samples": len(self.image_files),
            "image_size": self.image_size,
            "num_channels": self.num_channels,
            "sample_files": {
                "image": self.image_files[:3] if len(self.image_files) > 0 else [],
                "label": self.label_files[:3] if len(self.label_files) > 0 else [],
            },
        }

        return stats


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Convenience function to create train, validation, and test data loaders.
    """
    data_loader = WaterSegmentationDataLoader(data_dir)

    train_images, train_labels, val_images, val_labels, test_images, test_labels = (
        data_loader.split_data(
            test_size=test_size, val_size=val_size, random_state=random_state
        )
    )

    train_dataset = data_loader.create_dataset(
        train_images,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataset = data_loader.create_dataset(
        val_images,
        val_labels,
        batch_size=batch_size,
        shuffle=False,
    )

    test_dataset = data_loader.create_dataset(
        test_images,
        test_labels,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_dataset, val_dataset, test_dataset, data_loader


if __name__ == "__main__":
    data_dir = "data"

    train_ds, val_ds, test_ds, loader = create_data_loaders(
        data_dir=data_dir, batch_size=32
    )

    stats = loader.get_data_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nTesting data loading...")
    for images, masks in train_ds.take(1):
        print(f"Batch shape - Images: {images.shape}, Masks: {masks.shape}")
        print(
            f"Image range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]"
        )
        print(f"Mask range: [{tf.reduce_min(masks):.3f}, {tf.reduce_max(masks):.3f}]")
        break
