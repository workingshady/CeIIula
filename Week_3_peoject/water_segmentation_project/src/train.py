import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import optimizers

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from callbacks_utils import (
    get_early_stopping_callback,
    get_model_checkpoint_callbacks,
    get_reduce_lr_on_plateau_callback,
    get_tensorboard_callback,
)
from model import UNetMultiChannel
from preprocessing import create_data_loaders
from visualization import WaterSegmentationVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_metrics() -> List[tf.keras.metrics.Metric]:
    """
    Create a list of metrics for water segmentation.
    """
    metrics = [
        tf.keras.metrics.BinaryIoU(name="iou", target_class_ids=[1]),
        tfa.metrics.F1Score(
            name="f1_score", num_classes=1, average="micro", threshold=0.5
        ),
        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
        tf.keras.metrics.Precision(name="precision", thresholds=0.5),
        tf.keras.metrics.Recall(name="recall", thresholds=0.5),
    ]
    return metrics


def create_model(
    input_shape: Tuple[int, int, int] = (128, 128, 12),
    num_classes: int = 1,
    base_filters: int = 64,
    learning_rate: float = 1e-4,
    padding: str = "same",
    dropout: float = 0.5,
    batch_norm: bool = True,
) -> Tuple[tf.keras.Model, tf.keras.optimizers.Optimizer]:
    """
    Create and compile the UNet model using Binary Crossentropy loss only.
    """
    logger.info(f"Creating UNet model with input_shape={input_shape}")

    # Create model
    unet = UNetMultiChannel(
        input_shape=input_shape,
        num_classes=num_classes,
        base_filters=base_filters,
        padding=padding,
        dropout=dropout,
        batch_norm=batch_norm,
    )
    model = unet.get_model()

    # Create optimizer
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    # Use only Binary Crossentropy loss
    loss = tf.keras.losses.BinaryCrossentropy()

    # Create metrics
    metrics = create_metrics()

    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logger.info(f"Model compiled successfully with Binary Crossentropy loss")
    return model, optimizer


def create_callbacks(
    model_name: str,
    checkpoint_dir: str,
    logs_dir: str,
    experiment_name: str,
    monitor: str = "val_loss",
    patience: int = 15,
    min_lr: float = 1e-7,
) -> List[tf.keras.callbacks.Callback]:
    """
    Create training callbacks.
    """
    callbacks = []

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Model checkpoint callbacks
    checkpoint_callbacks = get_model_checkpoint_callbacks(
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        monitor=monitor,
        save_weights_only=True,
        verbose=1,
    )
    callbacks.extend(checkpoint_callbacks)

    # Early stopping
    early_stopping = get_early_stopping_callback(
        monitor=monitor, patience=patience, mode="min", restore_best_weights=True
    )
    callbacks.append(early_stopping)

    # Learning rate reduction
    reduce_lr = get_reduce_lr_on_plateau_callback(
        monitor=monitor,
        factor=0.5,
        patience=patience // 2,
        verbose=1,
        mode="min",
        min_lr=min_lr,
        min_delta=0.0001,
    )
    callbacks.append(reduce_lr)

    # TensorBoard logging
    tensorboard = get_tensorboard_callback(
        base_dir=logs_dir, experiment_name=experiment_name
    )
    callbacks.append(tensorboard)

    logger.info(f"Created {len(callbacks)} callbacks")
    return callbacks


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    callbacks: List[tf.keras.callbacks.Callback],
    epochs: int = 100,
    verbose: int = 2,
) -> tf.keras.callbacks.History:
    """
    Train the model.
    """
    logger.info(f"Starting training for {epochs} epochs")

    # Calculate steps per epoch
    train_steps = len(list(train_dataset))
    val_steps = len(list(val_dataset))

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True,
    )

    logger.info("Training completed successfully")
    return history


def save_training_summary(
    model: tf.keras.Model,
    history: tf.keras.callbacks.History,
    model_name: str,
    output_dir: str,
    data_stats: Dict,
) -> None:
    """
    Save training summary and model information.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model summary
    with open(output_path / f"{model_name}_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    history_dict = history.history
    np.save(output_path / f"{model_name}_history.npy", history_dict)

    # Save data statistics
    import json

    with open(output_path / f"{model_name}_data_stats.json", "w") as f:
        json.dump(data_stats, f, indent=2)

    logger.info(f"Training summary saved to {output_path}")


def create_sample_predictions(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    data_loader,
    output_dir: str,
    num_samples: int = 20,
    verbose: int = 1,
    threshold: float = 0.5,
) -> None:
    """
    Create sample predictions for visualization, including binary thresholded predictions.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    visualizer = WaterSegmentationVisualizer()

    # Get a batch from test dataset
    for images, masks in test_dataset.take(1):
        # Make predictions
        predictions = model.predict(images, verbose=verbose)

        # Create visualizations for first few samples
        for i in range(min(num_samples, len(images))):
            # Get original image (first 3 bands for RGB)
            img = images[i].numpy()
            mask = masks[i].numpy()
            pred = predictions[i]

            # Apply binary threshold to prediction
            pred_binary = (pred >= threshold).astype(pred.dtype)

            # Create RGB composite
            rgb_bands = (3, 2, 1)  # Red, Green, Blue bands
            if img.shape[-1] >= max(rgb_bands) + 1:
                # Plot ground truth
                visualizer.plot_image_and_mask(
                    img,
                    mask,
                    rgb_bands=rgb_bands,
                    title=f"Sample {i+1} - Ground Truth",
                    save_path=str(output_path / f"sample_{i+1}_ground_truth.png"),
                )

                # Plot raw prediction
                visualizer.plot_image_and_mask(
                    img,
                    pred,
                    rgb_bands=rgb_bands,
                    title=f"Sample {i+1} - Prediction (Raw)",
                    save_path=str(output_path / f"sample_{i+1}_prediction_raw.png"),
                )

                # Plot binary thresholded prediction
                visualizer.plot_image_and_mask(
                    img,
                    pred_binary,
                    rgb_bands=rgb_bands,
                    title=f"Sample {i+1} - Prediction (Binary, threshold={threshold})",
                    save_path=str(output_path / f"sample_{i+1}_prediction_binary.png"),
                )

        break

    logger.info(f"Sample predictions saved to {output_path}")


def main():

    data_dir = "data"
    batch_size = 16
    test_size = 0.2
    val_size = 0.2
    input_shape = (128, 128, 12)
    num_classes = 1
    base_filters = 64
    learning_rate = 1e-4
    epochs = 100
    patience = 15
    min_lr = 1e-7
    model_name = "unet_water_segmentation"
    output_dir = "results"
    checkpoint_dir = "models"
    logs_dir = "logs"
    experiment_name = "water_segmentation"
    random_state = 42

    verbose = 2
    create_samples = False

    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        train_dataset, val_dataset, test_dataset, data_loader = create_data_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )

        # Get data statistics
        data_stats = data_loader.get_data_statistics()
        logger.info(f"Data statistics: {data_stats}")

        # Create model
        logger.info("Creating model...")
        model, optimizer = create_model(
            input_shape=input_shape,
            num_classes=num_classes,
            base_filters=base_filters,
            learning_rate=learning_rate,
        )

        # Print model summary
        model.summary()

        # Create callbacks
        logger.info("Creating callbacks...")
        callbacks = create_callbacks(
            model_name=model_name,
            checkpoint_dir=checkpoint_dir,
            logs_dir=logs_dir,
            experiment_name=experiment_name,
            monitor="val_loss",
            patience=patience,
            min_lr=min_lr,
        )

        # Train model
        logger.info("Starting training...")
        history = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            callbacks=callbacks,
            epochs=epochs,
            verbose=verbose,
        )

        # Save training summary
        logger.info("Saving training summary...")
        save_training_summary(
            model=model,
            history=history,
            model_name=model_name,
            output_dir=output_dir,
            data_stats=data_stats,
        )

        # Create sample predictions if requested
        if create_samples:
            logger.info("Creating sample predictions...")
            create_sample_predictions(
                model=model,
                test_dataset=test_dataset,
                data_loader=data_loader,
                output_dir=output_dir,
            )

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
