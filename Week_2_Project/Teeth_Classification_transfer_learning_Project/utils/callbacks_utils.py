import datetime
import os
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


def get_early_stopping_callback(
    monitor: str = "val_loss",
    patience: int = 10,
    mode: str = "min",
    restore_best_weights: bool = True,
) -> tf.keras.callbacks.EarlyStopping:
    """
    Returns an EarlyStopping callback with the specified configuration.
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        restore_best_weights=restore_best_weights,
    )


def get_model_checkpoint_callbacks(
    model_name: str,
    checkpoint_dir: str,
    monitor: str = "val_loss",
    save_weights_only: bool = True,
    verbose: int = 1,
) -> List[tf.keras.callbacks.ModelCheckpoint]:
    """
    Returns a list of ModelCheckpoint callbacks:
    - One that saves weights after every epoch.
    - One that saves only the best weights (according to the monitor).
    """
    all_epochs_dir = os.path.join(checkpoint_dir, "all_epochs")
    os.makedirs(all_epochs_dir, exist_ok=True)

    checkpoint_all_epochs = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(all_epochs_dir, f"{model_name}_checkpoint.h5"),
        save_best_only=False,
        save_weights_only=save_weights_only,
        verbose=verbose,
    )

    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, f"{model_name}_best.h5"),
        monitor=monitor,
        save_best_only=True,
        save_weights_only=save_weights_only,
        verbose=verbose,
    )

    return [checkpoint_all_epochs, checkpoint_best]


def get_tensorboard_callback(
    base_dir: str, experiment_name: str
) -> tf.keras.callbacks.TensorBoard:
    """
    Returns a TensorBoard callback for logging training metrics.
    Configured to avoid JSON serialization errors with EagerTensors.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_dir, experiment_name, timestamp)
    print(f"[INFO] Saving TensorBoard logs to: {log_dir}")
    return TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
    )


def get_safe_tensorboard_callback(
    base_dir: str, experiment_name: str
) -> tf.keras.callbacks.TensorBoard:
    """
    Returns a TensorBoard callback with minimal logging to completely avoid
    JSON serialization issues. Use this if the regular TensorBoard callback
    still causes problems.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_dir, experiment_name, timestamp)
    print(f"[INFO] Saving safe TensorBoard logs to: {log_dir}")
    return TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,  # Disable graph writing completely
        write_images=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        write_steps_per_second=False,  # Disable additional metrics that might cause issues
    )


def get_reduce_lr_on_plateau_callback(
    monitor: str = "val_loss",
    factor: float = 0.1,
    patience: int = 10,
    verbose: int = 1,
    mode: str = "min",
    min_lr: float = 1e-7,
    min_delta: float = 0.0001,
) -> tf.keras.callbacks.ReduceLROnPlateau:
    """
    Returns a ReduceLROnPlateau callback with the specified configuration.
    """
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        verbose=verbose,
        mode=mode,
        min_delta=min_delta,
        min_lr=min_lr,
    )
