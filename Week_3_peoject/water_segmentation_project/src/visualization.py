from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from preprocessing import create_data_loaders

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class WaterSegmentationVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.band_names = [
            "Coastal aerosol",
            "Blue",
            "Green",
            "Red",
            "NIR",
            "SWIR1",
            "SWIR2",
            "QA Band",
            "Merit DEM",
            "Copernicus DEM",
            "ESA world cover map",
            "Water occurrence probability",
        ]

    def plot_12_bands(
        self,
        image: np.ndarray,
        title: str = "12-Channel Image Bands",
        save_path: Optional[str] = None,
    ) -> None:
        if image.shape[-1] != 12:
            raise ValueError(f"Expected 12 channels, got {image.shape[-1]}")
        fig, axes = plt.subplots(3, 4, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")
        for i in range(12):
            row, col = divmod(i, 4)
            band = image[:, :, i]
            band_norm = (
                (band - band.min()) / (band.max() - band.min())
                if band.max() > band.min()
                else band
            )
            im = axes[row, col].imshow(band_norm, cmap="viridis", aspect="auto")
            axes[row, col].set_title(self.band_names[i], fontsize=12)
            axes[row, col].axis("off")
            plt.colorbar(im, ax=axes[row, col], shrink=0.8)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_rgb_composite(
        self,
        image: np.ndarray,
        rgb_bands: Tuple[int, int, int] = (3, 2, 1),
        title: str = "RGB Composite",
        save_path: Optional[str] = None,
    ) -> None:
        if image.shape[-1] < max(rgb_bands) + 1:
            raise ValueError("Image doesn't have enough bands for RGB composite")
        rgb = np.zeros((image.shape[0], image.shape[1], 3))
        for i, b in enumerate(rgb_bands):
            band = image[:, :, b]
            rgb[:, :, i] = (
                (band - band.min()) / (band.max() - band.min())
                if band.max() > band.min()
                else band
            )
        plt.figure(figsize=(8, 8))
        plt.imshow(rgb)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.axis("off")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_mask(
        self,
        mask: np.ndarray,
        title: str = "Binary Mask",
        save_path: Optional[str] = None,
    ) -> None:
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        plt.figure(figsize=(8, 8))
        plt.imshow(mask, cmap="gray", vmin=0, vmax=1, alpha=0.9)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.colorbar(label="Water (1) / Non-water (0)")
        plt.axis("off")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_image_and_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        rgb_bands: Tuple[int, int, int] = (3, 2, 1),
        title: str = "Image and Mask",
        save_path: Optional[str] = None,
    ) -> None:
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        img_h, img_w = image.shape[:2]
        rgb = np.zeros((img_h, img_w, 3))
        for i, b in enumerate(rgb_bands):
            band = image[:, :, b]
            rgb[:, :, i] = (
                (band - band.min()) / (band.max() - band.min())
                if band.max() > band.min()
                else band
            )
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(title, fontsize=16, fontweight="bold")
        axes[0].imshow(rgb)
        axes[0].set_title("RGB Composite", fontsize=14)
        axes[0].axis("off")
        im = axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1, alpha=0.9)
        axes[1].set_title("Water Mask", fontsize=14)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], label="Water (1) / Non-water (0)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_history(
        self,
        history: Dict[str, List[float]],
        plot_name: str = "Training History_exp_idx",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot training and validation loss/metrics curves.

        Args:
            history: Dictionary containing training history with metrics
            plot_name: Name of the plot for the title
            save_path: Optional path to save the plot
        """
        metrics = [key for key in history.keys() if not key.startswith("val_")]
        n_metrics = len(metrics)
        n_rows = (n_metrics + 1) // 2  # Round up division

        fig, axes = plt.subplots(n_rows, 2, figsize=self.figsize)
        fig.suptitle(plot_name, fontsize=16, fontweight="bold")

        # Flatten axes for easier iteration if there's only one row
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, metric in enumerate(metrics):
            row = idx // 2
            col = idx % 2

            # Plot training metric
            axes[row, col].plot(history[metric], label="Training")

            # Plot validation metric if available
            val_metric = f"val_{metric}"
            if val_metric in history:
                axes[row, col].plot(history[val_metric], label="Validation")

            axes[row, col].set_title(metric.replace("_", " ").title(), fontsize=12)
            axes[row, col].set_xlabel("Epoch", fontsize=10)
            axes[row, col].set_ylabel("Value", fontsize=10)
            axes[row, col].grid(True, linestyle="--", alpha=0.7)
            axes[row, col].legend(loc="best")

        # Remove empty subplots if odd number of metrics
        if n_metrics % 2 != 0:
            fig.delaxes(axes[-1, -1])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


import random


def create_sample_visualizations(
    data_loader, output_dir: str = "data/visualizations"
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    visualizer = WaterSegmentationVisualizer()
    if data_loader.image_files:
        idx = random.randint(0, len(data_loader.image_files) - 1)
        img = data_loader.load_image(data_loader.image_files[idx])
        mask = data_loader.load_mask(data_loader.label_files[idx])
        img_norm = data_loader.normalize_image(img)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        visualizer.plot_12_bands(
            img_norm,
            title=f"Sample 12-Channel Image Bands (idx={idx})",
            save_path=str(output_path / f"sample_bands_{idx}.png"),
        )
        visualizer.plot_rgb_composite(
            img_norm,
            title=f"Sample RGB Composite (idx={idx})",
            save_path=str(output_path / f"sample_rgb_{idx}.png"),
        )
        visualizer.plot_mask(
            mask,
            title=f"Sample Water Mask (idx={idx})",
            save_path=str(output_path / f"sample_mask_{idx}.png"),
        )
        visualizer.plot_image_and_mask(
            img_norm,
            mask,
            title=f"Sample Image and Mask (idx={idx})",
            save_path=str(output_path / f"sample_image_mask_{idx}.png"),
        )
        print(f"Visualizations saved to {output_path} for random index {idx}")
    else:
        print("No image files found for visualization.")


if __name__ == "__main__":
    data_dir = "data"
    _, _, _, loader = create_data_loaders(data_dir=data_dir, batch_size=32)
    create_sample_visualizations(loader)
