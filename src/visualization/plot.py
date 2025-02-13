from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


def visualize_dataset_split(train_set: List, val_set: List, test_set: List):
    """
    Visualize the distribution of dataset splits using a bar plot.

    Args:
        train_set: Training dataset
        val_set: Validation dataset
        test_set: Test dataset
    """

    colors = {
        "train": "#2ecc71",
        "validation": "#3498db",
        "test": "#e74c3c",
    }

    dataset_sizes = {
        "Train": len(train_set),
        "Validation": len(val_set),
        "Test": len(test_set),
    }

    pd.Series(dataset_sizes).plot(
        kind="bar", color=list(colors.values()), figsize=(10, 6)
    )

    for i, v in enumerate(dataset_sizes.values()):
        plt.text(i, v, str(v), ha="center", va="bottom")

    plt.title("Dataset Distribution", fontsize=14, pad=15)
    plt.ylabel("Numbers", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


def display_random_samples(
    images: np.ndarray,
    labels: np.ndarray,
    label_decoder: Dict[int, str],
    num_samples: int = 1,
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = "viridis",
    title_fontsize: int = 12,
) -> None:
    """
    Display random samples from the dataset with their labels.

    Args:
        images: Array of images
        labels: Array of corresponding labels
        label_decoder: Dictionary mapping label indices to human-readable names
        num_samples: Number of random samples to display
        figsize: Figure size (width, height)
        cmap: Colormap for displaying images
        title_fontsize: Font size for title
    """
    combined = list(zip(images, labels))

    samples = random.sample(combined, k=min(len(combined), num_samples))

    fig = plt.figure(figsize=figsize)
    for idx, (image, label) in enumerate(samples, 1):
        plt.subplot(1, num_samples, idx)
        plt.imshow(image, cmap=cmap)
        plt.title(f"{label_decoder[label]}", fontsize=title_fontsize)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
