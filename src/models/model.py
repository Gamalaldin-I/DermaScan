from tensorflow.keras.layers import (
    Input,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class DeepSkinClassifier:
    """A deep learning classifier for skin condition analysis.

    This class implements a CNN architecture for classifying skin conditions
    into 7 different categories using TensorFlow/Keras.
    """

    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        """Initialize the DeepSkinClassifier.

        Args:
            input_shape: Tuple specifying input dimensions (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = self.create_architecture()

    def create_architecture(self) -> Sequential:
        """Create the CNN model architecture.

        Returns:
            Sequential: Compiled Keras Sequential model
        """
        model = Sequential(
            [
                Input(shape=self.input_shape),
                Conv2D(64, (3, 3), padding="same", activation="relu"),
                Conv2D(64, (3, 3), padding="same", activation="relu"),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), padding="same", activation="relu"),
                Dropout(0.0),
                Conv2D(128, (3, 3), padding="same", activation="relu"),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation="relu"),
                Dense(7, activation="softmax"),
            ]
        )

        return model

    def configure(self) -> None:
        """Configure the model with optimizer, loss function and metrics."""
        self.model.compile(
            optimizer=Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train_model(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: np.ndarray,
        val_labels: np.ndarray,
    ) -> tf.keras.callbacks.History:
        """Train the model with the provided data.

        Args:
            train_data: Training data
            train_labels: Training labels
            val_data: Validation data
            val_labels: Validation labels
            checkpoint_path: Path to save model checkpoints

        Returns:
            History object containing training metrics
        """
        training_history = self.model.fit(
            train_data,
            train_labels,
            epochs=20,
            batch_size=64,
            validation_data=(val_data, val_labels),
        )
        return training_history

    def evaluate_model(
        self, test_data: np.ndarray, test_labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance on test data.

        Args:
            test_data: Test data
            test_labels: Test labels

        Returns:
            Dict containing loss and accuracy metrics
        """
        results = self.model.evaluate(test_data, test_labels)
        return {
            "loss": round(results[0], 3),
            "accuracy": round(results[1], 3),
        }

    def visualize_training(
        self,
        training_history: tf.keras.callbacks.History,
        figsize: Tuple[int, int] = (14, 6),
    ) -> None:
        """Visualize training metrics over epochs.

        Args:
            training_history: History object from model training
            figsize: Figure size for the plot
        """
        fig, (acc_plot, loss_plot) = plt.subplots(1, 2, figsize=figsize)

        acc_plot.plot(
            training_history.history["accuracy"],
            label="Train",
            color="#1f77b4",
            linewidth=2,
        )

        acc_plot.plot(
            training_history.history["val_accuracy"],
            label="Validation",
            color="#ff7f0e",
            linestyle="--",
        )

        acc_plot.set_title("Model Accuracy")
        acc_plot.set_ylabel("Accuracy")
        acc_plot.set_xlabel("Epoch")

        acc_plot.grid(True, alpha=0.3)
        acc_plot.legend()

        loss_plot.plot(
            training_history.history["loss"],
            label="Train",
            color="#2ca02c",
            linewidth=2,
        )

        loss_plot.plot(
            training_history.history["val_loss"],
            label="Validation",
            color="#d62728",
            linestyle="--",
        )

        loss_plot.set_title("Loss")
        loss_plot.set_ylabel("Loss")
        loss_plot.set_xlabel("Epoch")

        loss_plot.grid(True, alpha=0.3)
        loss_plot.legend()

        plt.tight_layout()
        plt.show()

    def predict_classes(self, input_data: np.ndarray) -> np.ndarray:
        """Predict classes for input data.

        Args:
            input_data: Input data to classify

        Returns:
            Array of predicted class probabilities
        """
        return self.model.predict(input_data)

    def save_model(self, filepath: str) -> None:
        """Save model weights to file.

        Args:
            filepath: Path to save model
        """
        self.model.save(filepath)

    def load_model(self, filepath: str) -> Sequential:
        """Load model weights from file.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded Keras model
        """
        return models.load_model(filepath)

    def architecture_summary(self) -> None:
        """Print model architecture summary."""
        return self.model.summary()
