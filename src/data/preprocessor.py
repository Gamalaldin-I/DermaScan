from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Any
import numpy as np
import cv2


class ImagePreprocessor:
    """
    A class for preprocessing images and preparing datasets for machine learning.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)) -> None:
        """
        Initialize the ImagePreprocessor.

        Args:
            target_size (Tuple[int, int]): Target dimensions for image resizing (width, height)
        """
        self.target_size = target_size
        self.X_test = None
        self.y_test = None
        self.y_train = None
        self.X_train = None

    def process_single_image(self, image_path: List[str]) -> np.ndarray:
        """
        Load and preprocess a single image.

        Args:
            image_path (List[str]): Path to the image file

        Returns:
            np.ndarray: Preprocessed image array
        """
        image = cv2.imread(image_path[0])
        normalized_image = (
            cv2.resize(image, self.target_size).astype(np.float32) / 255.0
        )
        return normalized_image

    def load_images(self, image_paths: List[List[str]]) -> List[np.ndarray]:
        """
        Process multiple images.

        Args:
            image_paths (List[List[str]]): List of image paths

        Returns:
            List[np.ndarray]: List of preprocessed images
        """
        return [self.process_single_image(path) for path in image_paths]

    @staticmethod
    def encode_target(labels: Any, label_mapping: Dict) -> np.ndarray:
        """
        Encode categorical labels to numerical values.

        Args:
            labels: Input labels to encode
            label_mapping (Dict): Mapping dictionary for label encoding

        Returns:
            np.ndarray: Encoded labels
        """
        return labels.map(label_mapping).astype(np.int16)

    def create_train_test_split(
        self, features: np.ndarray, labels: np.ndarray, test_ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features,
            labels,
            test_size=test_ratio,
            stratify=labels,
            shuffle=True,
            random_state=0,
        )

        return self.X_train, self.X_test, self.y_train, self.y_test
