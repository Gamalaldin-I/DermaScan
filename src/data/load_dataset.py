import glob
from typing import List, Dict


class ImageLoader:
    """
    A class to handle image loading and label mapping operations.
    """

    def __init__(self, data):
        """
        Initialize the ImageLoader with a dataset.

        Args:
            data (pd.DataFrame): DataFrame containing image IDs and labels
        """
        self.data = data

    def load_images_path(self, image_directory: str) -> List[List[str]]:
        """
        Get full paths for all images in the dataset.

        Args:
            image_directory (str): Base directory containing the images

        Returns:
            List[str]: List of complete image file paths
        """
        return [
            glob.glob(f"{image_directory}/{path}.jpg") for path in self.data["image_id"]
        ]

    def map_labels(self, label_mapping: Dict[str, int]) -> List[int]:
        """
        Map text labels to numerical values using a dictionary.

        Args:
            label_mapping (Dict[str, int]): Dictionary mapping label text to numbers

        Returns:
            List[int]: List of mapped numerical labels
        """
        return self.data["dx"].map(label_mapping)
