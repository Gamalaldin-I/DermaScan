import yaml
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Union, Optional


def load_yaml_config(config_path: Union[str, Path]) -> dict:
    """
    Load and parse a YAML configuration file safely.

    Args:
        config_path (Union[str, Path]): Path to the YAML configuration file

    Returns:
        dict: Parsed YAML content as a dictionary

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML syntax is invalid
        PermissionError: If there are insufficient permissions to read the file
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"YAML file not found: {config_path}")

    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    except yaml.YAMLError as yaml_err:
        raise yaml.YAMLError(f"Invalid YAML syntax in {config_path}: {yaml_err}")

    except PermissionError:
        raise PermissionError(f"Insufficient permissions to read {config_path}")


def save_numpy_array(file_path: Union[str, Path], array: np.ndarray) -> Optional[str]:
    """
    Save a NumPy array to a file.

    Args:
        file_path (Union[str, Path]): Path where the array will be saved
        array (np.ndarray): NumPy array to save

    Returns:
        Optional[str]: Error message if save fails, None on success
    """
    try:
        np.save(file_path, array)
        return None

    except Exception as e:
        return f"Save failed: {str(e)}"


def load_numpy_array(file_path: Union[str, Path]) -> Union[np.ndarray, str]:
    """
    Load a NumPy array from a file.

    Args:
        file_path (Union[str, Path]): Path to the NumPy array file

    Returns:
        Union[np.ndarray, str]: Loaded NumPy array or error message if load fails
    """
    try:
        return np.load(file_path)

    except Exception as e:
        return f"Failed to load data: {str(e)}"


def export_tflite_model(keras_model: tf.keras.Model, output_path: str) -> str:
    """
    Exports a Keras model to an optimized TFLite format with FP16 quantization.

    Args:
        keras_model: The Keras model to convert
        output_path: Path where the TFLite model will be saved

    Returns:
        str: Path to the saved TFLite model

    Example:
        model_path = export_optimized_tflite(my_model, "model.tflite")
    """
    tflite_converter = tf.lite.TFLiteConverter.from_keras_model(keras_model.model)

    tflite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_converter.target_spec.supported_types = [tf.float16]

    optimized_model = tflite_converter.convert()

    with open(output_path, "wb") as model_file:
        model_file.write(optimized_model)

    return output_path
