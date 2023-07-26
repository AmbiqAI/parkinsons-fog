import os
import math
import random
import logging
import functools
from typing import Generator

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from rich.logging import RichHandler

def set_random_seed(seed: int | None = None) -> int:
    """Set random seed across libraries: TF, Numpy, Python

    Args:
        seed (int | None, optional): Random seed state to use. Defaults to None.

    Returns:
        int: Random seed
    """
    seed = seed or np.random.randint(2**16)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
    except ImportError:
        pass
    else:
        tf.random.set_seed(seed)
    return seed


def setup_logger(log_name: str) -> logging.Logger:
    """Setup logger with Rich

    Args:
        log_name (str): _description_

    Returns:
        logging.Logger: _description_
    """
    logger = logging.getLogger(log_name)
    if logger.handlers:
        return logger
    logging.basicConfig(level=logging.ERROR, force=True, handlers=[RichHandler()])
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.handlers = [RichHandler()]
    return logger


def env_flag(env_var: str, default: bool = False) -> bool:
    """Return the specified environment variable coerced to a bool, as follows:
    - When the variable is unset, or set to the empty string, return `default`.
    - When the variable is set to a truthy value, returns `True`.
      These are the truthy values:
          - 1
          - true, yes, on
    - When the variable is set to anything else, returns False.
       Example falsy values:
          - 0
          - no
    - Ignore case and leading/trailing whitespace.
    """
    environ_string = os.environ.get(env_var, "").strip().lower()
    if not environ_string:
        return default
    return environ_string in ["1", "true", "yes", "on"]


def numpy_dataset_generator(x: npt.NDArray, y: npt.NDArray) -> Generator[tuple[npt.NDArray, npt.NDArray], None, None]:
    """Create generator from numpy dataset where first axis is samples

    Args:
        x (npt.NDArray): X data
        y (npt.NDArray): Y data

    Yields:
        Generator[tuple[npt.NDArray, npt.NDArray], None, None]: Samples
    """
    for i in range(x.shape[0]):
        yield x[i], y[i]


def create_dataset_from_data(x: npt.NDArray, y: npt.NDArray, spec: tuple[tf.TensorSpec]) -> tf.data.Dataset:
    """Helper function to create dataset from static data
    Args:
        x (npt.NDArray): Numpy data
        y (npt.NDArray): Numpy labels
    Returns:
        tf.data.Dataset: Dataset
    """
    gen = functools.partial(numpy_dataset_generator, x=x, y=y)
    dataset = tf.data.Dataset.from_generator(generator=gen, output_signature=spec)
    return dataset
