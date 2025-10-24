from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Iterator, Protocol

import numpy as np

from .modal_client import RunMode


class SegmentationModelRunner(Protocol):
    """
    Protocol (interface) for segmentation model inference.
    This allows different implementations (Modal, local inference, etc.)
    """

    def invoke(
        self,
        input_gen: Iterable[np.ndarray],
        run_mode: RunMode,
        batch_size: int = 1,
        executor: ThreadPoolExecutor | None = None,
        logger=None,
    ) -> Iterator[np.ndarray | list[np.ndarray]]:
        """
        Invoke the segmentation model on a batch of images.
        
        Args:
            input_gen: Iterator of input images as numpy arrays
            run_mode: Prediction mode (PREDICTION, PROBABILITIES, or PREDICTION_MAX)
            batch_size: Number of images to process in a single batch
            executor: Optional ThreadPoolExecutor for parallel processing
            logger: Optional logger for logging
            
        Returns:
            Iterator of segmentation results (either single masks or list of probability maps)
        """
        ...

    def get_config(self):
        """Get the model configuration (including input size requirements, mpp, etc.)"""
        ...

