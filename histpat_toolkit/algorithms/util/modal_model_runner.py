from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Iterator

import numpy as np

from .config import MicroscopyConfig
from .modal_client import MODAL_AVAILABLE, RunMode, get_model_client, invoke_segmentation_model


class ModalSegmentationModelRunner:
    """
    Implementation of SegmentationModelRunner that uses Modal for inference.
    
    Requires modal to be installed: pip install modal
    """

    def __init__(self, model_app: str, model_name: str):
        """
        Args:
            model_app: Modal app name
            model_name: Model version/name
        """
        if not MODAL_AVAILABLE:
            raise ImportError(
                "modal is required for ModalSegmentationModelRunner. "
                "Install it with: pip install modal"
            )
        
        self.model_app = model_app
        self.model_name = model_name
        self._client = None
        self._config = None

    @property
    def client(self):
        """Lazy initialization of the modal client"""
        if self._client is None:
            self._client = get_model_client(self.model_app, self.model_name)
        return self._client

    def invoke(
        self,
        input_gen: Iterable[np.ndarray],
        run_mode: RunMode,
        batch_size: int = 1,
        executor: ThreadPoolExecutor | None = None,
        logger=None,
    ) -> Iterator[np.ndarray | list[np.ndarray]]:
        """Invoke the segmentation model using Modal"""
        return invoke_segmentation_model(
            self.client,
            input_gen,
            run_mode,
            batch_size=batch_size,
            executor=executor,
            logger=logger,
        )

    def get_config(self):
        """Get the model configuration from Modal"""
        if self._config is None:
            config_dict = self.client.get_config.remote()
            self._config = MicroscopyConfig.model_validate(config_dict)
        return self._config

