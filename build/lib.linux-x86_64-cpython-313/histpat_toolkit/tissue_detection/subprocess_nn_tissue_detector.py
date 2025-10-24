import os
import sys
import json
import subprocess
import io
import numpy as np

from .base_tissue_detector import BaseTissueDetector


class SubprocessNNTissueDetector(BaseTissueDetector):
    """
    A subprocess-based wrapper for NNTissueDetector that runs all OpenVINO-related
    operations in a separate subprocess to isolate the model inference.
    """
    
    def __init__(
        self,
        tile_size=512,
        binary=True,
        **kwargs,
    ) -> None:
        """
        This is a class for tissue detection using neural networks in a subprocess.
        
        Parameters:
            tile_size (int): size of the tiles to be passed to the model
                (default 512).
            binary (bool): whether to return binary mask or multiclass mask
            **kwargs: parameters for the BaseTissueDetector class
        """
        super().__init__(**kwargs)
        self.binary = binary
        self.tile_size = tile_size
        self.mpp = 10  # this is the scale of the images passed to the model while learning
        
        # Store parameters for subprocess
        self._detector_params = {
            "tile_size": tile_size,
            "binary": binary,
            "opening_ksize": self.opening_ksize,
            "closing_ksize": self.closing_ksize,
            "mpp": self.mpp,
        }
        
        # Get path to worker script
        self._worker_script = os.path.join(
            os.path.dirname(__file__),
            "nn_tissue_detector_worker.py"
        )
        
    def _detect_tissue(
        self,
        img: np.ndarray[np.uint8],
    ) -> np.ndarray[np.uint8]:
        """
        Detect tissue in the image using a subprocess.
        
        Parameters:
            img (np.ndarray): image to detect tissue on
            
        Returns:
            np.ndarray: mask of the tissue in the image (1 - tissue, 0 - background)
        """
        if img.shape[2] == 4:
            img = img[:, :, :3]  # remove alpha channel
        
        # Serialize image to numpy format
        img_buffer = io.BytesIO()
        np.save(img_buffer, img, allow_pickle=False)
        img_bytes = img_buffer.getvalue()
        
        # Prepare input data
        input_data = {
            "params": self._detector_params,
            "image_hex": img_bytes.hex()
        }
        
        input_json = json.dumps(input_data)
        
        # Run subprocess
        try:
            result = subprocess.run(
                [sys.executable, self._worker_script],
                input=input_json,
                capture_output=True,
                text=True,
                check=True,
                cwd=os.path.dirname(self._worker_script)
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Tissue detection subprocess failed:\n"
                f"Return code: {e.returncode}\n"
                f"Stdout: {e.stdout}\n"
                f"Stderr: {e.stderr}"
            ) from e
        
        # Parse output
        try:
            output_data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse subprocess output:\n"
                f"Stdout: {result.stdout}\n"
                f"Stderr: {result.stderr}"
            ) from e
        
        # Deserialize mask
        mask_bytes = bytes.fromhex(output_data["mask_hex"])
        mask_buffer = io.BytesIO(mask_bytes)
        mask = np.load(mask_buffer, allow_pickle=False)
        
        return mask.astype(np.uint8)
