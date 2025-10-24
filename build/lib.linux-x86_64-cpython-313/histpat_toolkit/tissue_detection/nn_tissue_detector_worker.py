#!/usr/bin/env python3
"""
Worker script for running NNTissueDetector in a subprocess.
This script receives parameters and image data via stdin and outputs
the tissue detection mask via stdout.
"""

import sys
import json
import io
import numpy as np
from nn_tissue_detector import NNTissueDetector


def main():
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    
    # Extract parameters
    params = input_data.get("params", {})
    
    # Decode image from base64 numpy array
    img_bytes = bytes.fromhex(input_data["image_hex"])
    img_buffer = io.BytesIO(img_bytes)
    img = np.load(img_buffer, allow_pickle=False)
    
    # Create detector instance
    detector = NNTissueDetector(**params)
    
    # Run tissue detection
    mask = detector._detect_tissue(img)
    
    # Serialize mask to output
    output_buffer = io.BytesIO()
    np.save(output_buffer, mask, allow_pickle=False)
    mask_bytes = output_buffer.getvalue()
    
    # Output result as JSON with hex-encoded numpy array
    result = {
        "mask_hex": mask_bytes.hex(),
        "shape": mask.shape,
        "dtype": str(mask.dtype)
    }
    
    print(json.dumps(result))


if __name__ == "__main__":
    main()
