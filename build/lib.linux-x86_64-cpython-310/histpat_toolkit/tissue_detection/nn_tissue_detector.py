import os

import numpy as np
from openvino import Core

from .base_tissue_detector import BaseTissueDetector


class NNTissueDetector(BaseTissueDetector):
    def __init__(
        self,
        tile_size=512,
        binary=True,
        **kwargs,
    ) -> None:
        """
        This is a class for tissue detection using neural networks

        Parameters:
            tile_size (int): size of the tiles to be passed to the model
                (default 512).
            **kwargs: parameters for the BaseTissueDetector class
        """
        super().__init__(**kwargs)
        self.binary = binary
        self.tile_size = tile_size
        self.mpp = 10  # this is the scale of the images passed to the model while learning

        # We are loading resnet18 model trained for tissues detection on scale 10mpp
        dirname = os.path.dirname(__file__)
        model_filename = os.path.join(
            dirname,
            "nn_models/resnet18_10mpp_16bit",
        )
        ie = Core()
        model = ie.read_model(
            model=model_filename + ".xml",
            weights=model_filename + ".bin",
        )
        self.compiled_model = ie.compile_model(model=model, device_name="CPU")

        self.input_layer_ir = self.compiled_model.input(0)
        self.output_layer_ir = self.compiled_model.output(0)

        self.tile_size = tile_size

    def postprocess(
        self,
        arr: np.ndarray[np.uint8],
    ) -> np.ndarray[np.uint8]:
        """
        This function postprocesses the output of the model
        to get the binary mask of the tissue in the image

        Parameters:
            arr (np.ndarray): output of the model in the form
            of 3d array (Channels, Height, Width)
        """
        pixel_type = np.argmax(arr, axis=0).astype(np.uint8)
        pixel_type -= 1

        if self.binary:
            ret = np.zeros(pixel_type.shape).astype(np.uint8)
            ret[pixel_type == 1] = 1  # tissue
            ret[pixel_type == 2] = 1  # connective tissue
            return ret
        return pixel_type

    def _detect_tissue(
        self,
        img: np.ndarray[np.uint8],
    ) -> np.ndarray[np.uint8]:
        if img.shape[2] == 4:
            img = img[:, :, :3]  # remove alpha channel

        dataset = ImageDataset(
            img,
            self.tile_size,
        )

        ret_mask = np.zeros((dataset.H * self.tile_size, dataset.W * self.tile_size)).astype(np.uint8)

        for tile_id in range(len(dataset)):
            arr, (x, y) = dataset[tile_id]
            # this is still just a np.ndarray
            x_tensor = np.expand_dims(arr, axis=0)
            input_data = {"x.1": x_tensor}
            pr_mask = self.compiled_model(input_data)[self.output_layer_ir][0]

            ret_mask[
                y * self.tile_size : (y + 1) * self.tile_size,
                x * self.tile_size : (x + 1) * self.tile_size,
            ] = self.postprocess(pr_mask)

        if self.binary:
            # we still want to smooth the mask in order to remove small holes and islands
            # this helps during the process of patch extraction
            return self.smooth_mask(
                ret_mask[: img.shape[0], : img.shape[1]],
                opening_ksize=self.opening_ksize,
                closing_ksize=self.closing_ksize,
            )
        return ret_mask[: img.shape[0], : img.shape[1]]


def div_up(a, b):
    return (a + b - 1) // b


class ImageDataset:

    def __init__(
        self,
        np_image: np.ndarray[np.uint8],
        tile_size: int,
    ) -> None:
        self.np_image = np_image
        self.tile_size = tile_size
        self.augmentation = lambda x: np.pad(
            x,
            (
                (0, tile_size - x.shape[0]),
                (0, tile_size - x.shape[1]),
                (0, 0),
            ),
            constant_values=255,
        )

        self.preprocessing = lambda x: x.transpose(2, 0, 1).astype("float32")

        self.w = np_image.shape[1]
        self.h = np_image.shape[0]
        self.W = div_up(self.w, tile_size)
        self.H = div_up(self.h, tile_size)

    def __len__(self):
        return int(self.W * self.H)

    def __getitem__(self, idx):
        x, y = idx % self.W, idx // self.W

        ts = self.tile_size
        arr = self.np_image[ts * y : min(self.h, ts * (y + 1)), ts * x : min(self.w, ts * (x + 1)), :]

        arr = self.augmentation(arr)
        arr = self.preprocessing(arr)

        return arr, (x, y)
