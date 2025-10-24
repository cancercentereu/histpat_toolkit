import os
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from io import BytesIO
from itertools import islice
from typing import Iterable, Iterator, TypeVar

import numpy as np
from PIL import Image
from pydantic import BaseModel

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    modal = None

IS_AVAILABLE = MODAL_AVAILABLE and bool(os.getenv("MODAL_TOKEN_SECRET") and os.getenv("MODAL_TOKEN_ID"))

STUB_NAME = "models-patho"


class RunMode(Enum):
    PREDICTION = 1
    PROBABILITIES = 2
    PREDICTION_MAX = 3


class Nucleus(BaseModel):
    x: int
    y: int
    S: float  # surface area
    p: float  # probability of positive class in range (0, 1)


class NucleiModelInferenceResult(BaseModel):
    results: list[list[Nucleus]]
    # in the future we might want to add more fields here


class MILModelInferenceResult(BaseModel):
    prediction: float
    attention: list[float]
    distribution: dict[str, float] | None = None


T = TypeVar("T")


def iter_batched(iterable: Iterable[T], n) -> Iterator[list[T]]:
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def numpy_to_bytes(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr)
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()


def bytes_to_numpy(img_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(img_bytes))
    return np.array(img)


def get_model_client(class_name: str, model_name: str):
    if not MODAL_AVAILABLE:
        raise ImportError(
            "modal is required for get_model_client. "
            "Install it with: pip install modal"
        )
    Model = modal.Cls.lookup(STUB_NAME, class_name)
    return Model(model_name=model_name)


def invoke_segmentation_model(
    model_client,
    input_gen: Iterable[np.ndarray],
    run_mode: RunMode,
    batch_size: int = 1,
    executor: ThreadPoolExecutor | None = None,
    logger=None,
) -> Iterator[np.ndarray | list[np.ndarray]]:
    if logger:
        logger.info("Modal: Invoking segmentation model")

    def handle_batch(batch: list[np.ndarray]):
        images = [numpy_to_bytes(img) for img in batch]
        result = model_client.infer.remote(input_images=images, run_mode=run_mode.value)
        if run_mode == RunMode.PREDICTION:
            return [bytes_to_numpy(img_bytes) for img_bytes in result["predictions"]]
        elif run_mode == RunMode.PROBABILITIES:
            return [[bytes_to_numpy(img_bytes) for img_bytes in prob] for prob in result["probabilities"]]
        else:
            raise ValueError(f"Unsupported run mode: {run_mode}")

    if executor is None:
        for batch in iter_batched(input_gen, batch_size):
            yield from handle_batch(batch)
    else:
        futures = [executor.submit(handle_batch, batch) for batch in iter_batched(input_gen, batch_size)]
        for future in futures:
            yield from future.result()


def invoke_nuclei_detection_model(
    model_client,
    input_gen: Iterable[tuple[np.ndarray, np.ndarray]],
    batch_size: int = 1,
    executor: ThreadPoolExecutor | None = None,
    logger=None,
) -> Iterator[list[Nucleus]]:
    if logger:
        logger.info("Modal: Invoking model")

    def handle_batch(batch: list[tuple[np.ndarray, np.ndarray]]):
        images = [numpy_to_bytes(img) for img, mask in batch]
        mask_images = [numpy_to_bytes(mask) for img, mask in batch]
        result_obj = model_client.infer.remote(images=images, mask_images=mask_images)
        result = NucleiModelInferenceResult(**result_obj)
        return result.results

    if executor is None:
        for batch in iter_batched(input_gen, batch_size):
            yield from handle_batch(batch)
    else:
        futures = [executor.submit(handle_batch, batch) for batch in iter_batched(input_gen, batch_size)]
        for future in futures:
            yield from future.result()


def invoke_embedding_model(
    model_client,
    input_gen: Iterable[np.ndarray],
    batch_size: int = 1,
    executor: ThreadPoolExecutor | None = None,
    logger=None,
) -> Iterator[list[float]]:
    if logger:
        logger.info("Modal: Invoking embedding model")

    def handle_batch(batch: list[np.ndarray]):
        images = [numpy_to_bytes(img) for img in batch]
        result = model_client.infer.remote(input_images=images)

        return result["features"]

    if executor is None:
        for batch in iter_batched(input_gen, batch_size):
            yield from handle_batch(batch)
    else:
        futures = [executor.submit(handle_batch, batch) for batch in iter_batched(input_gen, batch_size)]
        for future in futures:
            yield from future.result()


def invoke_mil_model(
    model_client,
    input: list[list[float]],
    logger=None,
) -> MILModelInferenceResult:
    if logger:
        logger.info("Modal: Invoking MIL model")

    result = model_client.infer.remote(embeddings=input)
    return MILModelInferenceResult(
        prediction=result["prediction"],
        attention=result["attention"],
        distribution=result.get("distribution", None),
    )
