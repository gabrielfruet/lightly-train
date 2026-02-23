#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image
from numpy.typing import NDArray

from lightly_train._transforms.scale_jitter import generate_discrete_sizes
from lightly_train.types import NDArrayImage, NDArrayOBBoxes


def numpy_image_to_tv_tensor_image(image_hwc: NDArrayImage) -> Image:
    """Convert a numpy image array to a torchvision tv_tensor Image.

    Args:
        image_hwc: A numpy array of shape (H, W, C) containing the image data.

    Returns:
        A torchvision tv_tensor Image containing the image data.
    """
    image_chw = image_hwc.transpose(2, 0, 1)
    return Image(image_chw)


def numpy_obb_to_tv_tensor_obb(
    oriented_bboxes: NDArrayOBBoxes, canvas_size: tuple[int, int]
) -> BoundingBoxes:
    """Convert oriented bounding boxes to a torchvision tv_tensor BoundingBoxes.

    Args:
        oriented_bboxes: A numpy array of shape (n_boxes, 5) containing the
            oriented bounding boxes in (x_center, y_center, width, height, angle) format.
        canvas_size: A tuple (height, width) representing the canvas size.

    Returns:
        A torchvision tv_tensor BoundingBoxes object in CXCYWHR format.
    """
    return BoundingBoxes(
        oriented_bboxes.astype(np.float64),  # type: ignore[arg-type]
        format=BoundingBoxFormat.CXCYWHR,
        canvas_size=canvas_size,
    )


def image_hwc_height_width(image: NDArrayImage) -> tuple[int, int]:
    """Get the height and width of an image from its shape.

    Args:
        image: A numpy array of shape (H, W, C) containing the image data.

    Returns:
        A tuple (height, width) containing the height and width of the image.
    """
    return image.shape[0], image.shape[1]


def convert_numpy_to_torchvision_input(
    image_hwc: NDArrayImage,
    oriented_bboxes: NDArrayOBBoxes | None,
) -> tuple[Image, BoundingBoxes | None]:
    """Convert a numpy image and oriented bounding boxes to torchvision tv_tensors.

    Args:
        image_hwc: A numpy array of shape (H, W, C) containing the image data.
        oriented_bboxes: A numpy array of shape (n_boxes, 5) containing the oriented
            bounding boxes in (x_center, y_center, width, height, angle) format,
            or None if there are no bounding boxes.

    Returns:
        A tuple containing the tv_tensors for Image and the BoundingBoxes if non-null.
    """
    tv_image = numpy_image_to_tv_tensor_image(image_hwc)
    tv_bboxes = (
        numpy_obb_to_tv_tensor_obb(
            oriented_bboxes=oriented_bboxes,
            canvas_size=image_hwc_height_width(image_hwc),
        )
        if oriented_bboxes is not None
        else None
    )
    return tv_image, tv_bboxes


def convert_torchvision_to_numpy_output(
    tv_image: Image,
    tv_bboxes: BoundingBoxes | None,
) -> tuple[NDArrayImage, NDArrayOBBoxes | None]:
    """Convert torchvision tv_tensors back to numpy arrays.

    Args:
        tv_image: A torchvision tv_tensor Image.
        tv_bboxes: A torchvision tv_tensor BoundingBoxes, or None.

    Returns:
        A tuple containing the numpy image (HWC format) and bounding boxes.
    """
    image_chw = tv_image.cpu().numpy()
    image_hwc = image_chw.transpose(1, 2, 0)
    bboxes = tv_bboxes.cpu().numpy() if tv_bboxes is not None else None
    return image_hwc, bboxes


class SeededRandomChoice(v2.Transform):
    """A transform that randomly selects one of the given transforms using a seeded random choice.

    This is useful for deterministic random selection during inference or validation.
    """

    def __init__(self, transforms: Sequence[v2.Transform], seed: int = 42) -> None:
        super().__init__()
        self.transforms = transforms
        self.generator = torch.Generator().manual_seed(seed)
        self._current_idx = self._generate_idx()

    def _generate_idx(self) -> int:
        return int(
            torch.randint(
                0,
                len(self.transforms),
                (1,),
                generator=self.generator,
            ).item()
        )

    def step(self) -> None:
        """Advance to the next random transform."""
        self._current_idx = self._generate_idx()

    def forward(self, *inputs: tuple) -> tuple:
        return self.transforms[self._current_idx](*inputs)


class TorchVisionScaleJitter(v2.Transform):
    """Scale jitter transform using torchvision v2.

    This is a pure torchvision transform that randomly selects from a set of discrete
    sizes for resizing the image. Unlike the albumentations-based version, this works
    directly with torchvision TVTensors.

    Args:
        sizes: Optional sequence of (height, width) tuples to choose from.
        target_size: Target (height, width) when sizes is not provided.
        scale_range: Range of scale factors (min, max) when sizes is not provided.
        num_scales: Number of discrete scales to generate when sizes is not provided.
        divisible_by: If provided, sizes will be adjusted to be divisible by this value.
        seed: Random seed for deterministic transform selection.
    """

    def __init__(
        self,
        *,
        sizes: Sequence[tuple[int, int]] | None = None,
        target_size: tuple[int, int] | None = None,
        scale_range: tuple[float, float] | None = None,
        num_scales: int | None = None,
        divisible_by: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.heights, self.widths = zip(
            *generate_discrete_sizes(
                sizes=sizes,
                target_size=target_size,
                scale_range=scale_range,
                num_scales=num_scales,
                divisible_by=divisible_by,
            )
        )

        transforms = [
            v2.Resize(size=(int(h), int(w)), antialias=True)
            for h, w in zip(self.heights, self.widths)
        ]

        self._transform = SeededRandomChoice(transforms, seed=seed)

    @contextmanager
    def same_seed(self):
        """Context manager to step through transforms while keeping the same seed."""
        self._transform.step()
        try:
            yield self
        finally:
            self._transform.step()

    def forward(self, *inputs: tuple) -> tuple:
        return self._transform(*inputs)
