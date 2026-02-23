#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from numpy.typing import NDArray
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image

from lightly_train._configs.validate import no_auto
from lightly_train._transforms.channel_drop import ChannelDrop
from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionTransformArgs,
)
from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformInput,
    TaskTransformOutput,
)
from lightly_train.types import NDArrayImage, NDArrayOBBoxes


class OrientedObjectDetectionTransformInput(TaskTransformInput):
    image: NDArrayImage
    bboxes: NDArrayOBBoxes
    class_labels: NDArray


class OrientedObjectDetectionTransformOutput(TaskTransformOutput):
    image: torch.Tensor
    bboxes: torch.Tensor
    class_labels: torch.Tensor


class OrientedObjectDetectionTransformArgs(ObjectDetectionTransformArgs):
    pass


def _numpy_image_to_tv_tensor(image_hwc: NDArrayImage) -> Image:
    image_chw = image_hwc.transpose(2, 0, 1)
    return Image(image_chw)


def _numpy_obb_to_tv_tensor_obb(
    oriented_bboxes: NDArrayOBBoxes, canvas_size: tuple[int, int]
) -> BoundingBoxes:
    return BoundingBoxes(
        oriented_bboxes.astype(np.float64),  # type: ignore[arg-type]
        format=BoundingBoxFormat.CXCYWHR,
        canvas_size=canvas_size,
    )


def _tv_tensor_to_tensor(
    image: Image, bboxes: BoundingBoxes
) -> tuple[torch.Tensor, torch.Tensor]:
    image_out = torch.asarray(image)
    bboxes_out = torch.asarray(bboxes)
    return image_out, bboxes_out


class OrientedObjectDetectionTransform(TaskTransform):
    transform_args_cls: type[OrientedObjectDetectionTransformArgs] = (
        OrientedObjectDetectionTransformArgs
    )

    def __init__(
        self,
        transform_args: OrientedObjectDetectionTransformArgs,
    ) -> None:
        super().__init__(transform_args=transform_args)

        self.transform_args: OrientedObjectDetectionTransformArgs = transform_args
        self.stop_step = (
            transform_args.stop_policy.stop_step if transform_args.stop_policy else None
        )

        if self.stop_step is not None:
            raise NotImplementedError(
                "Stopping certain augmentations after some steps is not implemented yet."
            )
        self.global_step = 0
        self.stop_ops = (
            transform_args.stop_policy.ops if transform_args.stop_policy else set()
        )
        self.past_stop = False

        self.transform = self._build_transform()

    def _build_transform(self) -> v2.Compose:
        transform_args = self.transform_args
        transforms_list: list[v2.Transform] = []

        if transform_args.channel_drop is not None:
            channel_drop_args = transform_args.channel_drop
            transforms_list.append(
                v2.Lambda(
                    lambda img: ChannelDrop(
                        num_channels_keep=channel_drop_args.num_channels_keep,
                        weight_drop=channel_drop_args.weight_drop,
                    ).apply(img)
                )
            )

        if transform_args.photometric_distort is not None:
            transforms_list.append(
                v2.RandomPhotometricDistort(
                    brightness=transform_args.photometric_distort.brightness,
                    contrast=transform_args.photometric_distort.contrast,
                    saturation=transform_args.photometric_distort.saturation,
                    hue=transform_args.photometric_distort.hue,
                    p=transform_args.photometric_distort.prob,
                )
            )

        if transform_args.random_zoom_out is not None:
            zoom_out = transform_args.random_zoom_out
            side_range_min, side_range_max = zoom_out.side_range

            class RandomZoomOutTorch(v2.Transform):
                def __init__(
                    self, fill: float, side_range: tuple[float, float], p: float
                ):
                    super().__init__()
                    self.fill = fill
                    self.side_range = side_range
                    self.p = p

                def forward(self, *inputs: Any) -> Any:
                    if torch.rand(1).item() > self.p:
                        return inputs

                    image = inputs[0]
                    if len(inputs) > 1:
                        bboxes = inputs[1]
                        labels = inputs[2] if len(inputs) > 2 else None
                    else:
                        bboxes = None
                        labels = None

                    if not isinstance(image, Image):
                        image = Image(image)

                    _, orig_h, orig_w = image.shape[-3:]

                    scale = (
                        torch.empty(1)
                        .uniform_(self.side_range[0], self.side_range[1])
                        .item()
                    )
                    new_h = int(orig_h * scale)
                    new_w = int(orig_w * scale)

                    if new_h > orig_h or new_w > orig_w:
                        pad_h = max(0, new_h - orig_h)
                        pad_w = max(0, new_w - orig_w)

                        if isinstance(self.fill, (int, float)):
                            fill_value = [float(self.fill)] * 3  # type: ignore[list-item]
                        else:
                            fill_value = self.fill

                        image = v2.functional.pad(
                            image,
                            padding=[0, 0, pad_w, pad_h],
                            fill=fill_value,
                            padding_mode="constant",
                        )

                        if bboxes is not None:
                            bboxes = BoundingBoxes(
                                np.asarray(bboxes).astype(np.float64),  # type: ignore[arg-type]
                                format=BoundingBoxFormat.CXCYWHR,
                                canvas_size=(new_h, new_w),
                            )

                    if bboxes is not None:
                        if labels is not None:
                            return image, bboxes, labels
                        return image, bboxes
                    return image

            transforms_list.append(
                RandomZoomOutTorch(
                    fill=zoom_out.fill,
                    side_range=zoom_out.side_range,
                    p=zoom_out.prob,
                )
            )

        if transform_args.random_iou_crop is not None:
            raise NotImplementedError(
                "RandomIoUCrop is not implemented yet for OrientedObjectDetectionTransform."
                "torchvision does not support it for now."
            )

        if transform_args.random_flip is not None:
            if transform_args.random_flip.horizontal_prob > 0.0:
                transforms_list.append(
                    v2.RandomHorizontalFlip(
                        p=transform_args.random_flip.horizontal_prob
                    )
                )
            if transform_args.random_flip.vertical_prob > 0.0:
                transforms_list.append(
                    v2.RandomVerticalFlip(p=transform_args.random_flip.vertical_prob)
                )

        if transform_args.random_rotate_90 is not None:
            transforms_list.append(v2.RandomRotation(degrees=[0, 90, 180, 270]))

        if transform_args.random_rotate is not None:
            degrees = transform_args.random_rotate.degrees
            transforms_list.append(
                v2.RandomRotation(
                    degrees=tuple(degrees) if isinstance(degrees, float) else degrees,  # type: ignore[arg-type]
                )
            )

        if transform_args.resize is not None:
            transforms_list.append(
                v2.Resize(
                    size=(
                        no_auto(transform_args.resize.height),
                        no_auto(transform_args.resize.width),
                    )
                )
            )

        transforms_list.append(
            v2.ToDtype(dtype=torch.float32, scale=True),
        )

        if transform_args.normalize is not None:
            norm_args = no_auto(transform_args.normalize)
            transforms_list.append(
                v2.Normalize(
                    mean=norm_args.mean,
                    std=norm_args.std,
                )
            )

        return v2.Compose(transforms_list)

    def __call__(
        self, input: OrientedObjectDetectionTransformInput
    ) -> OrientedObjectDetectionTransformOutput:
        if (
            self.stop_step is not None
            and self.global_step >= self.stop_step
            and not self.past_stop
        ):
            self.transform = self._build_transform()
            self.past_stop = True

        image_hwc = input["image"]
        oriented_bboxes = input["bboxes"]
        class_labels = input["class_labels"]

        tv_image = _numpy_image_to_tv_tensor(image_hwc)
        canvas_size = (image_hwc.shape[0], image_hwc.shape[1])
        tv_bboxes = _numpy_obb_to_tv_tensor_obb(oriented_bboxes, canvas_size)

        labels_tensor = torch.asarray(class_labels, dtype=torch.int64)

        tv_image_out, tv_bboxes_out = self.transform(tv_image, tv_bboxes)

        image_out, bboxes_out = _tv_tensor_to_tensor(tv_image_out, tv_bboxes_out)

        return {
            "image": image_out,
            "bboxes": bboxes_out,
            "class_labels": labels_tensor,
        }
