#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import pytest
from pytest import LogCaptureFixture

from lightly_train._commands.train_task_helpers import (
    BestAggregatedMetricValues,
    get_best_metrics,
    get_train_model_args,
    get_train_model_cls,
    pretty_format_args_dict,
)
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._metrics.task_metric import AggregatedMetricValues, TaskMetricArgs
from lightly_train._task_models.dinov2_ltdetr_object_detection.train_model import (
    DINOv2LTDETRObjectDetectionTrain,
    DINOv2LTDETRObjectDetectionTrainArgs,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.train_model import (
    DINOv3LTDETRObjectDetectionTrain,
    DINOv3LTDETRObjectDetectionTrainArgs,
)


def test_get_best_metrics__no_previous_best() -> None:
    last = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.8},
        watch_metric="val_metric/acc",
        watch_metric_value=0.8,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    result = get_best_metrics(
        best_agg_metric_values=None,
        last_agg_metric_values=last,
        step=0,
        metric_args=TaskMetricArgs(watch_metric="val_metric/acc"),
    )
    assert result.agg_metric_values is last
    assert result.step == 0


def test_get_best_metrics__max_mode_improvement() -> None:
    prev = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.5},
        watch_metric="val_metric/acc",
        watch_metric_value=0.5,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.8},
        watch_metric="val_metric/acc",
        watch_metric_value=0.8,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    result = get_best_metrics(
        best_agg_metric_values=best,
        last_agg_metric_values=last,
        step=1,
        metric_args=TaskMetricArgs(watch_metric="val_metric/acc"),
    )
    assert result.agg_metric_values is last
    assert result.step == 1


def test_get_best_metrics__max_mode_no_improvement() -> None:
    prev = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.9},
        watch_metric="val_metric/acc",
        watch_metric_value=0.9,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.7},
        watch_metric="val_metric/acc",
        watch_metric_value=0.7,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    result = get_best_metrics(
        best_agg_metric_values=best,
        last_agg_metric_values=last,
        step=1,
        metric_args=TaskMetricArgs(watch_metric="val_metric/acc"),
    )
    assert result is best


def test_get_best_metrics__min_mode_improvement() -> None:
    prev = AggregatedMetricValues(
        metric_values={"val_loss": 0.8},
        watch_metric="val_loss",
        watch_metric_value=0.8,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_loss": 0.3},
        watch_metric="val_loss",
        watch_metric_value=0.3,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metric_values=None,
    )
    result = get_best_metrics(
        best_agg_metric_values=best,
        last_agg_metric_values=last,
        step=2,
        metric_args=TaskMetricArgs(watch_metric="val_loss"),
    )
    assert result.agg_metric_values is last
    assert result.step == 2


def test_get_best_metrics__min_mode_no_improvement() -> None:
    prev = AggregatedMetricValues(
        metric_values={"val_loss": 0.3},
        watch_metric="val_loss",
        watch_metric_value=0.3,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_loss": 0.9},
        watch_metric="val_loss",
        watch_metric_value=0.9,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metric_values=None,
    )
    result = get_best_metrics(
        best_agg_metric_values=best,
        last_agg_metric_values=last,
        step=2,
        metric_args=TaskMetricArgs(watch_metric="val_loss"),
    )
    assert result is best


def test_get_best_metrics__missing_watch_metric(caplog: LogCaptureFixture) -> None:
    # watch_metric configured but not present in computed metrics
    # last is returned as best since no valid best exists.
    prev = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.9},
        watch_metric=None,
        watch_metric_value=None,
        watch_metric_mode=None,
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.95},
        watch_metric=None,
        watch_metric_value=None,
        watch_metric_mode=None,
        best_head_name=None,
        best_head_metric_values=None,
    )
    with caplog.at_level("WARNING"):
        result = get_best_metrics(
            best_agg_metric_values=best,
            last_agg_metric_values=last,
            step=1,
            metric_args=TaskMetricArgs(watch_metric="val_metric/nonexistent"),
        )
    assert "Unknown watch metric" in caplog.text
    assert result.agg_metric_values is last
    assert result.step == 1


def test_get_best_metrics__different_watch_metric_raises() -> None:
    prev = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.9},
        watch_metric="val_metric/acc",
        watch_metric_value=0.9,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_loss": 0.1},
        watch_metric="val_loss",
        watch_metric_value=0.1,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metric_values=None,
    )

    with pytest.raises(
        RuntimeError,
        match="Best and last aggregated metrics use different watch metrics",
    ):
        get_best_metrics(
            best_agg_metric_values=best,
            last_agg_metric_values=last,
            step=1,
            metric_args=TaskMetricArgs(watch_metric="val_metric/acc"),
        )


@pytest.mark.parametrize(
    ("model_args_cls", "model_name"),
    [
        (DINOv2LTDETRObjectDetectionTrainArgs, "dinov2/_vittest14-ltdetr"),
        (DINOv3LTDETRObjectDetectionTrainArgs, "dinov3/_vittest16-ltdetr"),
    ],
)
def test_get_train_model_args__scheduler_roundtrip(
    model_args_cls: type[DINOv2LTDETRObjectDetectionTrainArgs]
    | type[DINOv3LTDETRObjectDetectionTrainArgs],
    model_name: str,
) -> None:
    data_args = YOLOObjectDetectionDataArgs(
        path=Path("/tmp/data"),
        train=Path("train") / "images",
        val=Path("val") / "images",
        names={0: "class_0", 1: "class_1"},
    )

    resolved = get_train_model_args(
        model_args={"scheduler": "flat-cosine", "lr_warmup_steps": 123},
        model_args_cls=model_args_cls,
        total_steps=1000,
        model_name=model_name,
        model_init_args={},
        data_args=data_args,
    )

    assert resolved.scheduler == "flat-cosine"
    assert resolved.lr_warmup_steps == 123
    assert pretty_format_args_dict(resolved.model_dump())["scheduler"] == "flat-cosine"


@pytest.mark.parametrize(
    ("model_name", "expected_cls"),
    [
        ("dinov2/_vittest14-ltdetr", DINOv2LTDETRObjectDetectionTrain),
        ("dinov3/_vittest16-ltdetr", DINOv3LTDETRObjectDetectionTrain),
    ],
)
def test_get_train_model_cls__ltdetr_dispatch(
    model_name: str,
    expected_cls: type[DINOv2LTDETRObjectDetectionTrain]
    | type[DINOv3LTDETRObjectDetectionTrain],
) -> None:
    assert (
        get_train_model_cls(model_name=model_name, task="object_detection")
        is expected_cls
    )


def test_get_train_model_cls__dsp_not_supported() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported model name 'dinov2/_vittest14-ltdetr-dsp'",
    ):
        get_train_model_cls(
            model_name="dinov2/_vittest14-ltdetr-dsp",
            task="object_detection",
        )
