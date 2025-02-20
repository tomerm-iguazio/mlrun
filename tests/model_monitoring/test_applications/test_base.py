# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterator
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union
from unittest.mock import Mock, patch

import pytest

import mlrun
from mlrun.common.schemas.model_monitoring import ResultKindApp, ResultStatusApp
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationBase,
    ModelMonitoringApplicationMetric,
    ModelMonitoringApplicationResult,
    MonitoringApplicationContext,
)


class NoOpApp(ModelMonitoringApplicationBase):
    def do_tracking(self, monitoring_context: MonitoringApplicationContext):
        pass


class InProgressApp0(ModelMonitoringApplicationBase):
    def do_tracking(
        self, monitoring_context: MonitoringApplicationContext
    ) -> ModelMonitoringApplicationResult:
        monitoring_context.logger.info(
            "This test app is failing on purpose - ignore the failure!",
            project=monitoring_context.project_name,
        )
        raise ValueError


class InProgressApp1(ModelMonitoringApplicationBase):
    def do_tracking(
        self, monitoring_context: MonitoringApplicationContext
    ) -> ModelMonitoringApplicationResult:
        monitoring_context.logger.info(
            "It should work now",
            project=monitoring_context.project_name,
        )
        return ModelMonitoringApplicationResult(
            name="res0",
            value=0,
            status=ResultStatusApp.irrelevant,
            kind=ResultKindApp.mm_app_anomaly,
        )


@pytest.mark.filterwarnings("error")
def test_no_deprecation_instantiation() -> None:
    NoOpApp()


class TestEvaluate:
    @staticmethod
    @pytest.fixture(autouse=True)
    def _set_project() -> Iterator[None]:
        project = mlrun.get_or_create_project("test")
        with patch("mlrun.db.nopdb.NopDB.get_project", Mock(return_value=project)):
            yield

    @staticmethod
    def test_local_no_params() -> None:
        func_name = "test-app"
        run = InProgressApp0.evaluate(func_path=__file__, func_name=func_name)
        assert run.state() == "created"  # Should be "error", see ML-8507
        run = InProgressApp1.evaluate(func_path=__file__, func_name=func_name)
        assert run.state() == "completed"
        assert run.status.results == {
            "return": {
                "result_name": "res0",
                "result_value": 0.0,
                "result_kind": 4,
                "result_status": -1,
                "result_extra_data": "{}",
            }
        }, "The run results are different than expected"


@pytest.mark.parametrize(
    ("start", "end", "base_period", "expectation"),
    [
        (None, None, None, does_not_raise()),
        (
            datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc).isoformat(),
            datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc).isoformat(),
            None,
            does_not_raise(),
        ),
        (
            datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc).isoformat(),
            datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc).isoformat(),
            0,
            pytest.raises(
                mlrun.errors.MLRunValueError,
                match="`base_period` must be a nonnegative integer .*",
            ),
        ),
    ],
)
def test_window_generator_validation(
    start: Optional[str],
    end: Optional[str],
    base_period: Optional[int],
    expectation: AbstractContextManager,
) -> None:
    with expectation:
        next(ModelMonitoringApplicationBase._window_generator(start, end, base_period))


@pytest.mark.parametrize(
    ("start", "end", "base_period", "expected_windows"),
    [
        (
            datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc),
            datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc),
            None,
            [
                (
                    datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc),
                    datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc),
                ),
            ],
        ),
        (
            datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc),
            datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc),
            600,
            [
                (
                    datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc),
                    datetime(2008, 9, 1, 20, 2, 1, tzinfo=timezone.utc),
                ),
                (
                    datetime(2008, 9, 1, 20, 2, 1, tzinfo=timezone.utc),
                    datetime(2008, 9, 2, 6, 2, 1, tzinfo=timezone.utc),
                ),
                (
                    datetime(2008, 9, 2, 6, 2, 1, tzinfo=timezone.utc),
                    datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc),
                ),
            ],
        ),
        (
            datetime(2024, 12, 26, 14, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 12, 26, 14, 4, 0, tzinfo=timezone.utc),
            1,
            [
                (
                    datetime(2024, 12, 26, 14, 0, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 26, 14, 1, 0, tzinfo=timezone.utc),
                ),
                (
                    datetime(2024, 12, 26, 14, 1, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 26, 14, 2, 0, tzinfo=timezone.utc),
                ),
                (
                    datetime(2024, 12, 26, 14, 2, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 26, 14, 3, 0, tzinfo=timezone.utc),
                ),
                (
                    datetime(2024, 12, 26, 14, 3, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 26, 14, 4, 0, tzinfo=timezone.utc),
                ),
            ],
        ),
    ],
)
def test_windows(
    start: datetime,
    end: datetime,
    base_period: Optional[int],
    expected_windows: list[tuple[datetime, datetime]],
) -> None:
    assert (
        list(
            ModelMonitoringApplicationBase._window_generator(
                start=start.isoformat(), end=end.isoformat(), base_period=base_period
            )
        )
        == expected_windows
    ), "The generated windows are different than expected"


def test_job_handler() -> None:
    assert (
        ModelMonitoringApplicationBase.get_job_handler(
            "package.subpackage.module.AppClass"
        )
        == "package.subpackage.module.AppClass::_handler"
    )


@pytest.mark.parametrize(
    ("result", "expected_flattened_result"),
    [
        (
            ModelMonitoringApplicationMetric(name="m1", value=98),
            {"metric_name": "m1", "metric_value": 98},
        ),
        (
            [
                ModelMonitoringApplicationMetric(name="m0", value=-2),
                ModelMonitoringApplicationResult(
                    name="r0",
                    value=0,
                    status=ResultStatusApp.no_detection,
                    kind=ResultKindApp.mm_app_anomaly,
                ),
            ],
            [
                {"metric_name": "m0", "metric_value": -2},
                {
                    "result_name": "r0",
                    "result_value": 0,
                    "result_status": 0,
                    "result_kind": 4,
                    "result_extra_data": "{}",
                },
            ],
        ),
    ],
)
def test_flatten_data_result(
    result: Union[
        ModelMonitoringApplicationMetric,
        ModelMonitoringApplicationResult,
        list[Union[ModelMonitoringApplicationMetric, ModelMonitoringApplicationResult]],
    ],
    expected_flattened_result: Union[dict, list[dict]],
) -> None:
    assert (
        ModelMonitoringApplicationBase._flatten_data_result(result)
        == expected_flattened_result
    ), "The flattened result is different than expected"


class TestToJob:
    @staticmethod
    @pytest.fixture
    def project(tmpdir: Path) -> mlrun.projects.MlrunProject:
        return mlrun.get_or_create_project("test-to-job", context=str(tmpdir))

    @staticmethod
    @pytest.fixture
    def _set_project(project: mlrun.projects.MlrunProject) -> Iterator[None]:
        with patch("mlrun.db.nopdb.NopDB.get_project", Mock(return_value=project)):
            yield

    @staticmethod
    def test_base_is_blocked(project: mlrun.projects.MlrunProject) -> None:
        with pytest.raises(
            ValueError,
            match="You must provide a handler to the model monitoring application class",
        ):
            ModelMonitoringApplicationBase.to_job(project=project)

    @staticmethod
    @pytest.mark.usefixtures("_set_project")
    def test_with_class_handler(project: mlrun.projects.MlrunProject) -> None:
        job = ModelMonitoringApplicationBase.to_job(
            func_path=__file__,
            class_handler="NoOpApp",
            project=project,
        )
        assert isinstance(job, mlrun.runtimes.KubejobRuntime)
        run = job.run(local=True)
        assert run.state() == "completed"


@pytest.mark.parametrize(
    "endpoints", ["model-ep-1", ["model-ep-1"], [("model-ep-1", "model-ep-1-uid")]]
)
def test_handle_endpoints_type_evaluate(
    rundb_mock, endpoints: Union[str, list[str], list[tuple]]
) -> None:
    project = "test-endpoints-handler"
    endpoints_output = ModelMonitoringApplicationBase._handle_endpoints_type_evaluate(
        project, endpoints
    )

    assert endpoints_output == [("model-ep-1", "model-ep-1-uid")]
