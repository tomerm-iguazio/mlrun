# Copyright 2023 Iguazio
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

import datetime
from collections.abc import Iterator
from typing import NamedTuple, Optional, Union
from unittest.mock import patch

import nuclio
import numpy as np
import pandas as pd
import pytest

import mlrun
from mlrun.common.model_monitoring.helpers import (
    _MAX_FLOAT,
    FeatureStats,
    Histogram,
    get_kafka_topic,
    pad_features_hist,
    pad_hist,
)
from mlrun.common.schemas import EndpointType, ModelEndpoint
from mlrun.common.schemas.model_monitoring.constants import EventFieldType
from mlrun.datastore import KafkaOutputStream, OutputStream
from mlrun.datastore.datastore_profile import (
    DatastoreProfile,
    DatastoreProfileKafkaSource,
    DatastoreProfileKafkaTarget,
    DatastoreProfileV3io,
)
from mlrun.db.nopdb import NopDB
from mlrun.model_monitoring.controller import (
    _BatchWindow,
    _BatchWindowGenerator,
    _Interval,
)
from mlrun.model_monitoring.db._schedules import ModelMonitoringSchedulesFile
from mlrun.model_monitoring.helpers import (
    _BatchDict,
    _get_monitoring_time_window_from_controller_run,
    batch_dict2timedelta,
    filter_results_by_regex,
    get_invocations_fqn,
    get_output_stream,
    update_model_endpoint_last_request,
)
from mlrun.utils import datetime_now


class _HistLen(NamedTuple):
    counts_len: int
    edges_len: int


class TemplateFunction(mlrun.runtimes.ServingRuntime):
    def __init__(self):
        super().__init__()
        self.add_trigger(
            "cron_interval",
            spec=nuclio.CronTrigger(interval=f"{1}m"),
        )


@pytest.fixture
def feature_stats() -> FeatureStats:
    return FeatureStats(
        {
            "feat0": {"hist": [[0, 1], [1.1, 2.2, 3.3]]},
            "feat1": {
                "key": "val",
                "hist": [[4, 2, 0, 2, 0], [-5, 5, 6, 100, 101, 222]],
            },
        }
    )


@pytest.fixture
def histogram() -> Histogram:
    return Histogram([[0, 1], [1.1, 2.2, 3.3]])


@pytest.fixture
def padded_histogram(histogram: Histogram) -> Histogram:
    pad_hist(histogram)
    return histogram


@pytest.fixture
def orig_feature_stats_hist_data(feature_stats: FeatureStats) -> dict[str, _HistLen]:
    data: dict[str, _HistLen] = {}
    for feat_name, feat in feature_stats.items():
        hist = feat["hist"]
        data[feat_name] = _HistLen(counts_len=len(hist[0]), edges_len=len(hist[1]))
    return data


def _check_padded_hist_spec(hist: Histogram, orig_data: _HistLen) -> None:
    counts = hist[0]
    edges = hist[1]
    edges_len = len(edges)
    counts_len = len(counts)
    assert edges_len == counts_len + 1
    assert counts_len == orig_data.counts_len + 2
    assert edges_len == orig_data.edges_len + 2
    assert counts[0] == counts[-1] == 0
    assert (-edges[0]) == edges[-1] == _MAX_FLOAT


def test_pad_hist(histogram: Histogram) -> None:
    orig_data = _HistLen(
        counts_len=len(histogram[0]),
        edges_len=len(histogram[1]),
    )
    pad_hist(histogram)
    _check_padded_hist_spec(histogram, orig_data)


def test_padded_hist_unchanged(padded_histogram: Histogram) -> None:
    orig_hist = padded_histogram.copy()
    pad_hist(padded_histogram)
    assert padded_histogram == orig_hist, "A padded histogram should not be changed"


def test_pad_features_hist(
    feature_stats: FeatureStats,
    orig_feature_stats_hist_data: dict[str, _HistLen],
) -> None:
    pad_features_hist(feature_stats)
    for feat_name, feat in feature_stats.items():
        _check_padded_hist_spec(feat["hist"], orig_feature_stats_hist_data[feat_name])


def generate_sample_data(
    feature_stats: FeatureStats,
    num_samples: int = 50,
) -> pd.DataFrame:
    data = {}
    for feature in feature_stats.keys():
        data[feature] = []
        for sample in range(num_samples):
            loc = np.random.uniform(
                low=feature_stats[feature]["hist"][1][0],
                high=feature_stats[feature]["hist"][1][-1],
            )
            feature_data = np.random.normal(loc=loc, scale=1.5, size=1)
            data[feature].append(float(feature_data))
    return pd.DataFrame(data)


def test_calculate_input_statistics(
    feature_stats: FeatureStats,
) -> None:
    """In the following test we will generate a sample data and calculate the input statistics based on the feature
    statistics. In addition, we will add a string feature to the sample data and check that it was removed from the
    input statistics."""

    input_data = generate_sample_data(feature_stats)

    # add string feature to input data
    input_data["str_feat"] = "blabla"
    current_stats = mlrun.model_monitoring.helpers.calculate_inputs_statistics(
        sample_set_statistics=feature_stats,
        inputs=input_data,
    )
    # check that the string feature was removed
    assert "str_feat" not in current_stats.keys()

    # check that the current_stats have the same keys as the feature_stats
    assert current_stats.keys() == feature_stats.keys()

    # validate the expected keys in a certain feature statistics
    feature_statistics = current_stats[next(iter(feature_stats))]
    assert list(feature_statistics.keys()) == [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "hist",
    ]


class TestBatchInterval:
    @staticmethod
    @pytest.fixture
    def timedelta_seconds(request: pytest.FixtureRequest) -> int:
        if marker := request.node.get_closest_marker(
            TestBatchInterval.timedelta_seconds.__name__
        ):
            return marker.args[0]
        return int(datetime.timedelta(minutes=6).total_seconds())

    @staticmethod
    @pytest.fixture
    def first_request(request: pytest.FixtureRequest) -> int:
        if marker := request.node.get_closest_marker(
            TestBatchInterval.first_request.__name__
        ):
            return marker.args[0]
        return int(
            datetime.datetime(
                2021, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
            ).timestamp()
        )

    @staticmethod
    @pytest.fixture
    def last_updated(request: pytest.FixtureRequest) -> int:
        if marker := request.node.get_closest_marker(
            TestBatchInterval.last_updated.__name__
        ):
            return marker.args[0]
        return int(
            datetime.datetime(
                2021, 1, 1, 13, 1, 0, tzinfo=datetime.timezone.utc
            ).timestamp()
        )

    @staticmethod
    @pytest.fixture(autouse=True)
    def _patch_store_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "MLRUN_MODEL_ENDPOINT_MONITORING__STORE_PREFIXES__DEFAULT",
            "memory://users/pipelines/{{project}}/model-endpoints/{{kind}}",
        )
        mlrun.mlconf.reload()

    @staticmethod
    @pytest.fixture
    def schedules_file() -> Iterator[ModelMonitoringSchedulesFile]:
        file = ModelMonitoringSchedulesFile(project="test-intervals", endpoint_id="ep")
        file.create()
        yield file
        file.delete()

    @staticmethod
    @pytest.fixture
    def intervals(
        schedules_file: ModelMonitoringSchedulesFile,
        timedelta_seconds: int,
        first_request: int,
        last_updated: int,
    ) -> list[_Interval]:
        with schedules_file as f:
            return list(
                _BatchWindow(
                    schedules_file=f,
                    application="app",
                    timedelta_seconds=timedelta_seconds,
                    first_request=first_request,
                    last_updated=last_updated,
                ).get_intervals()
            )

    @staticmethod
    @pytest.fixture
    def expected_intervals() -> list[_Interval]:
        def dt(hour: int, minute: int) -> datetime.datetime:
            return datetime.datetime(
                2021, 1, 1, hour, minute, tzinfo=datetime.timezone.utc
            )

        def interval(start: tuple[int, int], end: tuple[int, int]) -> _Interval:
            return _Interval(dt(*start), dt(*end))

        return [
            interval((12, 0), (12, 6)),
            interval((12, 6), (12, 12)),
            interval((12, 12), (12, 18)),
            interval((12, 18), (12, 24)),
            interval((12, 24), (12, 30)),
            interval((12, 30), (12, 36)),
            interval((12, 36), (12, 42)),
            interval((12, 42), (12, 48)),
            interval((12, 48), (12, 54)),
            interval((12, 54), (13, 0)),
        ]

    @staticmethod
    def test_touching_intervals(intervals: list[_Interval]) -> None:
        assert len(intervals) > 1, "There should be more than one interval"
        for prev, curr in zip(intervals[:-1], intervals[1:]):
            assert prev[1] == curr[0], "The intervals should be touching"

    @staticmethod
    def test_intervals(
        intervals: list[_Interval], expected_intervals: list[_Interval]
    ) -> None:
        assert len(intervals) == len(
            expected_intervals
        ), "The number of intervals is not as expected"
        assert intervals == expected_intervals, "The intervals are not as expected"

    @staticmethod
    def test_last_interval_does_not_overflow(
        intervals: list[_Interval], last_updated: int
    ) -> None:
        assert (
            intervals[-1][1].timestamp() <= last_updated
        ), "The last interval should be after last_updated"

    @staticmethod
    @pytest.mark.parametrize(
        (
            "timedelta_seconds",
            "first_request",
            "last_updated",
            "expected_last_analyzed",
        ),
        [
            (60, 100, 300, 100),
            (60, 100, 110, 100),
            (60, 0, 0, 0),
        ],
    )
    def test_get_last_analyzed(
        timedelta_seconds: int,
        last_updated: int,
        first_request: int,
        expected_last_analyzed: int,
        schedules_file: ModelMonitoringSchedulesFile,
    ) -> None:
        with schedules_file as f:
            assert (
                _BatchWindow(
                    schedules_file=f,
                    application="special-app",
                    timedelta_seconds=timedelta_seconds,
                    first_request=first_request,
                    last_updated=last_updated,
                )._get_last_analyzed()
                == expected_last_analyzed
            ), "The last analyzed time is not as expected"

    @staticmethod
    @pytest.mark.timedelta_seconds(int(datetime.timedelta(days=6).total_seconds()))
    @pytest.mark.first_request(
        int(
            datetime.datetime(
                2020, 12, 25, 23, 0, 0, tzinfo=datetime.timezone.utc
            ).timestamp()
        )
    )
    @pytest.mark.last_updated(
        int(
            datetime.datetime(
                2021, 1, 1, 3, 1, 0, tzinfo=datetime.timezone.utc
            ).timestamp()
        )
    )
    def test_large_base_period(
        timedelta_seconds: int, intervals: list[_Interval]
    ) -> None:
        assert len(intervals) == 1, "There should be exactly one interval"
        assert timedelta_seconds == datetime.datetime.timestamp(
            intervals[0][1]
        ) - datetime.datetime.timestamp(
            intervals[0][0]
        ), "The time slot should be equal to timedelta_seconds (6 days)"


class TestBatchWindowGenerator:
    @staticmethod
    def test_last_updated_is_in_the_past() -> None:
        last_request = datetime.datetime(2023, 11, 16, 12, 0, 0)
        last_updated = _BatchWindowGenerator._get_last_updated_time(
            last_request=last_request, not_batch_endpoint=True
        )
        assert last_updated
        assert (
            last_updated < last_request.timestamp()
        ), "The last updated time should be before the last request"


class TestBumpModelEndpointLastRequest:
    @staticmethod
    @pytest.fixture
    def project() -> str:
        return "project"

    @staticmethod
    @pytest.fixture
    def db() -> NopDB:
        return NopDB()

    @staticmethod
    @pytest.fixture
    def empty_model_endpoint() -> ModelEndpoint:
        return ModelEndpoint(
            metadata=mlrun.common.schemas.ModelEndpointMetadata(
                name="test", project="test-project"
            ),
            spec=mlrun.common.schemas.ModelEndpointSpec(),
            status=mlrun.common.schemas.ModelEndpointStatus(),
        )

    @staticmethod
    @pytest.fixture
    def last_request() -> str:
        return "2023-12-05 18:17:50.255143"

    @staticmethod
    @pytest.fixture
    def model_endpoint(
        empty_model_endpoint: ModelEndpoint, last_request: str
    ) -> ModelEndpoint:
        empty_model_endpoint.status.last_request = last_request
        return empty_model_endpoint

    @staticmethod
    @pytest.fixture
    def function() -> mlrun.runtimes.ServingRuntime:
        return TemplateFunction()

    @staticmethod
    def test_update_last_request(
        project: str,
        model_endpoint: ModelEndpoint,
        db: NopDB,
        last_request: str,
        function: mlrun.runtimes.ServingRuntime,
    ) -> None:
        with patch.object(db, "patch_model_endpoint") as patch_patch_model_endpoint:
            with patch.object(db, "get_function", return_value=function):
                update_model_endpoint_last_request(
                    project=project,
                    model_endpoint=model_endpoint,
                    current_request=datetime.datetime.fromisoformat(last_request),
                    db=db,
                )
        patch_patch_model_endpoint.assert_called_once()
        assert patch_patch_model_endpoint.call_args.kwargs["attributes"][
            EventFieldType.LAST_REQUEST
        ] == datetime.datetime.fromisoformat(last_request)
        model_endpoint.metadata.endpoint_type = EndpointType.BATCH_EP

        with patch.object(db, "patch_model_endpoint") as patch_patch_model_endpoint:
            with patch.object(db, "get_function", return_value=function):
                update_model_endpoint_last_request(
                    project=project,
                    model_endpoint=model_endpoint,
                    current_request=datetime.datetime.fromisoformat(last_request),
                    db=db,
                )
        patch_patch_model_endpoint.assert_called_once()
        assert patch_patch_model_endpoint.call_args.kwargs["attributes"][
            EventFieldType.LAST_REQUEST
        ] == datetime.datetime.fromisoformat(last_request) + datetime.timedelta(
            minutes=1
        ) + datetime.timedelta(
            seconds=mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs
        ), "The patched last request time should be bumped by the given delta"

    @staticmethod
    def test_no_bump(
        project: str,
        model_endpoint: ModelEndpoint,
        db: NopDB,
    ) -> None:
        with patch.object(db, "patch_model_endpoint") as patch_patch_model_endpoint:
            with patch.object(
                db, "get_function", side_effect=mlrun.errors.MLRunNotFoundError
            ):
                model_endpoint.metadata.endpoint_type = EndpointType.BATCH_EP
                update_model_endpoint_last_request(
                    project=project,
                    model_endpoint=model_endpoint,
                    current_request=datetime_now(),
                    db=db,
                )
        patch_patch_model_endpoint.assert_not_called()

    @staticmethod
    def test_get_monitoring_time_window_from_controller_run(
        project: str,
        db: NopDB,
        function: mlrun.runtimes.ServingRuntime,
    ) -> None:
        with patch.object(db, "get_function", return_value=function):
            assert _get_monitoring_time_window_from_controller_run(
                project=project,
                db=db,
            ) == datetime.timedelta(minutes=1), "The window is different than expected"


def test_get_invocations_fqn() -> None:
    assert get_invocations_fqn("project") == "project.mlrun-infra.metric.invocations"


def test_batch_dict2timedelta() -> None:
    assert batch_dict2timedelta(
        _BatchDict(minutes=32, hours=0, days=4)
    ) == datetime.timedelta(minutes=32, days=4), "Different timedelta than expected"


def test_filter_results_by_regex():
    existing_result_names = [
        "mep1.app1.result.metric1",
        "mep1.app1.result.metric2",
        "mep1.app2.result.metric1",
        "mep1.app2.result.metric2",
        "mep1.app2.result.metric3",
        "mep1.app2.result.result-a",
        "mep1.app2.result.result-b",
        "mep1.app3.result.result1",
        "mep1.app3.result.result2",
        "mep1.app4.result.result1",
        "mep1.app4.result.result2",
    ]

    expected_result_names = [
        "mep1.app1.result.metric1",
        "mep1.app2.result.metric1",
        "mep1.app2.result.result-a",
        "mep1.app2.result.result-b",
        "mep1.app3.result.result1",
        "mep1.app3.result.result2",
        "mep1.app4.result.result1",
    ]

    results_names_filters = [
        "*.metric1",
        "app2.result-*",
        "app3.*",
        "app4.result1",
    ]
    filtered_results = filter_results_by_regex(
        existing_result_names=existing_result_names,
        result_name_filters=results_names_filters,
    )
    assert sorted(filtered_results) == sorted(expected_result_names)


@pytest.mark.parametrize(
    ("project", "function_name", "expected_topic"),
    [
        ("p1", None, "monitoring_stream__p1_v1"),
        ("mm", "model-monitoring-stream", "monitoring_stream__mm_v1"),
        (
            "mm",
            "model-monitoring-controller",
            "monitoring_stream__mm_model-monitoring-controller_v1",
        ),
        ("mm", "model-monitoring-stream", "monitoring_stream__mm_v1"),
        (
            "special-mm-12",
            "model-monitoring-writer",
            "monitoring_stream__special-mm-12_model-monitoring-writer_v1",
        ),
    ],
)
def test_get_kafka_topic(
    project: str,
    function_name: Optional[str],
    expected_topic: str,
) -> None:
    assert (
        get_kafka_topic(project=project, function_name=function_name) == expected_topic
    ), "The topic is different than expected"


@pytest.mark.parametrize(
    ("profile", "expected_output_stream_type"),
    [
        (
            DatastoreProfileKafkaSource(
                name="test-kafka-profile",
                brokers=["localhost"],
                topics=[],
                sasl_user="user1",
                sasl_pass="1234",
                kwargs_public={"api_version": (3, 9)},
            ),
            KafkaOutputStream,
        ),
        (
            DatastoreProfileV3io(
                name="test-v3io-profile", v3io_access_key="valid-access-key"
            ),
            OutputStream,
        ),
    ],
)
def test_get_output_stream(
    profile: DatastoreProfile,
    expected_output_stream_type: Union[type[KafkaOutputStream], type[OutputStream]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if isinstance(profile, DatastoreProfileV3io):
        monkeypatch.setenv("V3IO_API", mlrun.mlconf.v3io_api)

    output_stream = get_output_stream(profile=profile, project="test-proj", mock=True)
    assert isinstance(
        output_stream, expected_output_stream_type
    ), "The output stream is of an unexpected type"

    output_stream.push(2 * [{"k1": 0, "jump": "high"}])
    output_stream.push([{"k1": 1, "jump": "mid"}])


def test_get_output_stream_unsupported() -> None:
    with pytest.raises(
        mlrun.errors.MLRunValueError,
        match=(
            r".*an unexpected stream profile type: "
            r"<class 'mlrun\.datastore\.datastore_profile\.DatastoreProfileKafkaTarget'>"
            r".*"
        ),
    ):
        get_output_stream(
            project="nmo",
            function_name="model-monitoring-controller",
            profile=DatastoreProfileKafkaTarget(
                name="k-tgt", brokers="localhost", topic="t1"
            ),
        )
