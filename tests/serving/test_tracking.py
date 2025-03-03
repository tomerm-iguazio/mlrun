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

import json
from collections.abc import Iterator
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

import mlrun
import mlrun.common.schemas.model_monitoring.constants as mm_constants
from mlrun.common.schemas import ModelEndpointCreationStrategy
from mlrun.datastore.datastore_profile import (
    DatastoreProfileKafkaSource,
    register_temporary_client_datastore_profile,
    remove_temporary_client_datastore_profile,
)
from mlrun.platforms.iguazio import KafkaOutputStream
from mlrun.runtimes import ServingRuntime
from tests.serving.test_serving import _log_model

testdata = '{"inputs": [[5, 6]]}'


class ModelTestingClass(mlrun.serving.V2ModelServer):
    def load(self):
        self.context.logger.info(f"loading model {self.name}")

    def predict(self, request):
        print("predict:", request)
        multiplier = self.get_param("multiplier", 1)
        outputs = [value[0] * multiplier for value in request["inputs"]]
        return np.array(outputs)  # complex result type to check serialization


class ModelTestingCustomTrack(ModelTestingClass):
    def logged_results(self, request: dict, response: dict, op: str):
        return [[1]], [self.get_param("multiplier", 1)]


def test_tracking(rundb_mock):
    # test that predict() was tracked properly in the stream
    fn = mlrun.new_function("tests", kind="serving")
    fn.add_model(
        "my",
        ".",
        class_name=ModelTestingClass(multiplier=2, model_endpoint_uid="my-uid"),
    )
    fn.set_tracking("v3io://fake", stream_args={"mock": True, "access_key": "x"})

    server = fn.to_mock_server()
    server.test("/v2/models/my/infer", testdata)

    fake_stream = server.context.stream.output_stream._mock_queue
    assert len(fake_stream) == 1
    assert rec_to_data(fake_stream[0]) == ("my", "ModelTestingClass", [[5, 6]], [10])


def test_custom_tracking(rundb_mock):
    # test custom values tracking (using the logged_results() hook)
    fn = mlrun.new_function("tests", kind="serving")
    fn.add_model(
        "my",
        ".",
        class_name=ModelTestingCustomTrack(multiplier=2, model_endpoint_uid="my-uid"),
    )
    fn.set_tracking("v3io://fake", stream_args={"mock": True, "access_key": "x"})

    server = fn.to_mock_server()
    server.test("/v2/models/my/infer", testdata)

    fake_stream = server.context.stream.output_stream._mock_queue
    assert len(fake_stream) == 1
    assert rec_to_data(fake_stream[0]) == ("my", "ModelTestingCustomTrack", [[1]], [2])


def test_ensemble_tracking(rundb_mock):
    # test proper tracking of an ensemble (router + models are logged)
    fn = mlrun.new_function("tests", kind="serving")
    fn.set_topology(
        "router",
        mlrun.serving.VotingEnsemble(
            vote_type="regression", model_endpoint_uid="VotingEnsemble-uid"
        ),
    )
    fn.add_model(
        "1",
        ".",
        class_name=ModelTestingClass(multiplier=2, model_endpoint_uid="my-uid-1"),
    )
    fn.add_model(
        "2",
        ".",
        class_name=ModelTestingClass(multiplier=3, model_endpoint_uid="my-uid-2"),
    )
    fn.set_tracking("v3io://fake", stream_args={"mock": True, "access_key": "x"})

    server = fn.to_mock_server()
    resp = server.test("/v2/models/infer", testdata)

    fake_stream = server.context.stream.output_stream._mock_queue
    assert len(fake_stream) == 3
    print(resp)
    results = {}
    for rec in fake_stream:
        model, cls, inputs, outputs = rec_to_data(rec)
        results[model] = [cls, inputs, outputs]

    assert results == {
        "1": ["ModelTestingClass", [[5, 6]], [10]],
        "2": ["ModelTestingClass", [[5, 6]], [15]],
        "VotingEnsemble": ["VotingEnsemble", [[5, 6]], [12.5]],
    }


@pytest.mark.parametrize("enable_tracking", [True, False])
def test_tracked_function(rundb_mock, enable_tracking):
    with patch("mlrun.get_run_db", return_value=rundb_mock):
        project = mlrun.new_project("test-pro", save=False)
        fn = mlrun.new_function("test-fn", kind="serving", project=project.name)
        model_uri = _log_model(project)
        fn.add_model(
            "m1",
            model_uri,
            "ModelTestingClass",
            multiplier=5,
            model_endpoint_uid="my-uid",
            creation_strategy=ModelEndpointCreationStrategy.ARCHIVE,
        )
        fn.set_tracking("dummy://", enable_tracking=enable_tracking)
        server = fn.to_mock_server()
        server.test("/v2/models/m1/infer", testdata)
        dummy_stream = server.context.stream.output_stream
        if enable_tracking:
            assert (
                len(dummy_stream.event_list) == 1
            ), "expected stream to get one message"
        else:
            assert len(dummy_stream.event_list) == 0, "expected stream to be empty"


def rec_to_data(rec):
    data = json.loads(rec["data"])
    inputs = data["request"]["inputs"]
    outputs = data["resp"]["outputs"]
    return data["model"], data["class"], inputs, outputs


@pytest.fixture
def project() -> mlrun.MlrunProject:
    return mlrun.get_or_create_project("test-tracking")


@pytest.fixture
def _register_stream_profile(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    stream_profile_name = "special-stream"
    monkeypatch.setenv(
        mm_constants.ProjectSecretKeys.STREAM_PROFILE_NAME, stream_profile_name
    )
    profile = DatastoreProfileKafkaSource(
        name=stream_profile_name,
        brokers=["localhost"],
        topics=[],
        kwargs_public={"api_version": (3, 9)},
    )
    register_temporary_client_datastore_profile(profile)
    yield
    remove_temporary_client_datastore_profile(stream_profile_name)


@pytest.mark.usefixtures("rundb_mock", "_register_stream_profile")
def test_tracking_datastore_profile(project: mlrun.MlrunProject) -> None:
    fn = cast(
        ServingRuntime,
        project.set_function(
            name="test-tracking-from-profile", kind=ServingRuntime.kind
        ),
    )
    fn.add_model(
        "model1",
        ".",
        class_name=ModelTestingClass(multiplier=7, model_endpoint_uid="model1-uid"),
    )
    fn.set_tracking(stream_args={"mock": True})

    server = fn.to_mock_server()
    server.test("/v2/models/model1/predict", body=json.dumps({"inputs": [[-5.2, 0.6]]}))
    server.test(
        "/v2/models/model1/predict", body=json.dumps({"inputs": [[0, -0.1], [0.4, 0]]})
    )

    output_stream = cast(KafkaOutputStream, server.context.stream.output_stream)
    mocked_stream = output_stream._mock_queue
    assert len(mocked_stream) == 2

    event = mocked_stream[1]
    assert event["class"] == "ModelTestingClass"
    assert event["model"] == "model1"
    assert event["effective_sample_count"] == 2
    assert np.array_equal(event["request"]["inputs"], np.array([[0, -0.1], [0.4, 0]]))
    assert np.array_equal(event["resp"]["outputs"], np.array([0.0, 0.4 * 7]))
