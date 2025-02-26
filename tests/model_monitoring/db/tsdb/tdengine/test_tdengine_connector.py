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

import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
import taosws
import taoswswrap.tdengine_connection

from mlrun.common.schemas.model_monitoring import (
    ModelEndpointMonitoringMetric,
    ModelEndpointMonitoringMetricType,
)
from mlrun.datastore.datastore_profile import TDEngineDatastoreProfile
from mlrun.model_monitoring.db.tsdb.tdengine import TDEngineConnector

project = "test-tdengine-connector"
connection_string = os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__TSDB_CONNECTION")
database = "test_tdengine_connector_" + uuid.uuid4().hex


def drop_database(connection: taosws.Connection) -> None:
    connection.execute(f"DROP DATABASE IF EXISTS {database}")


def is_tdengine_defined() -> bool:
    return connection_string is not None and connection_string.startswith("taosws://")


@pytest.fixture
def connector() -> Iterator[TDEngineConnector]:
    connection = taosws.connect(connection_string)
    drop_database(connection)
    profile = TDEngineDatastoreProfile.from_dsn(
        profile_name="mm-profile", dsn=connection_string
    )
    conn = TDEngineConnector(project, profile=profile, database=database)
    try:
        yield conn
    finally:
        drop_database(connection)


@pytest.mark.parametrize(("with_result_extra_data"), [False, True])
@pytest.mark.skipif(not is_tdengine_defined(), reason="TDEngine is not defined")
def test_write_application_event(
    connector: TDEngineConnector, with_result_extra_data: bool
) -> None:
    endpoint_id = "1"
    app_name = "my_app"
    result_name = "my_Result"
    result_kind = 0
    start_infer_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_infer_time = datetime(2024, 1, 1, second=1, tzinfo=timezone.utc)
    result_status = 0
    result_value = 123
    data = {
        "endpoint_id": endpoint_id,
        "application_name": app_name,
        "result_name": result_name,
        "result_kind": result_kind,
        "start_infer_time": start_infer_time,
        "end_infer_time": end_infer_time,
        "result_status": result_status,
        # make sure we can write apostrophes (ML-7535)
        "result_extra_data": """{"question": "Who wrote 'To Kill a Mockingbird'?"}""",
        "result_value": result_value,
    }
    with pytest.raises(
        taoswswrap.tdengine_connection.TDEngineError, match="Database not exist"
    ):
        connector.write_application_event(data)
    connector.create_tables()  # DB is created here
    connector.write_application_event(data)
    read_data_kwargs = {
        "endpoint_id": endpoint_id,
        "start": datetime(2023, 1, 1, 1, 0, 0),
        "end": datetime(2025, 1, 1, 1, 0, 0),
        "metrics": [
            ModelEndpointMonitoringMetric(
                project=project,
                app=app_name,
                name=result_name,
                type=ModelEndpointMonitoringMetricType.RESULT,
            ),
        ],
        "type": "results",
        "with_result_extra_data": with_result_extra_data,
    }
    read_back_results = connector.read_metrics_data(**read_data_kwargs)
    assert len(read_back_results) == 1
    read_back_result = read_back_results[0]
    assert read_back_result.full_name == f"{project}.{app_name}.result.{result_name}"
    assert read_back_result.data
    assert read_back_result.result_kind.value == result_kind
    assert read_back_result.type == "result"
    assert len(read_back_result.values) == 1
    read_back_values = read_back_result.values[0]
    assert read_back_values.timestamp == end_infer_time
    assert read_back_values.value == result_value
    assert read_back_values.status == result_status
    if with_result_extra_data:
        assert read_back_values.extra_data == data["result_extra_data"]

    # Delete resources and verify that database is deleted
    connector.delete_tsdb_resources()

    with pytest.raises(taoswswrap.tdengine_connection.TDEngineError):
        connector.read_metrics_data(**read_data_kwargs)
