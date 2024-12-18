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

import pandas as pd
import pytest

import mlrun.common.schemas.model_monitoring as mm_schemas
from mlrun.model_monitoring.db.tsdb.base import TSDBConnector


class TestTSDBConnectorStaticMethods:
    @pytest.fixture
    def test_data(self):
        """Fixture to create shared test data."""
        return pd.DataFrame(
            {
                "result_kind": [0, 0, 0, 0],
                "application_name": ["my_app", "my_app", "my_app", "my_app"],
                "endpoint_id": ["mep_uid1", "mep_uid1", "mep_uid2", "mep_uid2"],
                "result_name": ["result1", "result2", "result1", "result3"],
            }
        )

    def test_df_to_metrics_grouped_dict(self, test_data):
        metrics_by_endpoint = TSDBConnector.df_to_metrics_grouped_dict(
            df=test_data, type="result", project="my_project"
        )
        assert ["result1", "result2"] == sorted(
            [result.name for result in metrics_by_endpoint["mep_uid1"]]
        )
        assert ["result1", "result3"] == sorted(
            [result.name for result in metrics_by_endpoint["mep_uid2"]]
        )

    def test_df_to_metrics_list(self, test_data):
        results = TSDBConnector.df_to_metrics_list(
            df=test_data, type="result", project="my_project"
        )
        assert ["result1", "result1", "result2", "result3"] == sorted(
            [result.name for result in results]
        )

    def test_df_to_events_intersection_dict(self, test_data):
        intersection_dict = TSDBConnector.df_to_events_intersection_dict(
            df=test_data, type="result", project="my_project"
        )
        results = intersection_dict[
            mm_schemas.INTERSECT_DICT_KEYS[
                mm_schemas.ModelEndpointMonitoringMetricType.RESULT
            ]
        ]
        assert len(results) == 1
        assert results[0].full_name == "my_project.my_app.result.result1"
