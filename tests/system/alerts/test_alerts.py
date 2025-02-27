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

import json
import time
import typing

import deepdiff
import pytest

import mlrun
import mlrun.alerts
import mlrun.common.schemas.alert as alert_objects
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.model_monitoring.api
import tests.system.common.helpers.notifications as notification_helpers
from mlrun import mlconf
from mlrun.common.schemas.model_monitoring.model_endpoints import (
    ModelEndpoint,
    ModelEndpointList,
)
from mlrun.datastore import get_stream_pusher
from mlrun.datastore.datastore_profile import DatastoreProfileV3io
from mlrun.model_monitoring.helpers import get_stream_path
from tests.system.base import TestMLRunSystem
from tests.system.model_monitoring import TestMLRunSystemModelMonitoring


@TestMLRunSystem.skip_test_if_env_not_configured
class TestAlerts(TestMLRunSystem):
    project_name = "alerts-test-project"

    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: typing.Optional[str] = None

    @pytest.mark.smoke
    def test_job_failure_alert(self):
        """
        validate that an alert is sent in case a job fails
        """
        function_name = "test-func-job-failure-alert"
        self.project.set_function(
            name=function_name,
            func=str(self.assets_path / "function.py"),
            handler="handler",
            image="mlrun/mlrun" if self.image is None else self.image,
            kind="job",
        )

        # nuclio function for storing notifications, to validate that alert notifications were sent on the failed job
        nuclio_function_url = notification_helpers.deploy_notification_nuclio(
            self.project, self.image
        )

        # create an alert with webhook notification
        alert_name = "failure-webhook"
        alert_summary = "Job failed"
        run_id = f"{function_name}-handler"
        notifications = self._generate_failure_notifications(nuclio_function_url)
        self._create_custom_alert_config(
            name=alert_name,
            entity_kind=alert_objects.EventEntityKind.JOB,
            entity_id=run_id,
            summary=alert_summary,
            event_name=alert_objects.EventKind.FAILED,
            notifications=notifications,
        )

        with pytest.raises(Exception):
            self.project.run_function(function_name, watch=False)
            self.project.run_function(function_name)

        # in order to trigger the periodic monitor runs function, to detect the failed run and send an event on it
        mlrun.utils.retry_until_successful(
            3,
            10 * 6,
            self._logger,
            True,
            self._validate_project_alerts_summary,
            expected_job_alerts_count=2,
        )

        # Validate that the notifications was sent on the failed job
        expected_notifications = ["notification failure"]
        self._validate_notifications_on_nuclio(
            nuclio_function_url, expected_notifications
        )
        # get all activations
        activations = self._run_db.list_alert_activations(
            project=self.project_name,
            name=alert_name,
            severity=[
                alert_objects.AlertSeverity.LOW,
                alert_objects.AlertSeverity.HIGH,
            ],
            entity=f"~{run_id}*",
            entity_kind=alert_objects.EventEntityKind.JOB,
            event_kind=alert_objects.EventKind.FAILED,
        )
        assert len(activations) == 2
        entities = []
        for activation in activations:
            assert activation.name == alert_name
            assert activation.severity == alert_objects.AlertSeverity.LOW
            assert run_id in activation.entity_id
            assert activation.event_kind == alert_objects.EventKind.FAILED
            assert activation.number_of_events == 1

            # save entity_ids to check pagination below
            entities.append(activation.entity_id)

        assert activations[0].activation_time >= activations[1].activation_time

        # get paginated activations
        activations, token = self._run_db.paginated_list_alert_activations(
            project=self.project_name,
            name=alert_name,
            severity=[
                alert_objects.AlertSeverity.LOW,
                alert_objects.AlertSeverity.HIGH,
            ],
            entity=f"~{run_id}*",
            page_size=1,
            entity_kind=alert_objects.EventEntityKind.JOB,
            event_kind=alert_objects.EventKind.FAILED,
        )
        assert len(activations) == 1
        entities.remove(activations[0].entity_id)

        activations, token = self._run_db.paginated_list_alert_activations(
            project=self.project_name,
            page_token=token,
        )
        assert len(activations) == 1
        entities.remove(activations[0].entity_id)

        # ensure that paginated requests returned different values
        assert len(entities) == 0
        assert token is None

        mlrun.get_run_db().delete_function(
            name=function_name, project=self.project.name
        )

    @pytest.mark.model_monitoring
    def test_drift_detection_alert(self):
        """
        validate that an alert is sent with different result kind and different detection result
        """
        # enable model monitoring - deploy writer function
        tsdb_profile = TestMLRunSystemModelMonitoring.get_tsdb_profile(
            self.mm_tsdb_profile_data
        )
        self.project.register_datastore_profile(tsdb_profile)

        stream_profile = TestMLRunSystemModelMonitoring.get_stream_profile(
            self.mm_stream_profile_data
        )
        self.project.register_datastore_profile(stream_profile)

        self.project.set_model_monitoring_credentials(
            tsdb_profile_name=tsdb_profile.name,
            stream_profile_name=stream_profile.name,
        )

        self.project.enable_model_monitoring(image=self.image or "mlrun/mlrun")

        # deploy nuclio func for storing notifications, to validate an alert notifications were sent on drift detection
        nuclio_function_url = notification_helpers.deploy_notification_nuclio(
            self.project, self.image
        )
        model_endpoint = mlrun.model_monitoring.api.get_or_create_model_endpoint(
            project=self.project.metadata.name,
            model_endpoint_name="test-endpoint",
            context=mlrun.get_or_create_ctx("demo"),
        )

        # waits for the writer function to be deployed
        writer = self.project.get_function(
            key=mm_constants.MonitoringFunctionNames.WRITER
        )
        writer._wait_for_function_deployment(db=writer._get_db())

        stream_uri = get_stream_path(
            project=self.project.metadata.name,
            function_name=mm_constants.MonitoringFunctionNames.WRITER,
            profile=DatastoreProfileV3io(name="tmp"),
        )
        output_stream = get_stream_pusher(stream_uri)

        result_name = (
            mm_constants.HistogramDataDriftApplicationConstants.GENERAL_RESULT_NAME
        )
        output_stream.push(
            self._generate_typical_event(
                model_endpoint.metadata.uid, result_name, model_endpoint.metadata.name
            )
        )

        time.sleep(5)
        # generate alerts for the different result kind and return text from the expected notifications that will be
        # used later to validate that the notifications were sent as expected
        expected_notifications = self._generate_alerts(
            nuclio_function_url,
            model_endpoint,
        )
        output_stream.push(
            self._generate_anomaly_events(
                model_endpoint.metadata.uid, result_name, model_endpoint.metadata.name
            )
        )

        # wait for the nuclio function to check for the stream inputs
        time.sleep(15)
        self._validate_notifications_on_nuclio(
            nuclio_function_url, expected_notifications
        )

        # wait for the periodic project summaries calculation to start
        mlrun.utils.retry_until_successful(
            3,
            10 * 6,
            self._logger,
            True,
            self._validate_project_alerts_summary,
            expected_endpoint_alerts_count=4,
        )

    def test_job_failure_alert_sliding_window(self):
        """

        This test simulates a scenario where a job is expected to fail twice within a two-minute window,
        which should trigger an alert. The monitoring interval is taken into account to ensure the sliding
        window of events includes all relevant job failures, preventing the alert system from missing
        events that occur just before the monitoring run.

        The test first triggers a job failure, waits for slightly more than two minutes, and then triggers
        another job failure to confirm that the alert does not trigger prematurely. Finally, a third failure
        within the adjusted window is used to confirm that the alert triggers as expected.
        """
        function_name = "test-func-failure-alert-sliding-window"
        self.project.set_function(
            name=function_name,
            func=str(self.assets_path / "function.py"),
            handler="handler",
            image="mlrun/mlrun" if self.image is None else self.image,
            kind="job",
        )

        # nuclio function for storing notifications, to validate that alert notifications were sent on the failed job
        nuclio_function_url = notification_helpers.deploy_notification_nuclio(
            self.project, self.image
        )

        # create an alert with webhook notification that should trigger when the job fails twice in two minutes
        alert_name = "failure-webhook"
        alert_summary = "Job failed"
        alert_criteria = alert_objects.AlertCriteria(period="30s", count=2)
        run_id = f"{function_name}-handler"
        notifications = self._generate_failure_notifications(nuclio_function_url)

        self._create_custom_alert_config(
            alert_name,
            alert_objects.EventEntityKind.JOB,
            run_id,
            alert_summary,
            alert_objects.EventKind.FAILED,
            notifications,
            criteria=alert_criteria,
        )

        # this is the first failure
        with pytest.raises(Exception):
            self.project.run_function(function_name)

        # wait for the periodic monitor runs function to run as it may take up to the maximum events_generation_interval
        # to detect the event + an extra 40s to simulate a delay that is slightly longer than the alert period
        time.sleep(mlconf.alerts.events_generation_interval + 40)

        # this is the second failure
        with pytest.raises(Exception):
            self.project.run_function(function_name)

        # validate that no notifications were sent yet, as the two failures did not occur within the same period
        expected_notifications = []
        self._validate_notifications_on_nuclio(
            nuclio_function_url, expected_notifications
        )

        # this failure should fall within the adjusted sliding window when combined with the second failure
        # should trigger the alert
        with pytest.raises(Exception):
            self.project.run_function(function_name)

        # validate that the alert was triggered and the notification was sent
        expected_notifications = ["notification failure"]

        # wait since there might be a delay
        mlrun.utils.retry_until_successful(
            3,
            10 * 3,
            self._logger,
            True,
            self._validate_notifications_on_nuclio,
            nuclio_function_url,
            expected_notifications,
        )
        mlrun.get_run_db().delete_function(
            name=function_name, project=self.project.name
        )

    @staticmethod
    def _generate_failure_notifications(nuclio_function_url):
        notification = mlrun.common.schemas.Notification(
            kind="webhook",
            name="failure",
            params={
                "url": nuclio_function_url,
                "override_body": {
                    "operation": "add",
                    "data": "notification failure",
                },
            },
        )
        return [alert_objects.AlertNotification(notification=notification)]

    @staticmethod
    def _generate_drift_notifications(nuclio_function_url, result_kind):
        first_notification = mlrun.common.schemas.Notification(
            kind="webhook",
            name="drift",
            params={
                "url": nuclio_function_url,
                "override_body": {
                    "operation": "add",
                    "data": f"first drift of {result_kind}",
                },
            },
        )
        second_notification = mlrun.common.schemas.Notification(
            kind="webhook",
            name="drift2",
            params={
                "url": nuclio_function_url,
                "override_body": {
                    "operation": "add",
                    "data": f"second drift of {result_kind}",
                },
            },
        )
        return [
            alert_objects.AlertNotification(notification=first_notification),
            alert_objects.AlertNotification(notification=second_notification),
        ]

    def _create_custom_alert_config(
        self,
        name,
        entity_kind,
        entity_id,
        summary,
        event_name,
        notifications,
        criteria=None,
    ):
        alert_data = mlrun.alerts.alert.AlertConfig(
            project=self.project_name,
            name=name,
            summary=summary,
            severity=alert_objects.AlertSeverity.LOW,
            entities=alert_objects.EventEntities(
                kind=entity_kind, project=self.project_name, ids=[entity_id]
            ),
            trigger=alert_objects.AlertTrigger(events=[event_name]),
            criteria=criteria,
            notifications=notifications,
        )

        mlrun.get_run_db().store_alert_config(name, alert_data)

    def _create_alert_config(
        self,
        name,
        summary,
        model_endpoint,
        events,
        notifications,
        criteria=None,
    ):
        alert_data = self.project.create_model_monitoring_alert_configs(
            name=name,
            summary=summary,
            endpoints=ModelEndpointList(endpoints=[model_endpoint]),
            events=events,
            notifications=notifications,
            result_names=[],
            severity=alert_objects.AlertSeverity.LOW,
            criteria=criteria,
        )
        self.project.store_alert_config(alert_data[0])

    def _validate_project_alerts_summary(
        self,
        expected_job_alerts_count=0,
        expected_endpoint_alerts_count=0,
        expected_other_alerts_count=0,
    ):
        project_summary = mlrun.get_run_db().get_project_summary(
            project=self.project_name
        )
        assert project_summary.job_alerts_count == expected_job_alerts_count
        assert project_summary.endpoint_alerts_count == expected_endpoint_alerts_count
        assert project_summary.other_alerts_count == expected_other_alerts_count

    @staticmethod
    def _validate_notifications_on_nuclio(nuclio_function_url, expected_notifications):
        sent_notifications = list(
            notification_helpers.get_notifications_from_nuclio_and_reset_notification_cache(
                nuclio_function_url
            )
        )
        assert (
            deepdiff.DeepDiff(
                sent_notifications, expected_notifications, ignore_order=True
            )
            == {}
        )

    @staticmethod
    def _generate_typical_event(
        endpoint_id: str,
        result_name: str,
        endpoint_name: str,
    ) -> dict[str, typing.Any]:
        return {
            mm_constants.WriterEvent.ENDPOINT_ID: endpoint_id,
            mm_constants.WriterEvent.ENDPOINT_NAME: endpoint_name,
            mm_constants.WriterEvent.APPLICATION_NAME: mm_constants.HistogramDataDriftApplicationConstants.NAME,
            mm_constants.WriterEvent.START_INFER_TIME: "2023-09-11T12:00:00",
            mm_constants.WriterEvent.END_INFER_TIME: "2023-09-11T12:01:00",
            mm_constants.WriterEvent.EVENT_KIND: "result",
            mm_constants.WriterEvent.DATA: json.dumps(
                {
                    mm_constants.ResultData.RESULT_NAME: result_name,
                    mm_constants.ResultData.RESULT_KIND: mm_constants.ResultKindApp.model_performance.value,
                    mm_constants.ResultData.RESULT_VALUE: 0.1,
                    mm_constants.ResultData.RESULT_STATUS: mm_constants.ResultStatusApp.detected.value,
                    mm_constants.ResultData.RESULT_EXTRA_DATA: json.dumps(
                        {"threshold": 0.3}
                    ),
                }
            ),
        }

    @staticmethod
    def _generate_anomaly_events(
        endpoint_id: str,
        result_name: str,
        endpoint_name: str,
    ) -> list[dict[str, typing.Any]]:
        data_drift_example = {
            mm_constants.WriterEvent.ENDPOINT_ID: endpoint_id,
            mm_constants.WriterEvent.ENDPOINT_NAME: endpoint_name,
            mm_constants.WriterEvent.APPLICATION_NAME: mm_constants.HistogramDataDriftApplicationConstants.NAME,
            mm_constants.WriterEvent.START_INFER_TIME: "2023-09-11T12:00:00",
            mm_constants.WriterEvent.END_INFER_TIME: "2023-09-11T12:01:00",
            mm_constants.WriterEvent.EVENT_KIND: "result",
            mm_constants.WriterEvent.DATA: json.dumps(
                {
                    mm_constants.ResultData.RESULT_NAME: result_name,
                    mm_constants.ResultData.RESULT_KIND: mm_constants.ResultKindApp.data_drift.value,
                    mm_constants.ResultData.RESULT_VALUE: 0.5,
                    mm_constants.ResultData.RESULT_STATUS: mm_constants.ResultStatusApp.detected.value,
                    mm_constants.ResultData.RESULT_EXTRA_DATA: json.dumps(
                        {"threshold": 0.3}
                    ),
                }
            ),
        }

        concept_drift_example = {
            mm_constants.WriterEvent.ENDPOINT_ID: endpoint_id,
            mm_constants.WriterEvent.ENDPOINT_NAME: endpoint_name,
            mm_constants.WriterEvent.APPLICATION_NAME: mm_constants.HistogramDataDriftApplicationConstants.NAME,
            mm_constants.WriterEvent.START_INFER_TIME: "2023-09-11T12:00:00",
            mm_constants.WriterEvent.END_INFER_TIME: "2023-09-11T12:01:00",
            mm_constants.WriterEvent.EVENT_KIND: "result",
            mm_constants.WriterEvent.DATA: json.dumps(
                {
                    mm_constants.ResultData.RESULT_NAME: result_name,
                    mm_constants.ResultData.RESULT_KIND: mm_constants.ResultKindApp.concept_drift.value,
                    mm_constants.ResultData.RESULT_VALUE: 0.9,
                    mm_constants.ResultData.RESULT_STATUS: mm_constants.ResultStatusApp.potential_detection.value,
                    mm_constants.ResultData.RESULT_EXTRA_DATA: json.dumps(
                        {"threshold": 0.7}
                    ),
                }
            ),
        }

        anomaly_example = {
            mm_constants.WriterEvent.ENDPOINT_ID: endpoint_id,
            mm_constants.WriterEvent.ENDPOINT_NAME: endpoint_name,
            mm_constants.WriterEvent.APPLICATION_NAME: mm_constants.HistogramDataDriftApplicationConstants.NAME,
            mm_constants.WriterEvent.START_INFER_TIME: "2023-09-11T12:00:00",
            mm_constants.WriterEvent.END_INFER_TIME: "2023-09-11T12:01:00",
            mm_constants.WriterEvent.EVENT_KIND: "result",
            mm_constants.WriterEvent.DATA: json.dumps(
                {
                    mm_constants.ResultData.RESULT_NAME: result_name,
                    mm_constants.ResultData.RESULT_KIND: mm_constants.ResultKindApp.mm_app_anomaly.value,
                    mm_constants.ResultData.RESULT_VALUE: 0.9,
                    mm_constants.ResultData.RESULT_STATUS: mm_constants.ResultStatusApp.detected.value,
                    mm_constants.ResultData.RESULT_EXTRA_DATA: json.dumps(
                        {"threshold": 0.4}
                    ),
                }
            ),
        }

        system_performance_example = {
            mm_constants.WriterEvent.ENDPOINT_ID: endpoint_id,
            mm_constants.WriterEvent.ENDPOINT_NAME: endpoint_name,
            mm_constants.WriterEvent.APPLICATION_NAME: mm_constants.HistogramDataDriftApplicationConstants.NAME,
            mm_constants.WriterEvent.START_INFER_TIME: "2023-09-11T12:00:00",
            mm_constants.WriterEvent.END_INFER_TIME: "2023-09-11T12:01:00",
            mm_constants.WriterEvent.EVENT_KIND: "result",
            mm_constants.WriterEvent.DATA: json.dumps(
                {
                    mm_constants.ResultData.RESULT_NAME: result_name,
                    mm_constants.ResultData.RESULT_KIND: mm_constants.ResultKindApp.system_performance.value,
                    mm_constants.ResultData.RESULT_VALUE: 0.9,
                    mm_constants.ResultData.RESULT_STATUS: mm_constants.ResultStatusApp.detected.value,
                    mm_constants.ResultData.RESULT_EXTRA_DATA: json.dumps(
                        {"threshold": 0.4}
                    ),
                }
            ),
        }

        return [
            data_drift_example,
            concept_drift_example,
            anomaly_example,
            system_performance_example,
        ]

    def _generate_alerts(
        self, nuclio_function_url: str, model_endpoint: ModelEndpoint
    ) -> list[str]:
        """Generate alerts for the different result kind and return data from the expected notifications."""
        expected_notifications = []
        alerts_kind_to_test = [
            alert_objects.EventKind.DATA_DRIFT_DETECTED,
            alert_objects.EventKind.CONCEPT_DRIFT_SUSPECTED,
            alert_objects.EventKind.MM_APP_ANOMALY_DETECTED,
            alert_objects.EventKind.SYSTEM_PERFORMANCE_DETECTED,
        ]
        # Create alert configurations for each alert_kind individually.
        # This ensures different notifications are raised, which is why we don't send all alert_kinds as events at once.

        for alert_kind in alerts_kind_to_test:
            alert_name = mlrun.utils.helpers.normalize_name(
                f"drift-webhook-{alert_kind}"
            )
            alert_summary = "Model is drifting"
            self._create_alert_config(
                name=alert_name,
                summary=alert_summary,
                model_endpoint=model_endpoint,
                events=[alert_kind],
                notifications=self._generate_drift_notifications(
                    nuclio_function_url, alert_kind.value
                ),
            )
            expected_notifications.extend(
                [
                    f"first drift of {alert_kind.value}",
                    f"second drift of {alert_kind.value}",
                ]
            )
        return expected_notifications
