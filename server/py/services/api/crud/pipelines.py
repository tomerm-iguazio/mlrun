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
#
import ast
import concurrent.futures
import http
import tempfile
import traceback
import typing

import kfp_server_api
import sqlalchemy.orm

import mlrun
import mlrun.common.formatters
import mlrun.common.helpers
import mlrun.common.schemas
import mlrun.errors
import mlrun.utils
import mlrun.utils.helpers
import mlrun.utils.singleton
import mlrun_pipelines.common.models
import mlrun_pipelines.common.ops
import mlrun_pipelines.imports
import mlrun_pipelines.mixins
import mlrun_pipelines.models
import mlrun_pipelines.utils

import framework.api.utils
import services.api.crud


class Pipelines(
    mlrun_pipelines.mixins.PipelineProviderMixin,
    metaclass=mlrun.utils.singleton.Singleton,
):
    def list_pipelines(
        self,
        db_session: sqlalchemy.orm.Session,
        project: typing.Optional[typing.Union[str, list[str]]] = None,
        namespace: typing.Optional[str] = None,
        sort_by: str = "",
        page_token: str = "",
        filter_: str = "",
        name_contains: str = "",
        format_: mlrun.common.formatters.PipelineFormat = mlrun.common.formatters.PipelineFormat.metadata_only,
        page_size: typing.Optional[int] = None,
    ) -> tuple[int, typing.Optional[int], list[dict]]:
        if format_ == mlrun.common.formatters.PipelineFormat.summary:
            # we don't support summary format in list pipelines since the returned runs doesn't include the workflow
            # manifest status that includes the nodes section we use to generate the DAG.
            # (There is a workflow manifest under the run's pipeline_spec field, but it doesn't include the status)
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Summary format is not supported for list pipelines, use get instead"
            )

        project_names = None
        if isinstance(project, list):
            project_names = project
        elif project and project != "*":
            project_names = [project]

        kfp_client = self.initialize_kfp_client(namespace)
        # If no filter is provided and the project is not "*",
        # automatically apply a filter to match runs where the project name
        # is a substring of the pipeline's name. This ensures that only pipelines
        # with the project name in their name are returned, helping narrow down the results.
        if not filter_ and project_names and len(project_names) == 1:
            mlrun.utils.logger.debug(
                "No filter provided. "
                "Applying project-based filter for project to match pipelines with project name as a substring",
                project=project_names[0],
            )
            filter_ = mlrun.utils.get_kfp_project_filter(project_name=project_names[0])
        runs, next_page_token = self._paginate_runs(
            kfp_client, page_token, page_size, sort_by, filter_
        )
        if project_names:
            runs = [
                run
                for run in runs
                if self.resolve_project_from_pipeline(run) in project_names
            ]
        runs = self._filter_runs_by_name(runs, name_contains)
        runs = self._format_runs(runs, format_, kfp_client)
        # In-memory filtering turns Kubeflow's counting inaccurate if there are multiple pages of data
        # so don't pass it to the client in such case
        total_size = -1 if next_page_token else len(runs)

        return total_size, next_page_token, runs

    def delete_pipelines_runs(
        self, db_session: sqlalchemy.orm.Session, project_name: str
    ):
        _, _, project_pipeline_runs = self.list_pipelines(
            db_session=db_session,
            project=project_name,
            format_=mlrun.common.formatters.PipelineFormat.metadata_only,
        )
        kfp_client = self.initialize_kfp_client()

        if project_pipeline_runs:
            mlrun.utils.logger.debug(
                "Detected pipeline runs for project, deleting them",
                project_name=project_name,
                pipeline_run_count=len(project_pipeline_runs),
            )

        runs_succeeded = 0
        runs_failed = 0
        experiment_ids = set()
        delete_run_futures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=mlrun.mlconf.workflows.concurrent_delete_worker_count,
            thread_name_prefix="delete_workflow_experiment_",
        ) as executor:
            for pipeline_run in project_pipeline_runs:
                pipeline_run = mlrun_pipelines.models.PipelineRun(pipeline_run)
                # delete pipeline run also terminates it if it is in progress
                delete_run_futures.append(
                    executor.submit(kfp_client._run_api.delete_run, pipeline_run.id)
                )
                if pipeline_run.experiment_id:
                    experiment_ids.add(pipeline_run.experiment_id)
            for future in concurrent.futures.as_completed(delete_run_futures):
                delete_run_exception = future.exception()
                if delete_run_exception is not None:
                    # we don't want to fail the entire delete operation if we failed to delete a single pipeline run
                    # so it won't fail the delete project operation. we will log the error and continue
                    mlrun.utils.logger.warning(
                        "Failed to delete pipeline run",
                        project_name=project_name,
                        pipeline_run_id=pipeline_run.id,
                        exc_info=delete_run_exception,
                    )
                    runs_failed += 1
                else:
                    runs_succeeded += 1
            else:
                mlrun.utils.logger.debug(
                    "Finished deleting pipeline runs",
                    project_name=project_name,
                    succeeded=runs_succeeded,
                    failed=runs_failed,
                )

        experiments_succeeded = 0
        experiments_failed = 0
        delete_experiment_futures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=mlrun.mlconf.workflows.concurrent_delete_worker_count,
            thread_name_prefix="delete_workflow_experiment_",
        ) as executor:
            for experiment_id in experiment_ids:
                mlrun.utils.logger.debug(
                    f"Detected experiment for project {project_name} and deleting it",
                    project_name=project_name,
                    experiment_id=experiment_id,
                )
                delete_experiment_futures.append(
                    executor.submit(
                        kfp_client._experiment_api.delete_experiment,
                        experiment_id,
                    )
                )
            for future in concurrent.futures.as_completed(delete_run_futures):
                delete_experiment_exception = future.exception()
                if delete_experiment_exception is not None:
                    experiments_failed += 1
                    mlrun.utils.logger.warning(
                        "Failed to delete an experiment",
                        project_name=project_name,
                        experiment_id=experiment_id,
                        exc_info=mlrun.errors.err_to_str(delete_experiment_exception),
                    )
                else:
                    experiments_succeeded += 1
            else:
                mlrun.utils.logger.debug(
                    "Finished deleting project experiments",
                    project_name=project_name,
                    succeeded=experiments_succeeded,
                    failed=experiments_failed,
                )

    def get_pipeline(
        self,
        db_session: sqlalchemy.orm.Session,
        run_id: str,
        project: typing.Optional[str] = None,
        namespace: typing.Optional[str] = None,
        format_: mlrun.common.formatters.PipelineFormat = mlrun.common.formatters.PipelineFormat.summary,
    ):
        kfp_client = self.initialize_kfp_client(namespace)
        run = None
        try:
            api_run_detail = kfp_client.get_run(run_id)
            run = mlrun_pipelines.models.PipelineRun(api_run_detail)
            if run:
                if project and project != "*":
                    run_project = self.resolve_project_from_pipeline(run)
                    if run_project != project:
                        raise mlrun.errors.MLRunNotFoundError(
                            f"Pipeline run with id {run_id} is not of project {project}"
                        )

                mlrun.utils.logger.debug(
                    "Got kfp run",
                    run_id=run_id,
                    run_name=run.get("name"),
                    project=project,
                    format_=format_,
                )
                run = self._format_run(run, format_, kfp_client)
        except kfp_server_api.ApiException as exc:
            raise mlrun.errors.err_for_status_code(
                exc.status, mlrun.errors.err_to_str(exc)
            ) from exc
        except mlrun.errors.MLRunHTTPStatusError:
            raise
        except Exception as exc:
            raise mlrun.errors.MLRunRuntimeError(
                f"Failed getting KFP run: {mlrun.errors.err_to_str(exc)}"
            ) from exc

        return run

    def retry_pipeline(
        self,
        db_session: sqlalchemy.orm.Session,
        run_id: str,
        project: str,
        namespace: typing.Optional[str] = None,
    ) -> str:
        """
        Retry a Kubeflow Pipeline (KFP) run.

        :param db_session: The SQLAlchemy session used for retrieving and storing pipeline information.
        :param run_id: The unique identifier of the pipeline run to retry.
        :param project: The name of the MLRun project associated with the pipeline run.
        :param namespace: (Optional) The Kubernetes namespace in which the pipeline is running.
                          Defaults to the configured namespace if not specified.
        :raises MLRunBadRequestError: If the pipeline run is not in a retryable state.
        :raises MLRunNotFoundError: If the pipeline run does not belong to the specified project
                                    or if the run ID is not found.
        :raises MLRunRuntimeError: If there is an error retrieving the pipeline run details.
        :raises MLRunHTTPStatusError: If there is an HTTP error interacting with KFP.
        :return: The unique identifier of the retried pipeline run.
        :rtype: str
        """
        kfp_client = self.initialize_kfp_client(namespace)
        try:
            api_run_detail = kfp_client.get_run(run_id)
        except kfp_server_api.ApiException as exc:
            raise mlrun.errors.err_for_status_code(
                exc.status, mlrun.errors.err_to_str(exc)
            ) from exc
        except mlrun.errors.MLRunHTTPStatusError:
            raise
        except Exception as exc:
            raise mlrun.errors.MLRunRuntimeError(
                f"Failed getting KFP run: {mlrun.errors.err_to_str(exc)}"
            ) from exc
        run = mlrun_pipelines.models.PipelineRun(api_run_detail)

        if project:
            run_project = self.resolve_project_from_pipeline(run)
            if run_project != project:
                raise mlrun.errors.MLRunNotFoundError(
                    f"Pipeline run with id {run_id} is not of project {project}"
                )

        # Check if the pipeline is in a completed state
        if (
            run.status
            not in mlrun_pipelines.common.models.RunStatuses.retryable_statuses()
        ):
            raise mlrun.errors.MLRunBadRequestError(
                f"Pipeline run {run_id} is not in a completed state. Current status: {run.status}"
            )

        mlrun.utils.logger.debug(
            "Retrying KFP run",
            run_id=run_id,
            run_name=run.get("name"),
            project=project,
        )
        return kfp_client.retry_run(
            run_id=run_id,
            project=run_project,
        )

    def create_pipeline(
        self,
        experiment_name: str,
        run_name: str,
        content_type: str,
        data: bytes,
        arguments: typing.Optional[dict] = None,
    ):
        if arguments is None:
            arguments = {}
        if "/yaml" in content_type:
            content_type = ".yaml"
        elif " /zip" in content_type:
            content_type = ".zip"
        else:
            framework.api.utils.log_and_raise(
                http.HTTPStatus.BAD_REQUEST.value,
                reason=f"unsupported pipeline type {content_type}",
            )
        mlrun.utils.logger.debug(
            "Writing pipeline to temp file", content_type=content_type
        )
        data = mlrun_pipelines.common.ops.replace_kfp_plaintext_secret_env_vars_with_secret_refs(
            byte_buffer=data,
            content_type=content_type,
            env_var_names=["MLRUN_AUTH_SESSION", "V3IO_ACCESS_KEY"],
            secrets_store=services.api.crud.Secrets(),
        )
        pipeline_file = tempfile.NamedTemporaryFile(suffix=content_type)
        with open(pipeline_file.name, "wb") as fp:
            fp.write(data)

        mlrun.utils.logger.info(
            "Creating pipeline",
            experiment_name=experiment_name,
            run_name=run_name,
            arguments=arguments,
        )

        try:
            kfp_client = self.initialize_kfp_client()
            experiment = mlrun_pipelines.models.PipelineExperiment(
                kfp_client.create_experiment(name=experiment_name)
            )
            run = mlrun_pipelines.models.PipelineRun(
                kfp_client.run_pipeline(
                    experiment.id, run_name, pipeline_file.name, params=arguments
                )
            )
        except Exception as exc:
            mlrun.utils.logger.warning(
                "Failed creating pipeline",
                traceback=traceback.format_exc(),
                exc=mlrun.errors.err_to_str(exc),
            )
            raise mlrun.errors.MLRunBadRequestError(
                f"Failed creating pipeline: {mlrun.errors.err_to_str(exc)}"
            )
        finally:
            pipeline_file.close()

        return run

    @staticmethod
    def initialize_kfp_client(namespace: typing.Optional[str] = None):
        return mlrun_pipelines.utils.get_client(mlrun.mlconf.kfp_url, namespace)

    def _paginate_runs(
        self,
        kfp_client: mlrun_pipelines.imports.kfp.Client,
        page_token: typing.Optional[str] = None,
        page_size: typing.Optional[int] = None,
        sort_by: typing.Optional[str] = None,
        filter_: typing.Optional[str] = None,
    ) -> tuple[list[mlrun_pipelines.models.PipelineRun], typing.Optional[int]]:
        next_page_token = -1
        if page_token or page_size:
            # If page token or page size is given, the client is performing the pagination.
            # So we don't need to paginate the runs ourselves, only pass on the page token and page size
            # and ignore the filter if needed.
            runs, next_page_token = self._list_runs_from_kfp(
                kfp_client,
                page_token,
                page_size or mlrun.common.schemas.PipelinesPagination.default_page_size,
                sort_by,
                filter_,
            )
        else:
            # Otherwise, we perform the pagination ourselves, and get all the runs to return.
            runs = []
            while next_page_token:
                page_runs, next_page_token = self._list_runs_from_kfp(
                    kfp_client,
                    page_token,
                    page_size or mlrun.common.schemas.PipelinesPagination.max_page_size,
                    sort_by,
                    filter_,
                )
                runs.extend(page_runs)
                page_token = next_page_token

        return runs, next_page_token

    def _list_runs_from_kfp(
        self,
        kfp_client: mlrun_pipelines.imports.kfp.Client,
        page_token: typing.Optional[str] = None,
        page_size: typing.Optional[int] = None,
        sort_by: typing.Optional[str] = None,
        filter_: typing.Optional[str] = None,
    ) -> tuple[list[mlrun_pipelines.models.PipelineRun], typing.Optional[str]]:
        try:
            response = kfp_client.list_runs(
                page_token=page_token,
                page_size=page_size
                or mlrun.common.schemas.PipelinesPagination.default_page_size,
                sort_by=sort_by if not page_token else "",
                filter=filter_ if not page_token else "",
            )
        except kfp_server_api.ApiException as exc:
            # extract the summary of the error message from the exception
            error_message = exc.body or exc.reason or exc
            if "message" in error_message:
                error_message = error_message["message"]
            raise mlrun.errors.err_for_status_code(
                exc.status, mlrun.errors.err_to_str(error_message)
            ) from exc

        return [
            mlrun_pipelines.models.PipelineRun(run) for run in response.runs or []
        ], response.next_page_token

    def _format_runs(
        self,
        runs: list[dict],
        format_: mlrun.common.formatters.PipelineFormat = mlrun.common.formatters.PipelineFormat.metadata_only,
        kfp_client: mlrun_pipelines.imports.kfp.Client = None,
    ) -> list[dict]:
        formatted_runs = []
        for run in runs:
            formatted_runs.append(self._format_run(run, format_, kfp_client))
        return formatted_runs

    def _format_run(
        self,
        run: mlrun_pipelines.models.PipelineRun,
        format_: mlrun.common.formatters.PipelineFormat = mlrun.common.formatters.PipelineFormat.metadata_only,
        kfp_client: mlrun_pipelines.imports.kfp.Client = None,
    ) -> dict:
        run.project = self.resolve_project_from_pipeline(run)
        if kfp_client:
            error = self.get_error_from_pipeline(kfp_client, run)
            if error:
                run.error = error
        return mlrun.common.formatters.PipelineFormat.format_obj(run, format_)

    def _resolve_project_from_command(
        self,
        command: list[str],
        hyphen_p_is_also_project: bool,
        has_func_url_flags: bool,
        has_runtime_flags: bool,
    ):
        # project has precedence over function url so search for it first
        for index, argument in enumerate(command):
            if (
                (argument == "-p" and hyphen_p_is_also_project)
                or argument == "--project"
            ) and index + 1 < len(command):
                return command[index + 1]
        if has_func_url_flags:
            for index, argument in enumerate(command):
                if (argument == "-f" or argument == "--func-url") and index + 1 < len(
                    command
                ):
                    function_url = command[index + 1]
                    if function_url.startswith("db://"):
                        (
                            project,
                            _,
                            _,
                            _,
                        ) = mlrun.common.helpers.parse_versioned_object_uri(
                            function_url[len("db://") :]
                        )
                        if project:
                            return project
        if has_runtime_flags:
            for index, argument in enumerate(command):
                if (argument == "-r" or argument == "--runtime") and index + 1 < len(
                    command
                ):
                    runtime = command[index + 1]
                    try:
                        parsed_runtime = ast.literal_eval(runtime)
                    except Exception as exc:
                        mlrun.utils.logger.warning(
                            "Failed parsing runtime. Skipping",
                            runtime=runtime,
                            exc=mlrun.errors.err_to_str(exc),
                        )
                    else:
                        if isinstance(parsed_runtime, dict):
                            project = parsed_runtime.get("metadata", {}).get("project")
                            if project:
                                return project

        return None

    def resolve_project_from_pipeline(
        self, pipeline: mlrun_pipelines.models.PipelineRun
    ):
        return self.resolve_project_from_workflow_manifest(pipeline.workflow_manifest())

    def get_error_from_pipeline(
        self, kfp_client, run: mlrun_pipelines.models.PipelineRun
    ):
        pipeline = kfp_client.get_run(run.id)
        return self.resolve_error_from_pipeline(pipeline)

    def _filter_runs_by_name(self, runs: list, target_name: str) -> list:
        """Filter runs by their name while ignoring the project string on them

        :param runs: list of runs to be filtered
        :param target_name: string that should be part of a valid run name
        :return: filtered list of runs
        """
        if not target_name:
            return runs

        def filter_by(run):
            run_name = run.get("name", "").removeprefix(
                self.resolve_project_from_pipeline(run) + "-"
            )
            if target_name in run_name:
                return True
            return False

        return list(filter(filter_by, runs))
