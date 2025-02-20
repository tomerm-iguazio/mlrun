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
import string
import typing
import unittest.mock

import pytest
import sqlalchemy.exc
import sqlalchemy.orm

import mlrun
import mlrun.common.db.sql_session
import mlrun.common.schemas
from mlrun.config import config

import framework.constants
import framework.db.init_db
import framework.db.sqldb.db
import framework.db.sqldb.models
import framework.utils.singletons.db
import services.api.initial_data


def test_add_data_version_empty_db():
    db, db_session = _initialize_db_without_migrations()
    # currently the latest is 1, which is also the value we'll put there if the db is not empty so change it to 3 to
    # differentiate between the two
    original_latest_data_version = services.api.initial_data.latest_data_version
    services.api.initial_data.latest_data_version = "3"
    assert db.get_current_data_version(db_session, raise_on_not_found=False) is None
    services.api.initial_data._add_initial_data(db_session)
    assert (
        db.get_current_data_version(db_session, raise_on_not_found=True)
        == services.api.initial_data.latest_data_version
    )
    services.api.initial_data.latest_data_version = original_latest_data_version


def test_add_data_version_non_empty_db():
    db, db_session = _initialize_db_without_migrations()
    # currently the latest is 1, which is also the value we'll put there if the db is not empty so change it to 3 to
    # differentiate between the two
    original_latest_data_version = services.api.initial_data.latest_data_version
    services.api.initial_data.latest_data_version = "3"

    assert db.get_current_data_version(db_session, raise_on_not_found=False) is None
    # fill db
    db.create_project(
        db_session,
        mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name="project-name"),
        ),
    )
    services.api.initial_data._add_initial_data(db_session)
    assert db.get_current_data_version(db_session, raise_on_not_found=True) == "1"
    services.api.initial_data.latest_data_version = original_latest_data_version


def test_perform_data_migrations_from_first_version():
    db, db_session = _initialize_db_without_migrations()

    # set version to 1
    db.create_data_version(db_session, "1")

    # keep a reference to the original functions, so we can restore them later
    original_perform_version_2_data_migrations = (
        services.api.initial_data._perform_version_2_data_migrations
    )
    services.api.initial_data._perform_version_2_data_migrations = unittest.mock.Mock()
    original_perform_version_3_data_migrations = (
        services.api.initial_data._perform_version_3_data_migrations
    )
    services.api.initial_data._perform_version_3_data_migrations = unittest.mock.Mock()
    original_perform_version_4_data_migrations = (
        services.api.initial_data._perform_version_4_data_migrations
    )
    services.api.initial_data._perform_version_4_data_migrations = unittest.mock.Mock()
    original_perform_version_5_data_migrations = (
        services.api.initial_data._perform_version_5_data_migrations
    )
    services.api.initial_data._perform_version_5_data_migrations = unittest.mock.Mock()
    original_perform_version_6_data_migrations = (
        services.api.initial_data._perform_version_6_data_migrations
    )
    services.api.initial_data._perform_version_6_data_migrations = unittest.mock.Mock()
    original_perform_version_7_data_migrations = (
        services.api.initial_data._perform_version_7_data_migrations
    )
    services.api.initial_data._perform_version_7_data_migrations = unittest.mock.Mock()

    original_perform_version_8_data_migrations = (
        services.api.initial_data._perform_version_8_data_migrations
    )
    services.api.initial_data._perform_version_8_data_migrations = unittest.mock.Mock()

    original_perform_version_9_data_migrations = (
        services.api.initial_data._perform_version_9_data_migrations
    )
    services.api.initial_data._perform_version_9_data_migrations = unittest.mock.Mock()

    # perform migrations
    services.api.initial_data._perform_data_migrations(db_session)

    # calling again should not trigger migrations again, since we're already at the latest version
    services.api.initial_data._perform_data_migrations(db_session)

    services.api.initial_data._perform_version_2_data_migrations.assert_called_once()
    services.api.initial_data._perform_version_3_data_migrations.assert_called_once()
    services.api.initial_data._perform_version_4_data_migrations.assert_called_once()
    services.api.initial_data._perform_version_5_data_migrations.assert_called_once()
    services.api.initial_data._perform_version_6_data_migrations.assert_called_once()
    services.api.initial_data._perform_version_7_data_migrations.assert_called_once()
    services.api.initial_data._perform_version_8_data_migrations.assert_called_once()
    services.api.initial_data._perform_version_9_data_migrations.assert_called_once()

    assert db.get_current_data_version(db_session, raise_on_not_found=True) == str(
        services.api.initial_data.latest_data_version
    )

    # restore original functions
    services.api.initial_data._perform_version_2_data_migrations = (
        original_perform_version_2_data_migrations
    )
    services.api.initial_data._perform_version_3_data_migrations = (
        original_perform_version_3_data_migrations
    )
    services.api.initial_data._perform_version_4_data_migrations = (
        original_perform_version_4_data_migrations
    )
    services.api.initial_data._perform_version_5_data_migrations = (
        original_perform_version_5_data_migrations
    )
    services.api.initial_data._perform_version_6_data_migrations = (
        original_perform_version_6_data_migrations
    )
    services.api.initial_data._perform_version_7_data_migrations = (
        original_perform_version_7_data_migrations
    )
    services.api.initial_data._perform_version_8_data_migrations = (
        original_perform_version_8_data_migrations
    )
    services.api.initial_data._perform_version_9_data_migrations = (
        original_perform_version_9_data_migrations
    )


def test_resolve_current_data_version_version_exists():
    db, db_session = _initialize_db_without_migrations()

    db.create_data_version(db_session, "1")
    assert services.api.initial_data._resolve_current_data_version(db, db_session) == 1


@pytest.mark.parametrize("table_exists", [True, False])
@pytest.mark.parametrize("db_type", ["mysql", "sqlite"])
def test_resolve_current_data_version_before_and_after_projects(table_exists, db_type):
    db, db_session = _initialize_db_without_migrations()

    original_latest_data_version = services.api.initial_data.latest_data_version
    services.api.initial_data.latest_data_version = 3

    if not table_exists:
        # simulating table doesn't exist in DB
        db.get_current_data_version = unittest.mock.Mock()
        if db_type == "sqlite":
            db.get_current_data_version.side_effect = sqlalchemy.exc.OperationalError(
                "no such table", None, None
            )
        elif db_type == "mysql":
            db.get_current_data_version.side_effect = sqlalchemy.exc.ProgrammingError(
                "Table 'mlrun.data_versions' doesn't exist", None, None
            )

    assert services.api.initial_data._resolve_current_data_version(db, db_session) == 3
    # fill db
    db.create_project(
        db_session,
        mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name="project-name"),
        ),
    )
    assert services.api.initial_data._resolve_current_data_version(db, db_session) == 1
    services.api.initial_data.latest_data_version = original_latest_data_version


def test_add_default_hub_source_if_needed():
    db, db_session = _initialize_db_without_migrations()

    # Start with no hub source
    hub_source = db.get_hub_source(
        db_session,
        index=mlrun.common.schemas.hub.last_source_index,
        raise_on_not_found=False,
    )
    assert hub_source is None

    # Create the default hub source
    services.api.initial_data._add_default_hub_source_if_needed(db, db_session)
    hub_source = db.get_hub_source(
        db_session,
        index=mlrun.common.schemas.hub.last_source_index,
    )
    assert hub_source.source.spec.path == config.hub.default_source.url

    # Change the config and make sure the hub source is updated
    config.hub.default_source.url = "http://some-other-url"
    services.api.initial_data._add_default_hub_source_if_needed(db, db_session)
    hub_source = db.get_hub_source(
        db_session,
        index=mlrun.common.schemas.hub.last_source_index,
    )
    assert hub_source.source.spec.path == config.hub.default_source.url

    # Make sure the hub source is not updated if it already exists
    with unittest.mock.patch(
        "services.api.initial_data._update_default_hub_source"
    ) as update_default_hub_source:
        services.api.initial_data._add_default_hub_source_if_needed(db, db_session)
        assert update_default_hub_source.call_count == 0


def test_migrate_function_kind():
    db, db_session = _initialize_db_without_migrations()
    num_of_functions = 10
    chunk_size = 1

    # Insert multiple functions
    for fn_counter in range(num_of_functions):
        fn_name = f"name-{fn_counter}"
        _insert_function(db, db_session, fn_name)

    # Insert a function with None as kind
    fn_name_none_kind = "name-10"
    _insert_function(db, db_session, fn_name_none_kind, function_kind=None)

    # Migrate function kind
    services.api.initial_data._ensure_function_kind(
        db, db_session, chunk_size=chunk_size
    )

    # Verify the migration for the first set of functions
    for fn_counter in range(num_of_functions):
        fn_name = f"name-{fn_counter}"
        _verify_function_kind(db, db_session, fn_name, expected_kind="remote")

    # Verify the migration for the function with None as kind
    _verify_function_kind(db, db_session, fn_name_none_kind, expected_kind="")


def test_create_project_summaries():
    db, db_session = _initialize_db_without_migrations()

    # Create a project
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name="project-name"),
    )

    with unittest.mock.patch.object(db, "_append_project_summary"):
        db.create_project(db_session, project)

    # Migrate the project summaries
    services.api.initial_data._create_project_summaries(db, db_session)

    # Check that the project summary was migrated
    migrated_project_summary = db.get_project_summary(db_session, project.metadata.name)

    assert migrated_project_summary.name == project.metadata.name


@pytest.mark.parametrize(
    "scheduled_object_labels, schedule_labels, expected_labels",
    [
        (
            {"label1": "value1"},
            {"label2": "value2"},
            {"label1": "value1", "label2": "value2"},
        ),
        ({"label1": "value1"}, {}, {"label1": "value1"}),
        ({}, {"label2": "value2"}, {"label2": "value2"}),
        (
            {"label1": "value1", "label3": "value3"},
            {"label2": "value2"},
            {"label1": "value1", "label2": "value2", "label3": "value3"},
        ),
        (
            {"label1": "value1", "label2": "value3"},
            {"label2": "value2"},
            {"label1": "value1", "label2": "value3"},
        ),
        (None, {"label2": "value2"}, {"label2": "value2"}),
        ({"label1": "value1"}, None, {"label1": "value1"}),
        (None, None, None),
    ],
)
def test_align_schedule_labels(
    scheduled_object_labels, schedule_labels, expected_labels
):
    db, db_session = _initialize_db_without_migrations()

    # Create a schedule
    db.create_schedule(
        session=db_session,
        project="project-name",
        name="schedule-name",
        kind=mlrun.common.schemas.ScheduleKinds.job,
        cron_trigger=mlrun.common.schemas.ScheduleCronTrigger.from_crontab("* * * * 1"),
        concurrency_limit=1,
        scheduled_object={"task": {"metadata": {"labels": scheduled_object_labels}}},
        labels=schedule_labels,
    )

    # Align schedule.labels and schedule.scheduled_object.task.metadata.labels
    db.align_schedule_labels(db_session)

    # Get updated schedules
    migrated_schedules = db.list_schedules(db_session)

    # Convert list[LabelRecord] to dict
    migrated_schedules_dict = {
        label.name: label.value for label in migrated_schedules[0].labels
    }

    assert (
        migrated_schedules[0].scheduled_object["task"]["metadata"]["labels"]
        or {} == migrated_schedules_dict
        or {} == expected_labels
    )


def test_add_producer_uri_to_artifact():
    db, db_session = _initialize_db_without_migrations()
    num_of_artifacts = 10
    chunk_size = 1

    producer_uri = "some-proj/some-uid"

    for artifact_counter in range(num_of_artifacts):
        artifact_key = f"name-{artifact_counter}"
        _insert_artifact(
            db, db_session, artifact_key, f"{producer_uri}-{artifact_counter}"
        )

    # Create artifact when uri field is not exists in spec.producer
    _insert_artifact(db, db_session, f"name-{10}", None, with_uri=False)

    # Create artifact with producer_uri is None in spec.producer.uri
    _insert_artifact(db, db_session, f"name-{11}", None)

    # migrate the artifact producer_uri
    services.api.initial_data._add_producer_uri_to_artifact(
        db,
        db_session,
        chunk_size=chunk_size,
    )

    # Verify migrated producer_uri for artifacts with expected values
    for artifact_counter in range(num_of_artifacts):
        artifact_key = f"name-{artifact_counter}"
        _verify_artifact_producer_uri(
            db, db_session, artifact_key, f"{producer_uri}-{artifact_counter}"
        )

    # Verify producer_uri for the artifacts with None as URI in spec.producer
    _verify_artifact_producer_uri(db, db_session, "name-10", "")
    _verify_artifact_producer_uri(db, db_session, "name-11", "")


@pytest.mark.parametrize(
    "system_id_source, expected_system_id",
    [
        # when no system id is configured, a new random one should be generated
        ("random", None),
        # when a system id is set in mlconf, it should be used
        ("mlconf", "123"),
    ],
)
def test_init_system_id(
    system_id_source, expected_system_id, monkeypatch: pytest.MonkeyPatch
):
    if system_id_source == "mlconf":
        monkeypatch.setattr(
            mlrun.mlconf, framework.constants.SYSTEM_ID_KEY, expected_system_id
        )

    db, db_session = _initialize_db_without_migrations()

    # start with no system id
    system_id = db.get_system_id(db_session)
    assert system_id is None

    # initialize the system id
    services.api.initial_data._init_system_id(db_session)
    system_id = db.get_system_id(db_session)
    assert system_id is not None

    if system_id_source == "random":
        # ensure the generated id has the correct length
        assert len(system_id) == 6
        # ensure the generated id contains only alphanumeric characters
        assert all(char in string.ascii_lowercase + string.digits for char in system_id)
    else:
        assert system_id == expected_system_id

    assert mlrun.mlconf.system_id == system_id

    # ensure reinitialization does not change an existing system id
    services.api.initial_data._init_system_id(db_session)
    system_id_after_second_init = db.get_system_id(db_session)
    assert system_id_after_second_init == system_id


def test_ensure_latest_tag_for_artifacts():
    # This test verifies that the migration to ensure the "latest" tag is assigned correctly to artifacts works as
    # expected. The test creates a set of artifacts with different iteration numbers and tags:

    # 1. project1 + key1 + iteration 0 (run1) -> 2 tags (v1, v2)

    # 2. project1 + key1 + iteration 1 (run2) -> 1 tag (latest)
    # 3. project1 + key1 + iteration 2 (run2) -> 2 tags (v1, latest)
    # 4. project1 + key1 + iteration 3 (run2) -> 2 tags (v1, latest)

    # 5. project2 + key1 + iteration 0 -> 1 tag (latest)
    # 6. project2 + key2 + iteration 0 -> 1 tag (latest)

    # The test then deletes the "latest" tags from the second artifact and verifies that only 2 artifacts have the
    # "latest" tag left. After performing the migration, the test verifies that the correct artifacts are tagged as
    # "latest".

    db, db_session = _initialize_db_without_migrations()
    key1 = "key1"
    project1 = "project1"
    key2 = "key2"
    project2 = "project2"
    tree1 = "tree1"
    tree2 = "tree2"

    def generate_artifact(key, tree=None):
        artifact = {
            "metadata": {"key": key},
            "kind": "artifact",
        }
        if tree:
            artifact["metadata"]["tree"] = tree
        return artifact

    # Step 1: Create artifacts with different iteration numbers and tags

    # Create artifact for project1 + key1 + iteration 0 (run1) -> 3 tags (v1, v2, latest)
    artifact_1_uid = db.store_artifact(
        db_session,
        key=key1,
        project=project1,
        iter=0,
        artifact=generate_artifact(key1, tree1),
        tag="v1",
    )
    db.store_artifact(
        db_session,
        key=key1,
        project=project1,
        iter=0,
        artifact=generate_artifact(key1, tree1),
        tag="v2",
    )

    # Create 2 artifacts with hyperparameters, each will receive the 'latest' tag
    # and the 'latest' tag is removed from the artifact from the previous run (run1)

    # project1 + key1 + iteration 1 (run2) -> 1 tag (latest)
    artifact_2_uid = db.store_artifact(
        db_session,
        key=key1,
        project=project1,
        iter=1,
        artifact=generate_artifact(key1, tree2),
    )

    # project1 + key1 + iteration 2 (run2) -> 2 tags (v1, latest)
    artifact_3_uid = db.store_artifact(
        db_session,
        key=key1,
        project=project1,
        iter=2,
        artifact=generate_artifact(key1, tree2),
        tag="v1",
    )

    # project1 + key1 + iteration 3 (run2) -> 2 tags (v1, latest)
    artifact_4_uid = db.store_artifact(
        db_session,
        key=key1,
        project=project1,
        iter=3,
        artifact=generate_artifact(key1, tree2),
        tag="v1",
    )

    # project2 + key1 + iteration 0 -> 1 tag (latest)
    artifact_5_uid = db.store_artifact(
        db_session,
        key=key1,
        project=project2,
        iter=0,
        artifact=generate_artifact(key1),
    )

    # project2 + key2 + iteration 0 -> 1 tag (latest)
    artifact_6_uid = db.store_artifact(
        db_session,
        key=key2,
        project=project2,
        iter=0,
        artifact=generate_artifact(key2),
    )

    assert (
        artifact_1_uid
        != artifact_2_uid
        != artifact_3_uid
        != artifact_4_uid
        != artifact_5_uid
        != artifact_6_uid
    )

    # Step 2: List the artifacts for project1, key1, and the "latest" tag
    artifacts = db.list_artifacts(
        db_session, project=project1, name=key1, tag="latest", as_records=True
    )
    assert len(artifacts) == 3

    # Read the artifacts that were stored to get their IDs
    artifact2 = db.read_artifact(
        db_session, project=project1, key=key1, uid=artifact_2_uid, as_record=True
    )
    artifact3 = db.read_artifact(
        db_session, project=project1, key=key1, uid=artifact_3_uid, as_record=True
    )
    artifact4 = db.read_artifact(
        db_session, project=project1, key=key1, uid=artifact_4_uid, as_record=True
    )
    artifact_2_id = artifact2.id
    artifact_3_id = artifact3.id
    artifact_4_id = artifact4.id

    # Step 3: Delete the "latest" tags manually from the second artifact and the forth artifact
    # (artifact_2_id, artifact_4_id)
    db._delete(
        db_session,
        framework.db.sqldb.db.ArtifactV2.Tag,
        obj_id=artifact_2_id,
        name="latest",
    )
    db._delete(
        db_session,
        framework.db.sqldb.db.ArtifactV2.Tag,
        obj_id=artifact_4_id,
        name="latest",
    )
    db_session.flush()

    # Step 4: Assert that only one artifact has the "latest" tag left (artifact_3)
    artifacts = db.list_artifacts(
        db_session, project=project1, name=key1, tag="latest", as_records=True
    )
    assert len(artifacts) == 1
    assert artifacts[0].id == artifact_3_id

    # Step 5: Perform migration to ensure the "latest" tag is reassigned correctly
    services.api.initial_data._ensure_latest_tag_for_artifacts(db_session, chunk_size=1)

    # Step 6: Verify that after migration, the correct artifacts are tagged as "latest"
    artifacts = db.list_artifacts(
        db_session, project=project1, name=key1, tag="latest", as_records=True
    )
    assert (
        len(artifacts) == 3
    ), f"Expected 3 artifacts with latest tag, found {len(artifacts)}"

    # Verify that artifact from the previous run (run1) wasn't tagged as latest
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        db.read_artifact(db_session, project=project1, key=key1, tag="latest", iter=0)

    # Ensure the tag was created correctly for the second artifact
    artifact = db.read_artifact(
        db_session, project=project1, key=key1, tag="latest", iter=1, as_record=True
    )
    assert len(artifact.tags) == 1
    assert artifact.tags[0].name == "latest"
    assert artifact.tags[0].project == project1
    assert artifact.tags[0].obj_name == key1
    assert artifact.tags[0].obj_id == artifact_2_id


def _initialize_db_without_migrations() -> (
    tuple[framework.db.sqldb.db.SQLDB, sqlalchemy.orm.Session]
):
    dsn = "sqlite:///:memory:?check_same_thread=false"
    mlrun.mlconf.httpdb.dsn = dsn
    mlrun.common.db.sql_session._init_engine(dsn=dsn)
    framework.utils.singletons.db.initialize_db()
    db_session = mlrun.common.db.sql_session.create_session(dsn=dsn)
    db = framework.db.sqldb.db.SQLDB(dsn)
    db.initialize(db_session)
    framework.db.init_db()
    return db, db_session


def _insert_function(
    db, db_session, fn_name, function_kind: typing.Optional[str] = "remote"
):
    function_body = {
        "metadata": {"name": fn_name},
        "kind": function_kind,
        "status": {"state": "online"},
        "spec": {"description": "some_description"},
    }

    # Insert function via db
    db.store_function(db_session, function_body, fn_name)

    # Ensure the function is inserted the legacy way
    db_function, _ = db._get_function_db_object(db_session, fn_name)
    db_function.kind = None
    fn_struct = db_function.struct
    fn_struct["kind"] = function_kind
    db_function.struct = fn_struct
    db_session.add(db_function)
    db._commit(db_session, db_function)
    db_session.flush()

    # Verify the function was inserted correctly
    db_function, _ = db._get_function_db_object(db_session, fn_name)
    assert db_function.kind is None
    assert db_function.struct["kind"] == function_kind


def _verify_function_kind(db, db_session, fn_name, expected_kind):
    db_function, _ = db._get_function_db_object(db_session, fn_name)
    assert "kind" not in db_function.struct
    assert db_function.kind == expected_kind

    # Verify migration was stored correctly
    migrated_function = db.get_function(db_session, fn_name)
    assert migrated_function["kind"] == expected_kind


def _insert_artifact(db, db_session, artifact_key, artifact_uri=None, with_uri=True):
    artifact = {
        "metadata": {"key": artifact_key},
        "spec": {"producer": {"uri": artifact_uri} if with_uri else {}},
    }
    uid = db.store_artifact(db_session, key=artifact_key, artifact=artifact)

    # Legacy insert: set producer_uri to None
    db_artifact = db._query(
        db_session, framework.db.sqldb.db.ArtifactV2, uid=uid
    ).one_or_none()
    db_artifact.producer_uri = None
    db_session.add(db_artifact)
    db._commit(db_session, db_artifact)
    db_session.flush()

    # Ensure producer_uri is None after insertion
    db_artifact = db._query(
        db_session, framework.db.sqldb.db.ArtifactV2, uid=uid
    ).one_or_none()
    assert db_artifact.producer_uri is None
    return uid, artifact_key


def _verify_artifact_producer_uri(db, db_session, artifact_key, expected_uri):
    artifact = db._query(
        db_session, framework.db.sqldb.db.ArtifactV2, key=artifact_key
    ).one_or_none()
    assert artifact.producer_uri == expected_uri
