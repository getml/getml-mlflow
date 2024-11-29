import logging
import os
import pathlib
from typing import Any, Union

import yaml

import mlflow
from mlflow import pyfunc, set_tracking_uri, set_experiment, start_run
from src.getml.mlflow.autologging import autolog as _autolog
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import (
    get_total_file_size,
    write_to,
)
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "getml"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements(include_cloudpickle=False):
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("scikit-learn", module="sklearn")]
    if include_cloudpickle:
        pip_deps += [_get_pinned_requirement("cloudpickle")]

    return pip_deps


def get_default_conda_env(include_cloudpickle=False):
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements(include_cloudpickle))


def _ignore(pipeline_id: str, directory: str, files: list[str]):
    if "pipelines" in directory:
        return directory, [f for f in files if pipeline_id == f]
    return directory, files


def _copy_getml_engine_folders(getml_project_folder: pathlib.Path, pipeline_id: str, dst_path: str):
    import shutil

    dst_project_path = pathlib.Path(dst_path) / "projects"

    # copy data structure but what is really necessary
    shutil.copytree(
        src=os.path.join(str(getml_project_folder)),
        dst=dst_project_path,
        ignore=lambda directory, files: _ignore(pipeline_id, directory, files),
    )


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="getml"))
def save_model(
    getml_pipeline,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    settings: Union[dict[str, Any], None] = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    import getml

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)
    path = os.path.abspath(path)

    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    current_user_home_dir = pathlib.Path.home()

    getml_project_name = (
        settings.get("project_name", getml.project.name) if settings else getml.project.name
    )  # type: ignore
    if settings and (wd := settings.get("working_dir")):
        if not pathlib.Path(wd).exists():
            raise Exception(f"{wd} Working directory does not exists")
        getml_working_dir = pathlib.Path(wd)
    elif (wd := current_user_home_dir / ".getML").exists():
        getml_working_dir = wd
    else:
        raise Exception("No default getML project directory")

    assert getml_project_name
    if not (getml_project_folder := getml_working_dir / "projects" / getml_project_name).exists():
        raise Exception(f"{getml_project_folder} does not exists")

    if mlflow_model is None:
        mlflow_model = Model()

    if metadata is not None:
        mlflow_model.metadata = metadata

    if settings is None:
        settings = {}

    settings["getml_project_name"] = getml_project_name
    settings["pipeline_id"] = getml_pipeline.id

    with open(os.path.join(path, "getml.yaml"), "w") as settings_file:
        yaml.safe_dump(settings, stream=settings_file)

    _copy_getml_engine_folders(getml_project_folder, getml_pipeline.id, path)
    # copy files from project folder
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.getml",
        data=None,
        conda_env=_CONDA_ENV_FILE_NAME,
        python=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        getml_version=getml.__version__,
        data=path,
        code=code_dir_subpath,
    )

    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    getml_pipeline,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    **kwargs,
):
    """Log an H2O model as an MLflow artifact for the current run.

    Args:
        getml_pipeline: getML pipeline to be saved.
        artifact_path: Run-relative artifact path.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        registered_model_name: If given, create a model version under
            ``registered_model_name``, also creating a registered model if one
            with the given name does not exist.
        signature: {{ signature }}
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata:  {{ metadata }}
        kwargs: kwargs to pass to ``getml.save_model`` method.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.

    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.getml,
        registered_model_name=registered_model_name,
        getml_pipeline=getml_pipeline,
        conda_env=conda_env,
        code_paths=code_paths,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        **kwargs,
    )


class _GetMLModelWrapper:
    def __init__(self, getml_pipeline):
        self.getml_pipeline = getml_pipeline

    def get_raw_model(self):
        return self.getml_pipeline

    def predict(self, data):
        import getml

        self._validate_incoming_data(data)
        roles = self._extract_roles_from_data_model()

        population = getml.data.DataFrame.from_pandas(
            data["population"], name="population", roles=roles["population"]
        )

        peripheral_frames = {}
        for name, peripheral_df in data["peripheral"].items():
            peripheral_frames[name] = getml.data.DataFrame.from_pandas(
                peripheral_df, name=name, roles=roles["peripherals"][name]
            )

        container = getml.data.Container(population=population, peripheral=peripheral_frames)

        return self.getml_pipeline.predict(container.full)

    def _validate_incoming_data(self, data):
        import pandas as pd

        assert "population" in data
        assert "peripheral" in data
        assert isinstance(data["population"], pd.DataFrame)
        assert isinstance(data["peripheral"], dict)

        peripheral_names_in_data = []

        for name, df in data["peripheral"].items():
            assert isinstance(df, pd.DataFrame)
            peripheral_names_in_data.append(name)

        for peripheral_table in self.getml_pipeline.data_model.population.children:
            if peripheral_table.name not in peripheral_names_in_data:
                raise Exception(
                    f"Peripheral table '{peripheral_table.name}' is missing in the data"
                )

    def _extract_roles_from_data_model(self):
        roles = {}
        roles["peripherals"] = {}
        roles["population"] = self.getml_pipeline.data_model.population.roles

        for peripheral in self.getml_pipeline.data_model.population.children:
            roles["peripherals"][peripheral.name] = peripheral.roles

        return roles


def _load_model(path):
    import shutil

    import getml


    with open(os.path.join(path, "getml.yaml")) as f:
        getml_settings = yaml.safe_load(f.read())

    getml_project_name = getml_settings["getml_project_name"]
    getml_pipeline_id = getml_settings["pipeline_id"]
    current_user_home_dir = pathlib.Path.home()
    getml_project_path = current_user_home_dir / ".getML" / "projects" / getml_project_name
    shutil.copytree(
        src=os.path.join(path, "projects"),
        dst=str(getml_project_path),
        dirs_exist_ok=True,
    )
    getml.set_project(getml_project_name)

    return getml.pipeline.load(getml_pipeline_id)


def _load_pyfunc(path):
    """Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``h2o`` flavor.

    """
    return _GetMLModelWrapper(_load_model(path))


def load_model(model_uri, dst_path=None):
    """Load an H2O model from a local file (if ``run_id`` is ``None``) or a run.

    This function expects there is an H2O instance initialised with ``h2o.init``.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
            artifact-locations>`_.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.

    Returns:
        An `H2OEstimator model object
        <http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html#models>`_.

    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    # Flavor configurations for models saved in MLflow version <= 0.8.0 may not contain a
    # `data` key; in this case, we assume the model artifact path to be `model.h2o`
    getml_model_file_path = os.path.join(local_model_path)
    return _load_model(path=getml_model_file_path)

@autologging_integration(FLAVOR_NAME)
def autolog(
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    max_tuning_runs=5,
    log_post_training_metrics=True,
):
    return _autolog(
        flavor_name=FLAVOR_NAME,
        log_input_examples=log_input_examples,
        log_model_signatures=log_model_signatures,
        log_models=log_models,
        log_datasets=log_datasets,
        log_post_training_metrics=log_post_training_metrics,
    )
