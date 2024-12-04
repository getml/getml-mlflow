import os
import pathlib
import yaml

from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.requirements_utils import _get_pinned_requirement


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

