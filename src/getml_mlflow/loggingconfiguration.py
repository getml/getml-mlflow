from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Optional

from dataclasses import dataclass, field

from mlflow import MlflowClient


@dataclass
class LoggingConfiguration:
    @dataclass
    class DataContainer:
        log_information: bool = True
        log_as_artifact: bool = True

    @dataclass
    class Function:
        log_parameters: bool = True
        log_return: bool = True
        log_as_trace: bool = True

    @dataclass
    class Pipeline:
        log_parameters: bool = True
        log_tags: bool = True
        log_scores: bool = True
        log_features: bool = True
        log_columns: bool = True
        log_targets: bool = True
        log_data_model: bool = True
        log_as_artifact: bool = True

    mlflow_client: MlflowClient = field(default_factory=MlflowClient)
    data_container: DataContainer = field(default_factory=DataContainer)
    function: Function = field(default_factory=Function)
    pipeline: Pipeline = field(default_factory=Pipeline)

    log_system_metrics: bool = True
    silent: bool = False
    create_runs: bool = True
    extra_tags: Optional[Dict[str, str]] = None
    getml_project_path: Optional[str] = None
