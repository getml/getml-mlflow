from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Optional

from dataclasses import dataclass, field

from mlflow import MlflowClient


@dataclass
class LoggingConfiguration:
    mlflow_client: MlflowClient = field(default_factory=MlflowClient)
    log_data_container_information: bool = True
    log_data_container_as_artifact: bool = True
    log_function_parameters: bool = True
    log_function_return: bool = True
    log_pipeline_parameters: bool = True
    log_pipeline_tags: bool = True
    log_pipeline_scores: bool = True
    log_pipeline_features: bool = True
    log_pipeline_columns: bool = True
    log_pipeline_targets: bool = True
    log_system_metrics: bool = True
    silent: bool = False
    create_runs: bool = True
    extra_tags: Optional[Dict[str, str]] = None
    getml_project_path: Optional[str] = None
