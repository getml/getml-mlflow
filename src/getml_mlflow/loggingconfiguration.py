from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from mlflow import MlflowClient


@dataclass
class DataContainerLoggingConfiguration:
    log_information: bool = True
    log_as_artifact: bool = True


@dataclass
class FunctionLoggingConfiguration:
    log_parameters: bool = True
    log_return: bool = True
    log_as_trace: bool = True


@dataclass
class PipelineLoggingConfiguration:
    log_parameters: bool = True
    log_tags: bool = True
    log_scores: bool = True
    log_features: bool = True
    log_columns: bool = True
    log_targets: bool = True
    log_data_model: bool = True
    log_as_artifact: bool = True


@dataclass
class GeneralLoggingConfiguration:
    mlflow_client: MlflowClient = field(default_factory=MlflowClient)
    log_system_metrics: bool = True
    silent: bool = False
    create_runs: bool = True
    extra_tags: Optional[Dict[str, str]] = None
    getml_project_path: Optional[Path] = None


@dataclass
class LoggingConfiguration:
    general: GeneralLoggingConfiguration = field(
        default_factory=GeneralLoggingConfiguration
    )
    data_container: DataContainerLoggingConfiguration = field(
        default_factory=DataContainerLoggingConfiguration
    )
    function: FunctionLoggingConfiguration = field(
        default_factory=FunctionLoggingConfiguration
    )
    pipeline: PipelineLoggingConfiguration = field(
        default_factory=PipelineLoggingConfiguration
    )
