from __future__ import annotations

import inspect
import logging
from functools import cached_property, wraps
from inspect import BoundArguments
from logging import Logger
from typing import Any, Callable, Dict, Optional, OrderedDict, Sequence, Union

import numpy
from getml import Pipeline
from getml.data import DataFrame, Subset
from getml.pipeline import Scores
from mlflow import MlflowClient
from mlflow.entities import Param, Run, Span
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager

from getml_mlflow.data.dataframelike import DataFrameLike
from getml_mlflow.logging.datacontainer import DataContainerLogger
from getml_mlflow.logging.numpy import NumpyLogger
from getml_mlflow.loggingconfiguration import LoggingConfiguration

logger: Logger = logging.getLogger(__name__)


class FunctionLogger:
    def __init__(
        self,
        mlflow_client: MlflowClient,
        run: Run,
        pipeline: Pipeline,
        *,
        logging_configuration_function: LoggingConfiguration.Function = LoggingConfiguration.Function(),
        logging_configuration_data_container: LoggingConfiguration.DataContainer = LoggingConfiguration.DataContainer(),
    ) -> None:
        self._mlflow_client: MlflowClient = mlflow_client
        self._run: Run = run
        self._pipeline: Pipeline = pipeline
        self._logging_configuration_function: LoggingConfiguration.Function = (
            logging_configuration_function
        )
        self._logging_configuration_data_container: LoggingConfiguration.DataContainer = logging_configuration_data_container

    def log(self, function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapper(*args, **kwargs) -> Any:
            bound_arguments: BoundArguments = inspect.signature(function).bind(
                *args, **kwargs
            )
            bound_arguments.apply_defaults()
            arguments: OrderedDict[str, Any] = bound_arguments.arguments

            span: Optional[Span] = self.log_trace_start(function, arguments)
            self.log_parameters(arguments)

            output: Any = function(*args, **kwargs)

            self.log_return(output)
            self.log_trace_end(span, output)

            return output

        return wrapper

    def log_parameters(self, arguments: OrderedDict[str, Any]) -> None:
        if not self._logging_configuration_function.log_parameters:
            return

        parameters: Dict[str, Any] = {}

        for argument_name, argument_value in arguments.items():
            if argument_name == "self":
                # Don't log self.
                pass
            elif isinstance(argument_value, Union[DataFrameLike, Subset]):
                self._input_data_container_logger.log_data_container(
                    argument_value, [argument_name]
                )
            elif (
                isinstance(argument_value, Sequence)
                and all(
                    isinstance(element, DataFrameLike) for element in argument_value
                )
            ) or (
                isinstance(argument_value, Dict)
                and all(
                    isinstance(element, DataFrameLike)
                    for element in argument_value.values()
                )
            ):
                self._input_data_container_logger.log_data_containers(
                    argument_value, argument_name
                )
            else:
                parameters[argument_name] = str(argument_value)

        self._mlflow_client.log_batch(
            self._run.info.run_id,
            params=[Param(key, value) for key, value in parameters.items()],
        )

    @cached_property
    def _input_data_container_logger(self) -> DataContainerLogger:
        return DataContainerLogger.as_input(
            self._mlflow_client,
            self._run.info.run_id,
            logging_configuration=self._logging_configuration_data_container,
        )

    @cached_property
    def _artifact_data_container_logger(self) -> DataContainerLogger:
        return DataContainerLogger.as_artifact(
            self._mlflow_client,
            self._run.info.run_id,
            logging_configuration=self._logging_configuration_data_container,
        )

    @cached_property
    def _numpy_logger(self) -> NumpyLogger:
        return NumpyLogger(self._mlflow_client, self._run.info.run_id)

    def log_return(self, output: Any) -> None:
        if not self._logging_configuration_function.log_return or output is None:
            return

        if isinstance(output, DataFrame):
            self._artifact_data_container_logger.log_data_container(
                data_container=output,
                context="output",
            )
        elif isinstance(output, numpy.ndarray):
            self._numpy_logger.log_ndarray_as_artifact(
                data=output,
                name="output",
                artifact_path="output",
            )
        elif isinstance(output, Union[Pipeline, Scores]):
            # Return values of type Pipeline and Scores are not logged.
            pass
        else:
            logger.info("Missing return logging for type '%s'", type(output))

    def log_trace_start(
        self, function: Callable[..., Any], arguments: OrderedDict[str, Any]
    ) -> Optional[Span]:
        if not self._logging_configuration_function.log_as_trace:
            return None

        span: Span = self._mlflow_client.start_trace(
            function.__name__,
            inputs=arguments,
            experiment_id=self._run.info.experiment_id,
            attributes={
                "pipeline": self._pipeline.id,
                "run": self._run.info.run_id,
            },
            tags={
                "pipeline": self._pipeline.id,
                "run": self._run.info.run_id,
            },
        )
        InMemoryTraceManager.get_instance().set_request_metadata(
            span.request_id,
            TraceMetadataKey.SOURCE_RUN,
            self._run.info.run_id,
        )
        return span

    def log_trace_end(self, span: Optional[Span], output: Any) -> None:
        if not self._logging_configuration_function.log_as_trace or span is None:
            return

        self._mlflow_client.end_trace(
            span.request_id,
            outputs={output.__class__.__name__: output},
            attributes={
                "pipeline": self._pipeline.id,
            },
        )
        self._mlflow_client.set_trace_tag(
            span.request_id, "pipeline", self._pipeline.id
        )
