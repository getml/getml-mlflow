from __future__ import annotations

import inspect
import logging
from functools import cached_property, wraps
from inspect import BoundArguments
from logging import Logger
from typing import Any, Callable, Dict, Optional, OrderedDict, Sequence, TypeVar

import numpy
from getml import Pipeline
from getml.data import DataFrame, Subset
from getml.pipeline import Scores
from mlflow import MlflowClient
from mlflow.entities import Param, Run, Span
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
from typing_extensions import ParamSpec

from getml_mlflow.data.dataframelike import DataFrameLikeT
from getml_mlflow.logging.datacontainer import DataContainerLogger
from getml_mlflow.logging.numpy import NumpyLogger
from getml_mlflow.loggingconfiguration import (
    DataContainerLoggingConfiguration,
    FunctionLoggingConfiguration,
)

logger: Logger = logging.getLogger(__name__)

CallableArgsTypes = ParamSpec("CallableArgsTypes")
CallableReturnType = TypeVar("CallableReturnType")


class FunctionLogger:
    def __init__(
        self,
        mlflow_client: MlflowClient,
        run: Run,
        pipeline: Pipeline,
        *,
        function_logging_configuration: FunctionLoggingConfiguration = FunctionLoggingConfiguration(),
        data_container_logging_configuration: DataContainerLoggingConfiguration = DataContainerLoggingConfiguration(),
    ) -> None:
        self._mlflow_client: MlflowClient = mlflow_client
        self._run: Run = run
        self._pipeline: Pipeline = pipeline
        self._function_logging_configuration: FunctionLoggingConfiguration = (
            function_logging_configuration
        )
        self._data_container_logging_configuration: DataContainerLoggingConfiguration = data_container_logging_configuration

    def log(
        self, function: Callable[CallableArgsTypes, CallableReturnType]
    ) -> Callable[CallableArgsTypes, CallableReturnType]:
        @wraps(function)
        def wrapper(
            *args: CallableArgsTypes.args, **kwargs: CallableArgsTypes.kwargs
        ) -> CallableReturnType:
            bound_arguments: BoundArguments = inspect.signature(function).bind(
                *args, **kwargs
            )
            bound_arguments.apply_defaults()
            arguments: OrderedDict[str, Any] = bound_arguments.arguments

            span: Optional[Span] = self.log_trace_start(function.__name__, arguments)
            self.log_parameters(arguments)

            output: CallableReturnType = function(*args, **kwargs)

            self.log_return(output)
            self.log_trace_end(span, output)

            return output

        return wrapper

    def log_parameters(self, arguments: OrderedDict[str, Any]) -> None:
        if not self._function_logging_configuration.log_parameters:
            return

        parameters: Dict[str, Any] = {}

        for argument_name, argument_value in arguments.items():
            if argument_name == "self":
                # Don't log self.
                pass
            elif isinstance(argument_value, (DataFrameLikeT, Subset)):
                self._input_data_container_logger.log_data_container(
                    argument_value, [argument_name]
                )
            elif (
                isinstance(argument_value, Sequence)
                and all(
                    isinstance(element, DataFrameLikeT) for element in argument_value
                )
            ) or (
                isinstance(argument_value, Dict)
                and all(
                    isinstance(element, DataFrameLikeT)
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
            logging_configuration=self._data_container_logging_configuration,
        )

    @cached_property
    def _artifact_data_container_logger(self) -> DataContainerLogger:
        return DataContainerLogger.as_artifact(
            self._mlflow_client,
            self._run.info.run_id,
            logging_configuration=self._data_container_logging_configuration,
        )

    @cached_property
    def _numpy_logger(self) -> NumpyLogger:
        return NumpyLogger(self._mlflow_client, self._run.info.run_id)

    def log_return(self, output: Any) -> None:
        if not self._function_logging_configuration.log_return or output is None:
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
        elif isinstance(output, (Pipeline, Scores)):
            # Return values of type Pipeline and Scores are not logged.
            pass
        else:
            logger.info("Missing return logging for type '%s'", type(output))

    def log_trace_start(
        self, function_name: str, arguments: OrderedDict[str, Any]
    ) -> Optional[Span]:
        if not self._function_logging_configuration.log_as_trace:
            return None

        span: Span = self._mlflow_client.start_trace(
            function_name,
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
        if not self._function_logging_configuration.log_as_trace or span is None:
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
