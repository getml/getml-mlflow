import functools
import inspect
import logging
from typing import Any, Callable, Dict, Optional, OrderedDict, Sequence, Union

import getml
import numpy
from mlflow import MlflowClient
from mlflow.entities import Param, Run, Span
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager

from getml_mlflow.data.dataframelike import DataFrameLike
from getml_mlflow.logging.datacontainer import DataContainerLogger
from getml_mlflow.logging.numpy import NumpyLogger
from getml_mlflow.loggingconfiguration import LoggingConfiguration


class FunctionLogger:
    def __init__(
        self,
        mlflowclient: MlflowClient,
        run: Run,
        pipeline: getml.Pipeline,
        logging_configuration_function: LoggingConfiguration.Function,
        logging_configuration_data_container: LoggingConfiguration.DataContainer,
    ) -> None:
        self._mlflowclient: MlflowClient = mlflowclient
        self._run: Run = run
        self._pipeline: getml.Pipeline = pipeline
        self._logging_configuration_function: LoggingConfiguration.Function = (
            logging_configuration_function
        )
        self._logging_configuration_data_container: LoggingConfiguration.DataContainer = logging_configuration_data_container

    def log(self, function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapper(*args, **kwargs) -> Any:
            bound_arguments: inspect.BoundArguments = inspect.signature(function).bind(
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

        data_container_logger: Optional[DataContainerLogger] = None
        if any(
            isinstance(
                value,
                Union[
                    DataFrameLike,
                    getml.data.Subset,
                ],
            )
            or (
                isinstance(value, Sequence)
                and all(isinstance(element, DataFrameLike) for element in value)
            )
            or (
                isinstance(value, Dict)
                and all(
                    isinstance(element, DataFrameLike) for element in arguments.values()
                )
            )
            for value in arguments.values()
        ):
            data_container_logger = DataContainerLogger.as_input(
                self._mlflowclient,
                self._run.info.run_id,
                logging_configuration=self._logging_configuration_data_container,
            )

        parameters: Dict[str, Any] = {}

        for argument_name, argument_value in arguments.items():
            if argument_name == "self":
                continue

            if isinstance(argument_value, Union[DataFrameLike, getml.data.Subset]):
                assert data_container_logger
                data_container_logger.log_data_container(
                    argument_value, [argument_name]
                )
                continue

            if (
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
                assert data_container_logger
                data_container_logger.log_data_containers(argument_value, argument_name)
                continue

            parameters[argument_name] = str(argument_value)

        self._mlflowclient.log_batch(
            self._run.info.run_id,
            params=[Param(key, value) for key, value in parameters.items()],
        )

    def log_return(self, output: Any) -> None:
        if not self._logging_configuration_function.log_return or output is None:
            return

        if isinstance(output, getml.DataFrame):
            DataContainerLogger.as_artifact(
                self._mlflowclient,
                self._run.info.run_id,
                logging_configuration=self._logging_configuration_data_container,
            ).log_data_container(
                data_container=output,
                context="output",
            )
        elif isinstance(output, numpy.ndarray):
            NumpyLogger(
                self._mlflowclient, self._run.info.run_id
            ).log_ndarray_as_artifact(
                data=output,
                name="output",
                artifact_path="output",
            )
        elif isinstance(output, Union[getml.Pipeline, getml.pipeline.Scores]):
            # Return values of type getml.Pipeline and getml.pipeline.Scores are not logged.
            pass
        else:
            logging.getLogger("getML").info(
                "Missing return logging for type '{}'", type(output)
            )

    def log_trace_start(
        self, function: Callable[..., Any], arguments: OrderedDict[str, Any]
    ) -> Optional[Span]:
        if not self._logging_configuration_function.log_as_trace:
            return None

        span: Span = self._mlflowclient.start_trace(
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
        InMemoryTraceManager().get_instance().set_request_metadata(
            span.request_id,
            TraceMetadataKey().SOURCE_RUN,
            self._run.info.run_id,
        )
        return span

    def log_trace_end(self, span: Optional[Span], output: Any) -> None:
        if not self._logging_configuration_function.log_as_trace or span is None:
            return

        self._mlflowclient.end_trace(
            span.request_id,
            outputs={output.__class__.__name__: output},
            attributes={
                "pipeline": self._pipeline.id,
            },
        )
        self._mlflowclient.set_trace_tag(span.request_id, "pipeline", self._pipeline.id)
