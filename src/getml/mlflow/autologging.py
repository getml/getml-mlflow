import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.utils.autologging_utils import safe_patch
from mlflow.utils.autologging_utils.client import MlflowAutologgingQueueingClient
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT

_logger = logging.getLogger(__name__)

@dataclass
class LogInfo:
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)


def autolog(
    flavor_name,
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
    flavor_name = "getml"
    from dataclasses import fields, is_dataclass

    import getml

    def _patch_pipeline_method(flavor_name, class_def, func_name, patched_fn, manage_run):
        safe_patch(
            flavor_name,
            class_def,
            func_name,
            patched_fn,
            manage_run=manage_run,
        )

    def _extract_pipeline_informations(getml_pipeline: getml.Pipeline) -> LogInfo:
        params = (
            "preprocessors",
            "feature_learners",
            "feature_selectors",
            "predictors",
            "share_selected_features",
        )
        pipeline_informations = {}

        for parameter_name in params:
            values = getattr(getml_pipeline, parameter_name)
            if isinstance(values, list):
                for v in values:
                    if is_dataclass(v):
                        name = v.__class__.__name__
                        for field in fields(v):
                            field_value = getattr(v, field.name)
                            if isinstance(field_value, (frozenset, set)):
                                field_value = json.dumps(list(field_value))
                            elif isinstance(field_value, getml.feature_learning.FastProp):
                                field_value = field_value.__class__.__name__
                            elif not isinstance(field_value, str):
                                field_value = json.dumps(field_value)

                            pipeline_informations[f"{parameter_name}.{name}.{field.name}"] = (
                                field_value
                            )
            elif isinstance(values, str):
                pipeline_informations[parameter_name] = values
            else:
               value_name = values.__class__.__name__
               pipeline_informations[parameter_name] = value_name
        tags = [str(t) for t in getml_pipeline.tags]
        return LogInfo(params=pipeline_informations, tags=dict(zip(tags, tags)))

    def _extract_fitted_pipeline_informations(getml_pipeline: getml.Pipeline) -> LogInfo:
        params = {
            "pipeline_id": getml_pipeline.id,
        }

        metrics = {}

        scores = getml_pipeline.scores

        if getml_pipeline.is_classification:
            metrics["train_auc"] = round(scores.auc,2)
            metrics["train_accuracy"] = round(scores.accuracy, 2)
            metrics["train_cross_entropy"] = round(scores.cross_entropy, 4)

        if getml_pipeline.is_regression:
            metrics["train_mae"] = scores.mae
            metrics["train_rmse"] = scores.rmse
            metrics["train_rsquared"] = round(scores.rsquared, 2)

        # for feature in getml_pipeline.features:
        #     metrics[f"{feature.name}.importance"] = json.dumps(feature.importance)
        #     metrics[f"{feature.name}.correlation"] = json.dumps(feature.correlation)

        # if len(getml_pipeline.targets) == 1:
        #     metrics["targets"] = getml_pipeline.targets[0]
        # else:
        #     for i, t in enumerate(getml_pipeline.targets):
        #         metrics[f"targets.{i}"] = t
        return LogInfo(
            params=params,
            metrics=metrics,
        )

    def _collect_available_engine_metrics() -> dict:
        import requests

        engine_metrics = {
            "engine_cpu_usage_per_virtual_core_in_pct": "http://localhost:1709/getcpuusage/",
            "memory_usage_in_pct": "http://localhost:1709/getmemoryusage/",
        }
        engine_metrics_to_be_tracked = {}
        for metric_name, metric_url in engine_metrics.items():
            if requests.get(metric_url).ok:
                engine_metrics_to_be_tracked[metric_name] = metric_url
        return engine_metrics_to_be_tracked

    def _extract_engine_system_metrics(
        autologging_client: MlflowAutologgingQueueingClient,
        run_id: str,
        stop_event: threading.Event,
        engine_metrics_to_be_tracked: dict,
    ) -> None:
        import numpy as np
        import requests

        step = 0
        collected_metrics_data = {}
        while not stop_event.is_set():
            for metric_name, metric_url in engine_metrics_to_be_tracked.items():
                collected_metrics_data[metric_name] = np.round(
                    requests.get(metric_url).json()["data"][0][-1], 2
                )
            autologging_client.log_metrics(
                run_id=run_id,
                metrics=collected_metrics_data,
                step=step,
            )
            step += 1
            stop_event.wait(1)

    def patched_fit_mlflow(original, self: getml.Pipeline, *args, **kwargs) -> getml.pipeline.Pipeline:
        autologging_client = MlflowAutologgingQueueingClient()
        assert (active_run := mlflow.active_run())
        run_id = active_run.info.run_id

        engine_metrics_to_be_tracked = _log_pretraining_metadata(autologging_client, self, run_id, *args)
        if engine_metrics_to_be_tracked:
            stop_event = threading.Event()
            metrics_thread = threading.Thread(
                target=_extract_engine_system_metrics,
                args=(autologging_client, run_id, stop_event, engine_metrics_to_be_tracked),
            )
            metrics_thread.start()
        else:
            print(
                "Engine metrics are not available. Please upgrade to the Enterprise edition. "
            )

        fit_output = original(self, *args, **kwargs)

        if engine_metrics_to_be_tracked:
            stop_event.set()
            metrics_thread.join()

        fitted_pipeline_log_info = _extract_fitted_pipeline_informations(self)
        autologging_client.log_metrics(
            run_id=run_id,
            metrics=fitted_pipeline_log_info.metrics,
        )

        autologging_client.flush(synchronous=True)
        return fit_output

    def patched_score_method(original, self: getml.Pipeline, *args, **kwargs) -> getml.pipeline.Scores:

        target = self.data_model.population.roles.target[0]
        pop_df = args[0].population.to_pandas()
        pop_df["predictions"] = self.predict(*args)
        pop_df['predictions'] = pop_df.round({'predictions': 0})['predictions'].astype(bool)
        pop_df[target] = pop_df[target].astype(bool)

        mlflow.evaluate(
            data = pop_df,
            targets=target,
            predictions="predictions",
            model_type=["regressor" if self.is_regression else "classifier"][0],
            evaluators=["default"],
        )
        return original(self, *args, **kwargs)

    def _log_pretraining_metadata(autologging_client: MlflowAutologgingQueueingClient,
                                  self: getml.Pipeline,
                                  run_id: str,
                                  *args
                                  ) -> dict:

        pipeline_log_info = _extract_pipeline_informations(self)
        autologging_client.log_params(
            run_id=run_id,
            params=pipeline_log_info.params,
        )
        if tags := pipeline_log_info.tags:
            autologging_client.set_tags(run_id=run_id, tags=tags)

        engine_metrics_to_be_tracked = _collect_available_engine_metrics()

        if log_datasets:
            try:
                datasets = []
                population_dataset: PandasDataset = mlflow.data.from_pandas(args[0].population.to_pandas(), name = args[0].population.base.name)
                tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value='Population')]
                datasets.append(DatasetInput(dataset=population_dataset._to_mlflow_entity(), tags=tags))

                for name, peripheral in args[0].peripheral.items():
                    tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value='Peripheral')]
                    peripheral_dataset: PandasDataset = mlflow.data.from_pandas(peripheral.to_pandas(), name = name)
                    datasets.append(DatasetInput(dataset=peripheral_dataset._to_mlflow_entity(), tags=tags))

                autologging_client.log_inputs(
                        run_id=run_id, datasets=datasets
                    )

            except Exception as e:
                _logger.warning(
                    "Failed to log training dataset information to MLflow Tracking. Reason: %s", e
                )
        return engine_metrics_to_be_tracked

    _patch_pipeline_method(
        flavor_name=flavor_name,
        class_def=getml.pipeline.Pipeline,
        func_name="fit",
        patched_fn=patched_fit_mlflow,
        manage_run=True,
    )

    _patch_pipeline_method(
        flavor_name=flavor_name,
        class_def=getml.pipeline.Pipeline,
        func_name="score",
        patched_fn=patched_score_method,
        manage_run=True,
    )
