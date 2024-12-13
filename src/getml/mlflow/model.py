import logging
from typing import Any, Dict

import pandas as pd

import getml
from getml.mlflow.autologging import autolog as _autolog
from mlflow import evaluate as _evaluate
from mlflow.pyfunc.model import PythonModel
from mlflow.utils.autologging_utils import autologging_integration

FLAVOR_NAME = "getml"

_logger = logging.getLogger(__name__)


class _GetMLModelWrapper(PythonModel):
    def __init__(self, getml_pipeline):
        self.getml_pipeline = getml_pipeline

    def get_raw_model(self):
        return self.getml_pipeline

    # FIXME: Where is implementation of this abstract base method?
    # def predict(
    #     self,
    #     context,
    #     model_input,
    #     params: Optional[Dict[str, Any]] = None,
    # )

    def predict(self, data: Dict[str, Any]):
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

        container = getml.data.Container(
            population=population, peripheral=peripheral_frames
        )

        return self.getml_pipeline.predict(container.full)

    def _validate_incoming_data(self, data):
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


def evaluate(
    model=None,
    data=None,
    *,
    model_type=None,
    targets=None,
    predictions=None,
    dataset_path=None,
    feature_names=None,
    evaluators=None,
    evaluator_config=None,
    custom_metrics=None,
    extra_metrics=None,
    custom_artifacts=None,
    validation_thresholds=None,
    baseline_model=None,
    env_manager="local",
    model_config=None,
    baseline_config=None,
    inference_params=None,
):
    print("wrapping evaluate")

    return _evaluate(
        model=model,
        data=data,
        model_type=model_type,
        targets=targets,
        predictions=predictions,
        dataset_path=dataset_path,
        feature_names=feature_names,
        evaluators=evaluators,
        evaluator_config=evaluator_config,
        custom_metrics=custom_metrics,
        extra_metrics=extra_metrics,
        custom_artifacts=custom_artifacts,
        validation_thresholds=validation_thresholds,
        baseline_model=baseline_model,
        env_manager=env_manager,
        model_config=model_config,
        baseline_config=baseline_config,
        inference_params=inference_params,
    )


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
