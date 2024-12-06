import logging
from src.getml.mlflow.autologging import autolog as _autolog
from mlflow.utils.autologging_utils import autologging_integration

FLAVOR_NAME = "getml"

_logger = logging.getLogger(__name__)


class _GetMLModelWrapper():
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
