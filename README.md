# getml-mlflow

## How to use

Prepare the Python environment.
```bash
$ uv venv
```

Install getml-mlflow via pip from pypi.
```bash
$ uv pip install getml-mlflow
```

Install latest getml-mlflow via pip from repository.
```bash
$ uv pip install "git+ssh://git@github.com/getml/getml-mlflow.git"
```

Run the mlflow server with its browser UI.
```bash
$ uv run mlflow ui
```

Open the mlflow UI in your browser.
```bash
$ open http://localhost:5000
```

### Documentation

See [User Guide](https://getml.com/dev/user_guide/concepts/mlflow_integration/)

See [API Reference](https://getml.com/dev/reference/mlflow/)

### Log via mlflow

To log information from getML pipelines and its `fit`, `score`, `predict` and `transform` methods into mlflow, you can activate the mlflow autologging capabilities.

```python
import getml_mlflow
getml_mlflow.autolog()
```

You can try this with our [demonstrational notebooks](https://github.com/getml/getml-demo/) and the [community variants](https://github.com/getml/getml-community/tree/main/demo-notebooks).

## Delete a deleted experiment

By deleting an experiment in the mlflow UI, the experiment is still preset in the aether...
Even when deleting the experiment via the mlflow CLI, the experiment is still present in the aether...

```bash
$ uv run mlflow experiments search --view all
Experiment Id       Name            Artifact Location
------------------  --------------  ------------------------------------
0                   Default         mlflow-artifacts:/0
888888888888888888  interstate94    mlflow-artifacts:/888888888888888888

$ uv run mlflow  experiments  delete -x 888888888888888888
Experiment with ID 888888888888888888 has been deleted.
```

Creating another experiment with the same name will result in the following error:
> RestException: RESOURCE_ALREADY_EXISTS: Experiment 'interstate94' already exists in deleted state. You can restore the experiment, or permanently delete the experiment from the .trash folder (under tracking server's root folder) in order to use this experiment name again.

You have to delete the experiment from the aether via
```bash
$ rm -rf mlruns/.trash/888888888888888888/
$ MLFLOW_TRACKING_URI="http://localhost:5000" uv run mlflow gc
```
