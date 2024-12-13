# getml-mlflow

## How to use

Prepare the Python environment.
```bash
$ uv venv
```

Install getml-mlflow via pip from branch.
```bash
$ uv pip install "git+ssh://git@github.com/getml/getml-mlflow.git@4-consolidate-evaluate-extra"
```

Run the mlflow server with its browser UI.
```bash
$ uv pip install mlflow
$ uv run mlflow ui -h 0.0.0.0 --dev
```

Open the mlflow UI in your browser.
```bash
$ open http://localhost:5000
```

Run jupyter lab with to experiment with getml and mlflow.
```bash
$ uv pip install jupyter
$ uv run jupyter-lab --ip=0.0.0.0
```

This should automatically open jupyter lab in your browser.

Alternatively, you can open jupyter lab in your browser by copying the url with the token from the console output.
```bash
$ open http://localhost:8888/lab?token=ffffffffffffffffffffffffffffffffffffffffffffffff
```

### Run the example

* Download the example notebook [interstate94.ipynb](/interstate94.ipynb) and add it to the jupyter lab.
* Run the notebook.
* Check the mlflow UI for gathered information of the experiment.


## Delete a deleted experiment

By deleting an experiment in the mlflow UI, the experiement is still preset in the aether...
Even when deleting the experiment via the mlflow CLI, the experiment is still present in the aether...

```bash
$ uv run mlflow  experiments search
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
MLFLOW_TRACKING_URI="http://localhost:5000" uv run mlflow gc
```
