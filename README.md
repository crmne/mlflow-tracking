# MLFlow Tracking Server

This repository neatly packages the MLFLow Tracking Server in a Docker container.

## Ports

MLFlow Tracking Server runs by default on port 5000. To port forward to `localhost:5000`:

    docker run crmne/mlflow-tracking -p 5000:5000

## Volumes

It is recommended that you mount `/mlruns` and `/mlartifacts` to persistent storage, e.g.:

    docker run crmne/mlflow-tracking -p 5000:5000 -v /mnt/mlflow/mlruns:/mlruns -v /mnt/mlflow/mlartifacts:/mlartifacts

## Runs and Artifacts

By default, this container will save the runs in `/mlruns/mlruns.db` and the artifacts in `/mlartifacts`,
but you can change it by appending the `--backend-store-uri` and `--default-artifact-root` options respectively for `mlflow server` to your `docker run`. This will allow you to log the runs to files or [any database supported by SQLAlchemy][db], and artifacts to [many cloud and network storage services][store]. Example:

    docker run crmne/mlflow-tracking -p 5000:5000 --backend-store-uri mysql://scott:tiger@localhost/mlflow --default-artifact-root s3://my-mlflow-bucket/

More information at https://mlflow.org/docs/latest/tracking.html#mlflow-tracking-servers

## Test your tracking server

`test.py` contains an example model to test MLFlow Tracking Server.

1. Run MLFlow Tracking Server

        docker run crmne/mlflow-tracking -p 5000:5000 -d

2. Install Pipenv (if you don't have it)

        pip install pipenv

3. Install dependencies

        pipenv install

4. Run example model

        pipenv run python test.py

[db]: https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls
[store]: https://mlflow.org/docs/latest/tracking.html#id10

