FROM python:3.7

RUN pip install mlflow

EXPOSE 5000/tcp

ENTRYPOINT [ "mlflow", "server", "--host", "0.0.0.0"]
CMD [ "--backend-store-uri", "sqlite:////mlruns/mlruns.db", \
      "--default-artifact-root", "/mlartifacts"]