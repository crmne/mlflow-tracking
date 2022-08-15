# Docker multistage build to reduce image size
FROM python:3.10 AS build
RUN python -m venv /opt/venv
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install mlflow psycopg2 mysqlclient google-cloud-storage

FROM python:3.10-slim
COPY --from=build /opt/venv /opt/venv
COPY --from=build /usr/lib /usr/lib
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 5000/tcp
ENTRYPOINT [ "mlflow", "server", "--host", "0.0.0.0"]
CMD [ "--backend-store-uri", "sqlite:////mlruns/mlruns.db", \
      "--default-artifact-root", "/mlartifacts"]
