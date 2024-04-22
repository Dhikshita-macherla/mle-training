# Stage 1: Build Stage
FROM continuumio/miniconda3:latest as builder
RUN groupadd -r mle-group && useradd -r -g mle-group mle-user

COPY deploy/conda/env.yml .
RUN conda env create -f env.yml
RUN echo "source activate mle-dev" > ~/.bashrc
ENV PATH /opt/conda/envs/mle-dev/bin:$PATH
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /mle-training/
WORKDIR /mle-training/
RUN apt-get update && apt-get install -y python3-pip && apt-get install -y python3
RUN python3 -m build

# Stage 2: Final Image
FROM continuumio/miniconda3:latest
ENV PATH /opt/conda/envs/mle-dev/bin:$PATH
COPY --from=builder /opt/conda/envs/mle-dev /opt/conda/envs/mle-dev
COPY --from=builder /mle-training /mle-training
USER mle-user
WORKDIR /mle-training
ENV MLFLOW_TRACKING_URI=http://localhost:5008
EXPOSE 5008
CMD ["sh", "-c", "\
    mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5008 & \
    pytest -v tests/functional_tests/ && \
    pytest -v tests/unit_tests/ && \
    python3 scripts/main.py data data/processed .artifacts/model .artifacts/scores \
"]
