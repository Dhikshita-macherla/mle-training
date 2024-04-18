FROM continuumio/miniconda3:latest

COPY deploy/conda/env.yml .
RUN conda env create -f env.yml
RUN echo "source activate mle-dev" > ~/.bashrc
ENV path /opt/conda/envs/mle-dev/bin:$PATH

COPY . /mle-training/
WORKDIR /mle-training/


RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get install -y python3

RUN pip install --upgrade setuptools && \
    pip install --upgrade build && \
    python3 -m build && \
    pip install dist/*.whl --force-reinstall && \
    pip install pandas numpy matplotlib scikit-learn pytest

CMD ["sh", "-c", "\
    mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5009 & \
    pytest -v tests/functional_tests/ && \
    pytest -v tests/unit_tests/ && \
    python3 scripts/main.py data data/processed .artifacts/model .artifacts/scores \
"]