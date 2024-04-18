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
    pip install dist/*.whl && \
    pip install pandas numpy matplotlib scikit-learn pytest

CMD ["sh", "-c", "\
    pytest -v tests/functional_tests/ && \
    pytest -v tests/unit_tests/ && \
    python3 scripts/ingestion.py -h && \
    python3 scripts/ingestion.py data && \
    python3 scripts/train.py -h && \
    python3 scripts/train.py data/processed .artifacts/models && \
    python3 scripts/score.py -h && \
    python3 scripts/score.py data/processed .artifacts/models .artifacts/scores \
"]