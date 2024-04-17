FROM python:3.9
WORKDIR /scripts
COPY . .

RUN apt-get update && \
    apt-get install -y pytest && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
EXPOSE 1234
ENV NAME mle-dev
CMD pytest -v tests/functional_tests/ && pytest -v tests/unit_tests/ && python scripts/main.py data data/processed .artifacts/model .artifacts/scores
