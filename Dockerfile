FROM python:3.9
WORKDIR /scripts
COPY . .

RUN apt-get install -y pytest
EXPOSE 1234
ENV NAME env
CMD pytest -v tests/functional_tests/ && pytest -v tests/unit_tests/ && python scripts/main.py data data/processed .artifacts/model .artifacts/scores
