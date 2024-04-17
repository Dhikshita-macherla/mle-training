FROM python:3.9
WORKDIR /
COPY . .

RUN pip install --no-cache-dir build
RUN python -m build
RUN pip install --no-cache-dir pytest
RUN pip install --no-cache-dir dist/*.whl

ENV NAME mle-dev
CMD ["sh", "-c", "pytest -v tests/functional_tests/ && \
                  pytest -v tests/unit_tests/ && \
                  python scripts/ingestion.py -h && \
                  python scripts/ingestion.py data && \
                  python scripts/train.py -h && \
                  python scripts/train.py data/processed .artifacts/models && \
                  python scripts/score.py -h && \
                  python scripts/score.py data/processed .artifacts/models .artifacts/scores"]
