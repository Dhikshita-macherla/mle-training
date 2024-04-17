FROM python:3.9
WORKDIR /scripts
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U pytest
EXPOSE 1234
ENV NAME env
CMD pytest -v tests/functional_tests/ && pytest -v tests/unit_tests/ && python scripts/main.py data data/processed .artifacts/model .artifacts/scores
