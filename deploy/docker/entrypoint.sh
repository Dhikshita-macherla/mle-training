mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5008 & \
pytest -v tests/functional_tests/ && \
pytest -v tests/unit_tests/ && \
python3 scripts/main.py data data/processed .artifacts/model .artifacts/scores && \
sleep 100