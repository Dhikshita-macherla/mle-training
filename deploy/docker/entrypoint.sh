pytest -v tests/functional_tests/ && \
pytest -v tests/unit_tests/ && \
python3 scripts/main.py data data/processed .artifacts/model .artifacts/scores && \
sleep 100