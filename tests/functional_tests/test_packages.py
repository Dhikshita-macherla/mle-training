def test_data_ingestion_import():
    try:
        from housePricePrediction import data_ingestion
    except Exception as e:
        assert False, (
            f"Error: { e.__str__() }. "
            " data_ingestion package is not \
                imported and installed correctly."
        )

def test_data_training():
    try:
        from housePricePrediction import data_training
    except Exception as e:
        assert False, (
            f"Error: { e.__str__() }. "
            " data_training package is not \
                imported and installed correctly."
        )

def test_scoring_logic():
    try:
        from housePricePrediction import scoring_logic
    except Exception as e:
        assert False, (
            f"Error: { e.__str__() }. "
            " scoring_logic package is not \
                imported and installed correctly."
        )
