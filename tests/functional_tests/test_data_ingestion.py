def test_data_ingestion_import():
    try:
        from housePricePrediction import data_ingestion
    except Exception as e:
        assert False, (
            f"Error: { e.__str__() }. "
            " housePricePrediction package is not \
                imported and installed correctly."
        )
