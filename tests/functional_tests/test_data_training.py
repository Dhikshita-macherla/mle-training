def test_data_training():
    try:
        from housePricePrediction import data_training
    except Exception as e:
        assert False, (
            f"Error: { e.__str__() }. "
            " data_training package is not \
                imported and installed correctly."
        )
