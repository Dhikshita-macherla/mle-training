def test_scoring_logic():
    try:
        from housePricePrediction import scoring_logic
    except Exception as e:
        assert False, (
            f"Error: { e.__str__() }. "
            " scoring_logic package is not \
                imported and installed correctly."
        )
