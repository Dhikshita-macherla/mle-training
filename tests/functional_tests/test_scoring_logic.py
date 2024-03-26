def test_pkg_import():
    try:
        import housePricePrediction.scoring_logic
    except Exception as e:
        assert False, (
            f"Error: { e.__str__() }. "
            " housePricePrediction package is not \
                imported and installed correctly."
        )
