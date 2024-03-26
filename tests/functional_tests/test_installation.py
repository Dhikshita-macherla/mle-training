def test_pkg_import():
    try:
        import mypackage
    except Exception as e:
        assert False, (
            f"Error: {e}. "
            " mypackage package is not \
                imported and installed correctly."
        )
