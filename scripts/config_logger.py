import logging
import logging.config


def configure_logger(
        logger=None, cfg=None, log_file=None, console=True,
        log_level=logging.DEBUG):
    logger = logging.getLogger()
    LOGGING_DEFAULT_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {"format": "%(message)s"},
        },
        "root": {"level": "DEBUG"},
    }

    if not cfg:
        logging.config.dictConfig(LOGGING_DEFAULT_CONFIG)
    else:
        logging.config.dictConfig(cfg)

    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s -%(message)s"
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()
            sh.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s -%(message)s"
            )
            sh.setFormatter(formatter)
            logger.addHandler(sh)

    return logger
