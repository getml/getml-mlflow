import logging


def set_up():
    logger = logging.getLogger("getML")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="{asctime} {levelname} getML: {message}",
        style="{",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
