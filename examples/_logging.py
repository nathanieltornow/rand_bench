import logging


def setup_logger():
    logger = logging.getLogger("qpu_bench")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logger
