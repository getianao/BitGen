# my_logger.py
import logging
from logging.handlers import RotatingFileHandler
import colorlog
import os


# Validate and set the logging level
valid_levels = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET
}

class MyLogger:
    _logger_instance = None

    @staticmethod
    def get_logger(
        name: str = "AppLogger",
        log_file: str = None,
        max_bytes: int = 10485760,
        backup_count: int = 5,
    ):
        # Ensure the logger is created only once
        if MyLogger._logger_instance is None:
            MyLogger._logger_instance = logging.getLogger(name)
            logging_level = valid_levels.get(
                os.getenv("LOG", "INFO").upper(), logging.INFO
            )
            MyLogger._logger_instance.setLevel(logging_level)

            # Create a console handler with color support
            console_handler = logging.StreamHandler()
            formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(levelname)s: %(message)s",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "white",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
            console_handler.setFormatter(formatter)

            # Add console handler to the logger
            MyLogger._logger_instance.addHandler(console_handler)

            # If log_file is provided, create a file handler
            if log_file:
                rotating_handler = RotatingFileHandler(
                    log_file, maxBytes=max_bytes, backupCount=backup_count
                )
                rotating_handler.setLevel(logging.DEBUG)
                rotating_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(levelname)s - %(message)s"
                    )
                )
                MyLogger._logger_instance.addHandler(rotating_handler)

        return MyLogger._logger_instance

    @staticmethod
    def debug(msg: str):
        logger = MyLogger.get_logger()  # Retrieve the singleton logger instance
        logger.debug(msg)

    @staticmethod
    def info(msg: str):
        logger = MyLogger.get_logger()
        logger.info(msg)

    @staticmethod
    def warning(msg: str):
        logger = MyLogger.get_logger()
        logger.warning(msg)

    @staticmethod
    def error(msg: str):
        logger = MyLogger.get_logger()
        logger.error(msg)

    @staticmethod
    def critical(msg: str):
        logger = MyLogger.get_logger()
        logger.critical(msg)
