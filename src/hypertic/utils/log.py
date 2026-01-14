import logging
import sys


def setup_logging(level: str = "INFO", format_string: str | None = None, log_file: str | None = None) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if format_string is None:
        format_string = "%(levelname)s - %(message)s"

    handlers: list[logging.Handler] = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    logger = logging.getLogger("hypertic")
    logger.setLevel(numeric_level)

    logger.handlers.clear()

    for handler in handlers:
        logger.addHandler(handler)

    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    logger_name = name if name.startswith("hypertic.") else f"hypertic.{name}"

    return logging.getLogger(logger_name)


def mask_connection_string(conn_string: str) -> str:
    if not conn_string:
        return conn_string

    import re

    pattern = r"(://[^:]*:)([^@]+)(@)"

    def replace_password(match):
        return match.group(1) + "***" + match.group(3)

    return re.sub(pattern, replace_password, conn_string)


setup_logging()
