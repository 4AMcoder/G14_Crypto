import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name="trading_bot", log_file="logs/trading_bot.log", level=logging.INFO):
    """
    Sets up a logger with file and console handlers.

    Parameters:
    - name (str): Name of the logger.
    - log_file (str): Path to the log file.
    - level (int): Logging level (e.g., logging.INFO).

    Returns:
    - logger (logging.Logger): Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Logger initialized.")
    logger.error("This is a test error message.")
