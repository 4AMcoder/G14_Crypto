# utils/logger.py
import os
import logging
import logging.config
import yaml
from pathlib import Path

def setup_logger(
    name: str = "trading_bot",
    config_path: str = "config/logging_config.yaml",
    default_level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logger using YAML config file with fallback to basic configuration.
    """
    try:
        # Ensure logs directory exists in data folder
        logs_dir = Path('data/logs')
        logs_dir.mkdir(exist_ok=True, parents=True)
        
        # Get absolute path of the config file
        config_path = Path(config_path).resolve()
        
        if config_path.exists():
            with open(config_path, 'rt') as f:
                try:
                    config = yaml.safe_load(f)
                    
                    # Ensure log file path is absolute and in data/logs
                    handlers = config.get('handlers', {})
                    file_handler = handlers.get('file_handler', {})
                    if 'filename' in file_handler:
                        log_file = Path(file_handler['filename'])
                        if not log_file.is_absolute():
                            file_handler['filename'] = str(Path.cwd() / 'data' / 'logs' / log_file.name)
                    
                    logging.config.dictConfig(config)
                    logger = logging.getLogger(name)
                    logger.debug(f"Loaded logging configuration from {config_path}")
                    return logger
                except Exception as e:
                    print(f"Error in logging config: {str(e)}")
                    raise
        else:
            # Fallback to basic configuration if YAML not found
            logger = logging.getLogger(name)
            logger.setLevel(default_level)
            
            # Create handlers with absolute paths in data/logs
            log_file = Path.cwd() / 'data' / 'logs' / 'trading_bot.log'
            file_handler = logging.handlers.RotatingFileHandler(
                str(log_file),
                maxBytes=5*1024*1024,
                backupCount=3
            )
            console_handler = logging.StreamHandler()
            
            # Create formatters and add it to handlers
            file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_format = logging.Formatter('%(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)
            console_handler.setFormatter(console_format)
            
            # Add handlers to the logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            logger.warning(f"Config file not found at {config_path}. Using basic configuration.")
            return logger
            
    except Exception as e:
        print(f"Error setting up logger: {str(e)}")
        # If anything goes wrong, set up basic logging
        logging.basicConfig(
            level=default_level,
            format='%(levelname)s - %(message)s'
        )
        logger = logging.getLogger(name)
        logger.error(f"Error setting up logger: {str(e)}")
        logger.warning("Falling back to basic logging configuration")
        return logger

def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one."""
    return logging.getLogger(name)

if __name__ == "__main__":
    # Print debug information
    print(f"Current working directory: {os.getcwd()}")
    print(f"Logs directory: {Path('data/logs').absolute()}")
    
    # Test the logger
    logger = setup_logger()
    logger.info("Logger initialized successfully")
    
    # Test child logger
    child_logger = get_logger("trading_bot.backtest")
    child_logger.info("Child logger test message")