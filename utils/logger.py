import logging
import logging.config
import yaml
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str,
    config_path: str,
    default_level: int = logging.INFO,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up logger using YAML configuration.
    
    Args:
        name: Name of the logger
        config_path: Path to logging YAML config file
        default_level: Default logging level if config fails
        log_to_console: Whether to log to console
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Ensure data/logs directory exists
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Update the file handler path in the config
        if 'handlers' in config and 'file_handler' in config['handlers']:
            config['handlers']['file_handler']['filename'] = str(log_dir / f"{name}.log")
        
        # Modify console handler based on log_to_console parameter
        if not log_to_console and 'handlers' in config and 'console_handler' in config['handlers']:
            if 'root' in config and 'handlers' in config['root']:
                config['root']['handlers'].remove('console_handler')
            for logger in config.get('loggers', {}).values():
                if 'handlers' in logger:
                    logger['handlers'] = [h for h in logger['handlers'] if h != 'console_handler']
        
        logging.config.dictConfig(config)
        logger = logging.getLogger(name)
        
    except Exception as e:
        # Fallback to basic configuration if YAML loading fails
        print(f"Error loading logging config: {e}. Using basic configuration.")
        logger = logging.getLogger(name)
        logger.setLevel(default_level)
        
        # Remove any existing handlers
        logger.handlers = []
        
        # Ensure data/logs directory exists
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setLevel(default_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler (if enabled)
        if log_to_console:
            ch = logging.StreamHandler()
            ch.setLevel(default_level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get an existing logger"""
    return logging.getLogger(name)