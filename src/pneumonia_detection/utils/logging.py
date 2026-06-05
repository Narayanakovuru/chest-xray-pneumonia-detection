import logging
import sys
from pathlib import Path
from typing import Optional

def configure_logger(
    log_dir: str = "logs",
    log_file: str = "training.log",
    level: str = "INFO"
) -> logging.Logger:
    """Setup root logger handler to write outputs to both stdout and a file.

    Args:
        log_dir (str): Directory where logs should be stored.
        log_file (str): Filename for the log file.
        level (str): String representation of log level (e.g. "INFO").

    Returns:
        logging.Logger: Root logger setup.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Ensure logs path exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure logging formats
    log_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clean previous handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File Handler
    file_handler = logging.FileHandler(log_path / log_file, encoding="utf-8")
    file_handler.setFormatter(log_format)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    logging.getLogger("PIL").setLevel(logging.WARNING)  # Mute PIL spam
    
    root_logger.info(f"Logging initialized. Level: {level} | Log file: {log_path / log_file}")
    return root_logger
