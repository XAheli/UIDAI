"""
Logging utilities for the UIDAI data processing pipeline.
Provides consistent logging across all modules with file and console output.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Handle both direct execution and module import
try:
    from .config import LOGS_PATH, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL
except ImportError:
    from config import LOGS_PATH, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = LOG_LEVEL,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name (usually module name)
        log_file: Optional log file name (will be placed in logs/ directory)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = LOGS_PATH / log_file
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_timestamped_log_file(prefix: str) -> str:
    """
    Generate a timestamped log file name.
    
    Args:
        prefix: Prefix for the log file name
    
    Returns:
        Log file name with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.log"


class ProgressLogger:
    """
    A context manager for logging progress of long-running operations.
    """
    
    def __init__(self, logger: logging.Logger, operation: str, total: int = 0):
        self.logger = logger
        self.operation = operation
        self.total = total
        self.start_time = None
        self.processed = 0
        self.errors = 0
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.operation} (Total: {self.total:,})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.processed / elapsed if elapsed > 0 else 0
        
        if exc_type:
            self.logger.error(
                f"Failed: {self.operation} - Error: {exc_val}\n"
                f"  Processed: {self.processed:,}/{self.total:,}, Errors: {self.errors:,}, "
                f"Time: {elapsed:.1f}s"
            )
            return False
        
        self.logger.info(
            f"Completed: {self.operation}\n"
            f"  Processed: {self.processed:,}/{self.total:,}, Errors: {self.errors:,}, "
            f"Time: {elapsed:.1f}s, Rate: {rate:.1f} records/sec"
        )
        return True
    
    def update(self, count: int = 1, error: bool = False):
        """Update progress counters."""
        self.processed += count
        if error:
            self.errors += count
    
    def log_progress(self, message: str):
        """Log an intermediate progress message."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        pct = (self.processed / self.total * 100) if self.total > 0 else 0
        self.logger.info(f"{message} - Progress: {pct:.1f}% ({self.processed:,}/{self.total:,}) in {elapsed:.1f}s")
