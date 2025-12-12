"""
Structured logging configuration using structlog.
"""

import logging
import sys
from typing import Optional
import structlog
from structlog.types import EventDict, WrappedLogger


def add_app_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add application context to log events."""
    event_dict["app"] = "qdrant-quantization-benchmark"
    event_dict["version"] = "0.1.0"
    return event_dict


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    verbose: bool = False,
    quiet: bool = False
) -> structlog.BoundLogger:
    """
    Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Output logs as JSON (for CloudWatch/monitoring)
        verbose: Enable verbose output (sets level to DEBUG)
        quiet: Suppress most output (sets level to ERROR)
        
    Returns:
        Configured structlog logger
    """
    # Determine effective log level
    if quiet:
        effective_level = "ERROR"
    elif verbose:
        effective_level = "DEBUG"
    else:
        effective_level = level.upper()
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, effective_level)
    )
    
    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        add_app_context,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    
    # Add exception formatting
    if not json_output:
        processors.append(structlog.processors.ExceptionPrettyPrinter())
    
    # Choose renderer based on output format
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Use colorized console output for human readability
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback
            )
        )
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, effective_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger()
    
    # Log configuration
    logger.debug(
        "logging_configured",
        level=effective_level,
        json_output=json_output,
        verbose=verbose,
        quiet=quiet
    )
    
    return logger


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a logger instance.
    
    Args:
        name: Optional logger name (typically module name)
        
    Returns:
        Configured structlog logger
    """
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(module=name)
    return logger


class LoggerMixin:
    """
    Mixin class to add logging capability to other classes.
    
    Usage:
        class MyClass(LoggerMixin):
            def __init__(self):
                self.setup_logger(self.__class__.__name__)
                
            def my_method(self):
                self.log.info("doing_something", param=value)
    """
    
    def setup_logger(self, name: str) -> None:
        """
        Set up logger for this class instance.
        
        Args:
            name: Logger name (typically class name)
        """
        self.log = get_logger(name)


# Progress tracking helper
class ProgressLogger:
    """Helper for logging progress of long-running operations."""
    
    def __init__(
        self,
        logger: structlog.BoundLogger,
        operation: str,
        total: int,
        update_interval: int = 1000
    ):
        """
        Initialize progress logger.
        
        Args:
            logger: Structlog logger instance
            operation: Name of the operation
            total: Total number of items
            update_interval: Log progress every N items
        """
        self.logger = logger
        self.operation = operation
        self.total = total
        self.update_interval = update_interval
        self.processed = 0
        
        self.logger.info(
            f"{operation}_started",
            operation=operation,
            total=total
        )
    
    def update(self, count: int = 1) -> None:
        """
        Update progress counter.
        
        Args:
            count: Number of items processed
        """
        self.processed += count
        
        if self.processed % self.update_interval == 0 or self.processed == self.total:
            percent = (self.processed / self.total) * 100
            self.logger.info(
                f"{self.operation}_progress",
                operation=self.operation,
                processed=self.processed,
                total=self.total,
                percent=f"{percent:.1f}"
            )
    
    def complete(self) -> None:
        """Mark operation as complete."""
        self.logger.info(
            f"{self.operation}_completed",
            operation=self.operation,
            total_processed=self.processed,
            expected_total=self.total
        )


# Timing context manager
class Timer:
    """Context manager for timing operations with structured logging."""
    
    def __init__(self, logger: structlog.BoundLogger, operation: str, **kwargs):
        """
        Initialize timer.
        
        Args:
            logger: Structlog logger instance
            operation: Name of the operation being timed
            **kwargs: Additional context to include in logs
        """
        self.logger = logger
        self.operation = operation
        self.context = kwargs
        self.start_time = None
        self.duration_ms = None
    
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        self.logger.debug(
            f"{self.operation}_started",
            operation=self.operation,
            **self.context
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log results."""
        import time
        self.duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type is None:
            self.logger.info(
                f"{self.operation}_completed",
                operation=self.operation,
                duration_ms=f"{self.duration_ms:.2f}",
                **self.context
            )
        else:
            self.logger.error(
                f"{self.operation}_failed",
                operation=self.operation,
                duration_ms=f"{self.duration_ms:.2f}",
                error=str(exc_val),
                **self.context
            )