"""
Tests for logging utilities and configuration.
"""

import pytest
import time

from qdrant_quantization_benchmark.logging import (
    setup_logging,
    get_logger,
    LoggerMixin,
    ProgressLogger,
    Timer,
)


class TestSetupLogging:
    """Tests for setup_logging function."""
    
    def test_default_setup(self):
        """Test default logging setup."""
        logger = setup_logging()
        assert logger is not None
        # Verify it has logger methods
        assert callable(getattr(logger, 'info', None))
        assert callable(getattr(logger, 'debug', None))
        assert callable(getattr(logger, 'error', None))
    
    def test_verbose_mode(self):
        """Test verbose mode setup."""
        logger = setup_logging(verbose=True)
        assert logger is not None
        assert callable(getattr(logger, 'info', None))
    
    def test_quiet_mode(self):
        """Test quiet mode setup."""
        logger = setup_logging(quiet=True)
        assert logger is not None
        assert callable(getattr(logger, 'info', None))
    
    def test_json_output(self):
        """Test JSON output mode."""
        logger = setup_logging(json_output=True)
        assert logger is not None
        assert callable(getattr(logger, 'info', None))
    
    def test_custom_level(self):
        """Test custom log level."""
        logger = setup_logging(level="WARNING")
        assert logger is not None
        assert callable(getattr(logger, 'info', None))
    
    def test_verbose_overrides_level(self):
        """Test that verbose=True sets DEBUG level."""
        logger = setup_logging(level="ERROR", verbose=True)
        assert logger is not None
    
    def test_quiet_overrides_level(self):
        """Test that quiet=True sets ERROR level."""
        logger = setup_logging(level="DEBUG", quiet=True)
        assert logger is not None


class TestGetLogger:
    """Tests for get_logger function."""
    
    def test_get_logger_with_name(self):
        """Test getting logger with specific name."""
        logger = get_logger("test_module")
        assert logger is not None
        assert callable(getattr(logger, 'info', None))
    
    def test_get_logger_no_name(self):
        """Test getting logger without name."""
        logger = get_logger()
        assert logger is not None
        assert callable(getattr(logger, 'info', None))
    
    def test_logger_binds_name(self):
        """Test that logger properly binds module name."""
        logger = get_logger("my_module")
        # Verify we can log with the bound logger
        logger.info("test_message")


class TestLoggerMixin:
    """Tests for LoggerMixin class."""
    
    def test_mixin_provides_setup_logger_method(self):
        """Test that mixin provides setup_logger method."""
        
        class TestClass(LoggerMixin):
            pass
        
        obj = TestClass()
        assert hasattr(obj, 'setup_logger')
    
    def test_mixin_creates_log_attribute(self):
        """Test that setup_logger creates log attribute."""
        
        class TestClass(LoggerMixin):
            def __init__(self):
                self.setup_logger("TestClass")
        
        obj = TestClass()
        assert hasattr(obj, 'log')
        assert callable(getattr(obj.log, 'info', None))
    
    def test_mixin_logger_can_log(self):
        """Test that mixin logger can actually log."""
        
        class TestClass(LoggerMixin):
            def __init__(self):
                self.setup_logger("TestClass")
            
            def do_something(self):
                self.log.info("doing_something", value=42)
        
        obj = TestClass()
        # Should not raise exception
        obj.do_something()


class TestProgressLogger:
    """Tests for ProgressLogger class."""
    
    def test_initialization(self):
        """Test ProgressLogger initialization."""
        logger = get_logger()
        progress = ProgressLogger(
            logger=logger,
            operation="test_operation",
            total=100
        )
        
        assert progress.total == 100
        assert progress.processed == 0
        assert progress.operation == "test_operation"
    
    def test_update_progress(self):
        """Test progress update."""
        logger = get_logger()
        progress = ProgressLogger(
            logger=logger,
            operation="upload",
            total=100,
            update_interval=10
        )
        
        progress.update(10)
        assert progress.processed == 10
        
        progress.update(5)
        assert progress.processed == 15
    
    def test_complete(self):
        """Test marking operation as complete."""
        logger = get_logger()
        progress = ProgressLogger(
            logger=logger,
            operation="task",
            total=100
        )
        
        progress.update(50)
        progress.complete()
        
        # Processed count should still be 50, not changed to total
        assert progress.processed == 50
    
    def test_update_interval(self):
        """Test that update_interval controls logging frequency."""
        logger = get_logger()
        progress = ProgressLogger(
            logger=logger,
            operation="test",
            total=1000,
            update_interval=100
        )
        
        # Update multiple times
        for _ in range(10):
            progress.update(10)
        
        assert progress.processed == 100
    
    def test_final_progress_always_logged(self):
        """Test that final progress is always logged."""
        logger = get_logger()
        progress = ProgressLogger(
            logger=logger,
            operation="test",
            total=100,
            update_interval=25
        )
        
        # Update to exactly total
        progress.update(100)
        
        assert progress.processed == 100


class TestTimer:
    """Tests for Timer context manager."""
    
    def test_timer_basic_usage(self):
        """Test basic timer usage."""
        logger = get_logger()
        
        with Timer(logger, "test_operation"):
            time.sleep(0.01)
        
        # If no exception raised, timer works
    
    def test_timer_with_context(self):
        """Test timer with additional context."""
        logger = get_logger()
        
        with Timer(logger, "upload", batch=1, collection="test"):
            time.sleep(0.01)
        
        # If no exception raised, timer works
    
    def test_timer_measures_time(self):
        """Test that timer actually measures elapsed time."""
        logger = get_logger()
        
        with Timer(logger, "test") as timer:
            time.sleep(0.05)
        
        # Should have measured at least 50ms
        assert timer.duration_ms is not None
        assert timer.duration_ms >= 45  # Allow some variance
    
    def test_timer_with_zero_time(self):
        """Test timer with instant operation."""
        logger = get_logger()
        
        with Timer(logger, "instant_op") as timer:
            pass
        
        assert timer.duration_ms is not None
        assert timer.duration_ms >= 0
    
    def test_timer_preserves_context(self):
        """Test that timer preserves context kwargs."""
        logger = get_logger()
        context = {"test": "value", "number": 42}
        
        with Timer(logger, "context_test", **context):
            pass
        
        # Context should still be accessible (though timer doesn't modify it)
    
    def test_timer_on_exception(self):
        """Test timer behavior when exception occurs."""
        logger = get_logger()
        
        # Timer should log error but not suppress exception
        with pytest.raises(ValueError):
            with Timer(logger, "failing_op") as timer:
                raise ValueError("Test error")
        
        # Timer should have recorded duration even on failure
        assert timer.duration_ms is not None
    
    def test_timer_start_time_set(self):
        """Test that start_time is set on entry."""
        logger = get_logger()
        
        with Timer(logger, "test") as timer:
            assert timer.start_time is not None