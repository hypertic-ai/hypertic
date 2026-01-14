import logging
from unittest.mock import MagicMock, patch

import pytest

from hypertic.utils.log import get_logger, mask_connection_string, setup_logging


@pytest.mark.unit
class TestLogging:
    @pytest.mark.parametrize(
        "module_name",
        ["test_module", "another_module", "hypertic.agents"],
    )
    def test_get_logger(self, module_name):
        """Test logger creation for different module names."""
        logger = get_logger(module_name)
        assert logger is not None
        assert logger.name.endswith(module_name)

    def test_get_logger_different_modules(self):
        """Test that different modules get different loggers."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        assert logger1.name.endswith("module1")
        assert logger2.name.endswith("module2")
        assert logger1 is not logger2

    def test_get_logger_with_hypertic_prefix(self):
        """Test logger with hypertic prefix."""
        logger = get_logger("hypertic.test")
        assert logger.name == "hypertic.test"

    def test_get_logger_without_hypertic_prefix(self):
        """Test logger without hypertic prefix gets it added."""
        logger = get_logger("test")
        assert logger.name == "hypertic.test"

    @pytest.mark.parametrize(
        "level",
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    def test_setup_logging_levels(self, level):
        """Test logging setup with different levels."""
        setup_logging(level=level)
        logger = logging.getLogger("hypertic")
        assert logger.level == getattr(logging, level)

    def test_setup_logging_default(self):
        """Test logging setup with default parameters."""
        setup_logging()
        logger = logging.getLogger("hypertic")
        assert logger.level == logging.INFO

    def test_setup_logging_custom_format(self):
        """Test logging setup with custom format."""
        custom_format = "%(name)s - %(levelname)s - %(message)s"
        setup_logging(format_string=custom_format)
        logger = logging.getLogger("hypertic")
        assert len(logger.handlers) > 0

    @patch("hypertic.utils.log.logging.FileHandler")
    def test_setup_logging_with_file(self, mock_file_handler):
        """Test logging setup with log file."""
        mock_handler = MagicMock()
        mock_file_handler.return_value = mock_handler
        setup_logging(log_file="test.log")
        mock_file_handler.assert_called_once_with("test.log")

    def test_setup_logging_no_propagate(self):
        """Test that logger doesn't propagate."""
        setup_logging()
        logger = logging.getLogger("hypertic")
        assert logger.propagate is False


@pytest.mark.unit
class TestMaskConnectionString:
    def test_mask_connection_string_empty(self):
        """Test masking empty connection string."""
        result = mask_connection_string("")
        assert result == ""

    def test_mask_connection_string_none(self):
        """Test masking None connection string."""
        result = mask_connection_string(None)
        assert result is None

    def test_mask_connection_string_with_password(self):
        """Test masking password in connection string."""
        conn_str = "mongodb://user:password123@localhost:27017/db"
        result = mask_connection_string(conn_str)
        assert "password123" not in result
        assert "***" in result
        assert "user" in result
        assert "@localhost" in result

    def test_mask_connection_string_no_password(self):
        """Test connection string without password."""
        conn_str = "mongodb://localhost:27017/db"
        result = mask_connection_string(conn_str)
        assert result == conn_str

    def test_mask_connection_string_postgres(self):
        """Test masking PostgreSQL connection string."""
        conn_str = "postgresql://user:secret@localhost:5432/mydb"
        result = mask_connection_string(conn_str)
        assert "secret" not in result
        assert "***" in result

    def test_mask_connection_string_redis(self):
        """Test masking Redis connection string."""
        conn_str = "redis://:mypassword@localhost:6379/0"
        result = mask_connection_string(conn_str)
        assert "mypassword" not in result
        assert "***" in result
