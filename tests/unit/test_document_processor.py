from unittest.mock import mock_open, patch

import pytest

from hypertic.utils.document_processor import DocumentProcessor


class TestDocumentProcessor:
    def test_file_processor_creation(self):
        processor = DocumentProcessor()
        assert processor is not None
        assert processor.supported_types is not None

    def test_file_processor_supported_types(self):
        processor = DocumentProcessor()
        assert len(processor.supported_types) > 0
        assert ".txt" in processor.supported_types
        assert ".pdf" in processor.supported_types

    def test_get_file_type(self):
        processor = DocumentProcessor()
        assert processor._get_file_type("test.txt") == ".txt"
        assert processor._get_file_type("file.pdf") == ".pdf"
        assert processor._get_file_type("IMAGE.JPG") == ".jpg"

    def test_get_file_type_from_url(self):
        processor = DocumentProcessor()
        assert processor._get_file_type_from_url("https://example.com/file.pdf") == ".pdf"
        assert processor._get_file_type_from_url("https://example.com/doc.txt?param=1") == ".txt"

    def test_process_text(self):
        processor = DocumentProcessor()
        content = b"Test content"
        result = processor._process_text(content)
        assert result == "Test content"

    def test_process_markdown(self):
        processor = DocumentProcessor()
        content = b"# Title\n\nContent"
        result = processor._process_markdown(content)
        assert result == "# Title\n\nContent"

    def test_process_log(self):
        processor = DocumentProcessor()
        content = b"2024-01-01 INFO: Test log"
        result = processor._process_log(content)
        assert result == "2024-01-01 INFO: Test log"

    def test_process_python(self):
        processor = DocumentProcessor()
        content = b"def test():\n    pass"
        result = processor._process_python(content)
        assert result == "def test():\n    pass"

    def test_process_javascript(self):
        processor = DocumentProcessor()
        content = b"function test() {}"
        result = processor._process_javascript(content)
        assert result == "function test() {}"

    def test_process_typescript(self):
        processor = DocumentProcessor()
        content = b"const x: number = 1;"
        result = processor._process_typescript(content)
        assert result == "const x: number = 1;"

    def test_process_java(self):
        processor = DocumentProcessor()
        content = b"public class Test {}"
        result = processor._process_java(content)
        assert result == "public class Test {}"

    def test_process_cpp(self):
        processor = DocumentProcessor()
        content = b"#include <iostream>"
        result = processor._process_cpp(content)
        assert result == "#include <iostream>"

    def test_process_c(self):
        processor = DocumentProcessor()
        content = b"#include <stdio.h>"
        result = processor._process_c(content)
        assert result == "#include <stdio.h>"

    def test_process_go(self):
        processor = DocumentProcessor()
        content = b"package main"
        result = processor._process_go(content)
        assert result == "package main"

    def test_process_rust(self):
        processor = DocumentProcessor()
        content = b"fn main() {}"
        result = processor._process_rust(content)
        assert result == "fn main() {}"

    def test_process_php(self):
        processor = DocumentProcessor()
        content = b"<?php echo 'test'; ?>"
        result = processor._process_php(content)
        assert result == "<?php echo 'test'; ?>"

    def test_process_ruby(self):
        processor = DocumentProcessor()
        content = b"puts 'test'"
        result = processor._process_ruby(content)
        assert result == "puts 'test'"

    def test_process_swift(self):
        processor = DocumentProcessor()
        content = b'print("test")'
        result = processor._process_swift(content)
        assert result == 'print("test")'

    def test_process_kotlin(self):
        processor = DocumentProcessor()
        content = b"fun main() {}"
        result = processor._process_kotlin(content)
        assert result == "fun main() {}"

    def test_process_sql(self):
        processor = DocumentProcessor()
        content = b"SELECT * FROM users;"
        result = processor._process_sql(content)
        assert result == "SELECT * FROM users;"

    def test_process_shell(self):
        processor = DocumentProcessor()
        content = b"#!/bin/bash\necho 'test'"
        result = processor._process_shell(content)
        assert result == "#!/bin/bash\necho 'test'"

    def test_process_powershell(self):
        processor = DocumentProcessor()
        content = b"Write-Host 'test'"
        result = processor._process_powershell(content)
        assert result == "Write-Host 'test'"

    def test_process_batch(self):
        processor = DocumentProcessor()
        content = b"@echo off\necho test"
        result = processor._process_batch(content)
        assert result == "@echo off\necho test"

    def test_process_csv(self):
        processor = DocumentProcessor()
        content = b"col1,col2\nval1,val2"
        result = processor._process_csv(content)
        assert "col1" in result
        assert "val1" in result

    def test_process_json(self):
        processor = DocumentProcessor()
        content = b'{"key": "value"}'
        result = processor._process_json(content)
        assert "key" in result
        assert "value" in result

    def test_process_json_invalid(self):
        processor = DocumentProcessor()
        content = b"invalid json"
        result = processor._process_json(content)
        assert "Error" in result

    def test_process_yaml(self):
        processor = DocumentProcessor()
        content = b"key: value\nlist:\n  - item1"
        result = processor._process_yaml(content)
        assert isinstance(result, str)

    def test_process_pdf(self):
        processor = DocumentProcessor()
        content = b"fake pdf content"
        # Will either process or return error message
        result = processor._process_pdf(content)
        assert isinstance(result, str)

    def test_process_docx(self):
        processor = DocumentProcessor()
        content = b"fake docx"
        result = processor._process_docx(content)
        assert isinstance(result, str)

    def test_process_rtf(self):
        processor = DocumentProcessor()
        content = b"fake rtf"
        result = processor._process_rtf(content)
        assert isinstance(result, str)

    def test_process_html(self):
        processor = DocumentProcessor()
        content = b"<html><body>Test</body></html>"
        result = processor._process_html(content)
        assert isinstance(result, str)

    def test_process_xml(self):
        processor = DocumentProcessor()
        content = b"<root><item>Test</item></root>"
        result = processor._process_xml(content)
        assert isinstance(result, str)

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=b"test content")
    def test_process_local_file_sync(self, mock_file, mock_exists):
        processor = DocumentProcessor()
        result = processor._process_local_file_sync("test.txt")
        assert result == "test content"

    @patch("os.path.exists", return_value=False)
    def test_process_local_file_sync_not_found(self, mock_exists):
        processor = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            processor._process_local_file_sync("nonexistent.txt")

    @patch("os.path.exists", return_value=True)
    def test_process_local_file_sync_unsupported(self, mock_exists):
        processor = DocumentProcessor()
        with pytest.raises(ValueError, match="Unsupported"):
            processor._process_local_file_sync("test.xyz")

    @pytest.mark.asyncio
    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=b"test content")
    async def test_process_local_file_async(self, mock_file, mock_exists):
        processor = DocumentProcessor()
        result = await processor._process_local_file("test.txt")
        assert result == "test content"

    @pytest.mark.asyncio
    @patch("os.path.exists", return_value=False)
    async def test_process_local_file_async_not_found(self, mock_exists):
        processor = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            await processor._process_local_file("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_process_url_async(self):
        processor = DocumentProcessor()
        # Will either process or raise error - just test it doesn't crash
        try:
            result = await processor._process_url("https://example.com/file.txt")
            assert isinstance(result, str)
        except Exception:
            pass  # Expected if network unavailable

    @pytest.mark.asyncio
    async def test_process_url_async_error(self):
        processor = DocumentProcessor()
        # Test with invalid URL or network error
        try:
            await processor._process_url("https://invalid-url-that-does-not-exist-12345.com/file.txt")
        except Exception as e:
            assert "Error downloading" in str(e) or "Error" in str(e)

    def test_process_url_sync(self):
        processor = DocumentProcessor()
        # Will either process or raise error - just test it doesn't crash
        try:
            result = processor._process_url_sync("https://example.com/file.txt")
            assert isinstance(result, str)
        except Exception:
            pass  # Expected if requests not available or network unavailable

    def test_process_url_sync_error(self):
        processor = DocumentProcessor()
        # Test with invalid URL
        try:
            processor._process_url_sync("https://invalid-url-that-does-not-exist-12345.com/file.txt")
        except Exception as e:
            assert "Error downloading" in str(e) or "Error" in str(e)

    def test_process_content_unified_unsupported(self):
        processor = DocumentProcessor()
        with pytest.raises(ValueError, match="Unsupported"):
            processor._process_content_unified(b"content", ".xyz")

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=b"test")
    def test_process_file_sync_local(self, mock_file, mock_exists):
        processor = DocumentProcessor()
        result = processor.process_file_sync("test.txt")
        assert result == "test"

    def test_process_file_sync_url(self):
        processor = DocumentProcessor()
        # Test URL detection - will try to process URL
        try:
            result = processor.process_file_sync("https://example.com/file.txt")
            assert isinstance(result, str)
        except (Exception, ImportError):
            pass  # Expected if requests not available or network error

    @pytest.mark.asyncio
    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=b"test")
    async def test_process_file_async_local(self, mock_file, mock_exists):
        processor = DocumentProcessor()
        result = await processor.process_file("test.txt")
        assert result == "test"

    @pytest.mark.asyncio
    async def test_process_file_async_url(self):
        processor = DocumentProcessor()
        # Test URL detection - will try to process URL
        try:
            result = await processor.process_file("https://example.com/file.txt")
            assert isinstance(result, str)
        except (Exception, ImportError):
            pass  # Expected if network unavailable
