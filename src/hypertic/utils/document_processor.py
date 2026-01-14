import os
from urllib.parse import urlparse

import aiohttp


class DocumentProcessor:
    def __init__(self):
        self.supported_types = {
            ".pdf": self._process_pdf,
            ".md": self._process_markdown,
            ".txt": self._process_text,
            ".csv": self._process_csv,
            ".json": self._process_json,
            ".docx": self._process_docx,
            ".doc": self._process_doc,
            ".rtf": self._process_rtf,
            ".html": self._process_html,
            ".htm": self._process_html,
            ".xml": self._process_xml,
            ".yaml": self._process_yaml,
            ".yml": self._process_yaml,
            ".py": self._process_python,
            ".js": self._process_javascript,
            ".ts": self._process_typescript,
            ".java": self._process_java,
            ".cpp": self._process_cpp,
            ".c": self._process_c,
            ".go": self._process_go,
            ".rs": self._process_rust,
            ".php": self._process_php,
            ".rb": self._process_ruby,
            ".swift": self._process_swift,
            ".kt": self._process_kotlin,
            ".sql": self._process_sql,
            ".sh": self._process_shell,
            ".bash": self._process_shell,
            ".zsh": self._process_shell,
            ".fish": self._process_shell,
            ".ps1": self._process_powershell,
            ".bat": self._process_batch,
            ".log": self._process_log,
        }

    async def process_file(self, file_path: str) -> str:
        """Process any supported file type (async)"""
        if file_path.startswith(("http://", "https://")):
            return await self._process_url(file_path)
        else:
            return await self._process_local_file(file_path)

    def process_file_sync(self, file_path: str) -> str:
        """Process any supported file type (sync)"""
        if file_path.startswith(("http://", "https://")):
            return self._process_url_sync(file_path)
        else:
            return self._process_local_file_sync(file_path)

    async def _process_url(self, url: str) -> str:
        """Download and process URL content"""
        try:
            async with aiohttp.ClientSession() as session, session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    file_type = self._get_file_type_from_url(url)
                    return self._process_content_unified(content, file_type)
                else:
                    raise Exception(f"Failed to download {url}: {response.status}")
        except Exception as e:
            raise Exception(f"Error downloading {url}: {e}") from e

    def _process_url_sync(self, url: str) -> str:
        """Download and process URL content (sync)"""
        try:
            import requests

            response = requests.get(url)
            if response.status_code == 200:
                content = response.content
                file_type = self._get_file_type_from_url(url)
                return self._process_content_unified(content, file_type)
            else:
                raise Exception(f"Failed to download {url}: {response.status_code}")
        except Exception as e:
            raise Exception(f"Error downloading {url}: {e}") from e

    async def _process_local_file(self, file_path: str) -> str:
        """Process local file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = self._get_file_type(file_path)
        if file_type not in self.supported_types:
            raise ValueError(f"Unsupported file type: {file_type}")

        with open(file_path, "rb") as f:
            content = f.read()

        return self._process_content_unified(content, file_type)

    def _process_local_file_sync(self, file_path: str) -> str:
        """Process local file (sync)"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = self._get_file_type(file_path)
        if file_type not in self.supported_types:
            raise ValueError(f"Unsupported file type: {file_type}")

        with open(file_path, "rb") as f:
            content = f.read()

        return self._process_content_unified(content, file_type)

    def _process_content_unified(self, content: bytes, file_type: str) -> str:
        """Process content based on file type (unified for both async and sync)"""
        if file_type not in self.supported_types:
            raise ValueError(f"Unsupported file type: {file_type}")

        processor = self.supported_types[file_type]
        return processor(content)

    def _get_file_type(self, file_path: str) -> str:
        """Get file type from path"""
        return os.path.splitext(file_path)[1].lower()

    def _get_file_type_from_url(self, url: str) -> str:
        """Get file type from URL"""
        parsed = urlparse(url)
        path = parsed.path
        return os.path.splitext(path)[1].lower()

    def _process_pdf(self, content: bytes) -> str:
        """Process PDF content (unified)"""
        try:
            import io

            import PyPDF2

            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except ImportError:
            return "PDF processing requires PyPDF2. This should be installed by default. If missing, install with: pip install PyPDF2"
        except Exception as e:
            return f"Error processing PDF: {e}"

    def _process_markdown(self, content: bytes) -> str:
        """Process Markdown content (unified)"""
        return content.decode("utf-8")

    def _process_text(self, content: bytes) -> str:
        """Process plain text content (unified)"""
        return content.decode("utf-8")

    def _process_log(self, content: bytes) -> str:
        """Process log file content (unified)"""
        return content.decode("utf-8")

    def _process_csv(self, content: bytes) -> str:
        """Process CSV content (unified)"""
        import csv
        import io

        text = ""
        try:
            csv_reader = csv.reader(io.StringIO(content.decode("utf-8")))
            for row in csv_reader:
                text += ", ".join(row) + "\n"
        except Exception:
            text = content.decode("utf-8")

        return text.strip()

    def _process_json(self, content: bytes) -> str:
        """Process JSON content (unified)"""
        import json

        try:
            data = json.loads(content.decode("utf-8"))
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error processing JSON: {e}"

    def _process_docx(self, content: bytes) -> str:
        """Process DOCX content (unified)"""
        try:
            import io

            import docx

            doc = docx.Document(io.BytesIO(content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except ImportError:
            return "DOCX processing requires python-docx. This should be installed by default. If missing, install with: pip install python-docx"
        except Exception as e:
            return f"Error processing DOCX: {e}"

    def _process_doc(self, content: bytes) -> str:
        """Process DOC content (unified)"""
        try:
            import os
            import subprocess
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            try:
                result = subprocess.run(["antiword", tmp_file_path], capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    return result.stdout
                else:
                    return "Error: antiword not available or failed"
            finally:
                os.unlink(tmp_file_path)
        except Exception as e:
            return f"Error processing DOC: {e}"

    def _process_rtf(self, content: bytes) -> str:
        """Process RTF content (unified)"""
        try:
            import striprtf

            result = striprtf.from_bytes(content).text
            return str(result)
        except ImportError:
            return "RTF processing requires striprtf. This should be installed by default. If missing, install with: pip install striprtf"
        except Exception as e:
            return f"Error processing RTF: {e}"

    def _process_html(self, content: bytes) -> str:
        """Process HTML content (unified)"""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(content, "html.parser")
            return soup.get_text()
        except ImportError:
            return (
                "HTML processing requires beautifulsoup4. This should be installed by default. If missing, install with: pip install beautifulsoup4"
            )
        except Exception as e:
            return f"Error processing HTML: {e}"

    def _process_xml(self, content: bytes) -> str:
        """Process XML content (unified)"""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(content, "xml")
            return soup.get_text()
        except ImportError:
            return "XML processing requires beautifulsoup4. This should be installed by default. If missing, install with: pip install beautifulsoup4"
        except Exception as e:
            return f"Error processing XML: {e}"

    def _process_yaml(self, content: bytes) -> str:
        """Process YAML content (unified)"""
        try:
            import yaml

            data = yaml.safe_load(content.decode("utf-8"))
            result = yaml.dump(data, default_flow_style=False)
            return str(result) if result is not None else ""
        except ImportError:
            return "YAML processing requires PyYAML. This should be installed by default. If missing, install with: pip install PyYAML"
        except Exception as e:
            return f"Error processing YAML: {e}"

    def _process_python(self, content: bytes) -> str:
        """Process Python code (unified)"""
        return content.decode("utf-8")

    def _process_javascript(self, content: bytes) -> str:
        """Process JavaScript code (unified)"""
        return content.decode("utf-8")

    def _process_typescript(self, content: bytes) -> str:
        """Process TypeScript code (unified)"""
        return content.decode("utf-8")

    def _process_java(self, content: bytes) -> str:
        """Process Java code (unified)"""
        return content.decode("utf-8")

    def _process_cpp(self, content: bytes) -> str:
        """Process C++ code (unified)"""
        return content.decode("utf-8")

    def _process_c(self, content: bytes) -> str:
        """Process C code (unified)"""
        return content.decode("utf-8")

    def _process_go(self, content: bytes) -> str:
        """Process Go code (unified)"""
        return content.decode("utf-8")

    def _process_rust(self, content: bytes) -> str:
        """Process Rust code (unified)"""
        return content.decode("utf-8")

    def _process_php(self, content: bytes) -> str:
        """Process PHP code (unified)"""
        return content.decode("utf-8")

    def _process_ruby(self, content: bytes) -> str:
        """Process Ruby code (unified)"""
        return content.decode("utf-8")

    def _process_swift(self, content: bytes) -> str:
        """Process Swift code (unified)"""
        return content.decode("utf-8")

    def _process_kotlin(self, content: bytes) -> str:
        """Process Kotlin code (unified)"""
        return content.decode("utf-8")

    def _process_sql(self, content: bytes) -> str:
        """Process SQL code (unified)"""
        return content.decode("utf-8")

    def _process_shell(self, content: bytes) -> str:
        """Process shell script content (unified)"""
        return content.decode("utf-8")

    def _process_powershell(self, content: bytes) -> str:
        """Process PowerShell script content (unified)"""
        return content.decode("utf-8")

    def _process_batch(self, content: bytes) -> str:
        """Process batch file content (unified)"""
        return content.decode("utf-8")
