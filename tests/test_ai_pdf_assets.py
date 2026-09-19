import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

from rac_ai_scientist import cli
from rac_ai_scientist.hosts import ai_researcher


class MissingAsset(FileNotFoundError):
    """LocalEntryNotFoundError inherits FileNotFoundError in the fixed Hub SDK."""


class AiPdfAssetTests(unittest.TestCase):
    def test_missing_docling_asset_is_not_reported_as_a_missing_pdf(self):
        converter_module = ModuleType('research_agent.inno.environment.markdown_browser.mdconvert')
        hub_errors = ModuleType('huggingface_hub.errors')
        hub_errors.LocalEntryNotFoundError = MissingAsset

        class PdfConverter:
            def convert(self, path, **kwargs):
                if path == 'missing.pdf':
                    raise FileNotFoundError(path)
                if path == 'readable.pdf':
                    raise MissingAsset('offline layout cache is missing')
                return 'native conversion result'

        converter_module.PdfConverter = PdfConverter
        with patch.dict(sys.modules, {converter_module.__name__: converter_module, hub_errors.__name__: hub_errors}):
            ai_researcher._install_pdf_asset_error()
            ai_researcher._install_pdf_asset_error()
            with self.assertRaisesRegex(RuntimeError, 'Docling conversion asset missing'):
                PdfConverter().convert('readable.pdf', file_extension='.pdf')
            with self.assertRaises(FileNotFoundError):
                PdfConverter().convert('missing.pdf', file_extension='.pdf')
            self.assertEqual(PdfConverter().convert('available.pdf', file_extension='.pdf'), 'native conversion result')

    def test_doctor_blocks_missing_docling_cache_without_network_access(self):
        self.check_doctor('docling-project/docling-layout-heron', 2)

    def test_doctor_blocks_missing_default_table_model_without_network_access(self):
        self.check_doctor('docling-project/docling-models', 2)

    def test_doctor_accepts_available_docling_cache_without_network_access(self):
        self.check_doctor(None, 0)

    def test_doctor_blocks_unloadable_opencv_runtime(self):
        self.check_doctor(None, 2, ImportError('libGL.so.1: cannot open shared object file'))

    def check_doctor(self, missing_repo, expected, import_error=None):
        hub = ModuleType('huggingface_hub')
        errors = ModuleType('huggingface_hub.errors')
        errors.LocalEntryNotFoundError = MissingAsset
        def cached(repo_id, **kwargs):
            if repo_id == missing_repo:
                raise MissingAsset('cache missing')
            return '/cache/snapshot'
        hub.snapshot_download = Mock(side_effect=cached)
        args = SimpleNamespace(project_root='.', host='ai_researcher', upstream='/pinned')
        with patch.dict(sys.modules, {hub.__name__: hub, errors.__name__: errors}), \
                patch.object(cli, '_selected_upstream', return_value=Path('/pinned')), \
                patch.object(cli.importlib.util, 'find_spec', return_value=object()), \
                patch.object(cli.importlib, 'import_module', return_value=object(), side_effect=import_error), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli._doctor_host(args), expected)
        expected_calls = [call('docling-project/docling-layout-heron', revision='main', local_files_only=True)]
        if missing_repo != 'docling-project/docling-layout-heron':
            expected_calls.append(call('docling-project/docling-models', revision='v2.3.0', local_files_only=True))
        self.assertEqual(hub.snapshot_download.call_args_list, expected_calls)


if __name__ == '__main__':
    unittest.main()
