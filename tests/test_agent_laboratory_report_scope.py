import sys
import types
import unittest
from unittest.mock import patch

from rac_ai_scientist.hosts.agent_laboratory import _install_report_writing_scope


def upstream_report_writing(self):
    return research_topic, compile_pdf


class AgentLaboratoryReportScopeTests(unittest.TestCase):
    def test_report_writer_uses_each_workflows_topic_and_pdf_setting(self):
        class FakeWorkflow:
            def __init__(self, topic, compile_pdf):
                self.research_topic = topic
                self.compile_pdf = compile_pdf

        upstream_globals = {"__builtins__": __builtins__}
        FakeWorkflow.report_writing = types.FunctionType(
            upstream_report_writing.__code__, upstream_globals
        )
        ai_lab_repo = types.ModuleType("ai_lab_repo")
        ai_lab_repo.LaboratoryWorkflow = FakeWorkflow
        with patch.dict(sys.modules, {"ai_lab_repo": ai_lab_repo}):
            with self.assertRaises(NameError):
                FakeWorkflow("first topic", False).report_writing()
            _install_report_writing_scope()
            _install_report_writing_scope()
            self.assertEqual(FakeWorkflow("first topic", False).report_writing(), ("first topic", False))
            self.assertEqual(FakeWorkflow("second topic", True).report_writing(), ("second topic", True))


if __name__ == "__main__":
    unittest.main()
