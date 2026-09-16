import asyncio
import os
import sys
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

from rac_ai_scientist.artifacts import artifact_kind
from rac_ai_scientist.hosts.ai_researcher import AIResearcherBridge
from rac_ai_scientist.schemas import Budget, Usage
from test_data_to_paper_native_reset import bridge_for
from rac_ai_scientist.hosts.data_to_paper import DataToPaperBridge
from rac_ai_scientist.hosts.auto_research_claw import AutoResearchClawBridge
from rac_ai_scientist.artifacts import snapshot_workspace


class RuntimeBlockersTests(unittest.TestCase):
    def test_d2p_retry_resets_native_conversation_before_reentering(self):
        with tempfile.TemporaryDirectory() as raw:
            bridge = bridge_for(Path(raw), None)
            bridge.runner.stages_to_conversations_lens = {'code': 2}
            events = []
            bridge.runner.reset_to_stage = lambda stage: events.append(('reset', stage))
            bridge.runner._run_stage = lambda stage: events.append(('run', stage))
            bridge.completed.update({'data_exploration', 'research_goal', 'data_analysis', 'paper_writing'})
            with patch.object(DataToPaperBridge, '_available_cards', return_value=[SimpleNamespace(capability_id='data_analysis', available=True)]), patch.object(DataToPaperBridge, '_stages_for', return_value=('code',)):
                result = bridge.invoke('data_analysis', None)
            self.assertIsNone(result.error)
            self.assertEqual(events, [('reset', 'code'), ('run', 'code')])
            self.assertNotIn('paper_writing', bridge.completed)

    def test_auto_exports_real_scope_and_literature_evidence(self):
        with tempfile.TemporaryDirectory() as raw:
            bridge = object.__new__(AutoResearchClawBridge)
            bridge.workspace = Path(raw)
            bridge.run_dir = Path(raw) / 'auto_research_claw_native'
            for name, content in [('stage-01/goal.md', 'Research question and scope'), ('stage-05/shortlist.jsonl', '{"title":"Selected study"}')]:
                path = bridge.run_dir / name
                path.parent.mkdir(parents=True)
                path.write_text(content)
            bridge._normalize_products()
            artifacts = snapshot_workspace(bridge.workspace)
            self.assertEqual({a.kind for a in artifacts}, {'plan', 'literature'})

    def test_literature_review_is_literature_not_terminal_review(self):
        self.assertEqual(artifact_kind(Path('state/agent_laboratory/literature_review.txt')), 'literature')
        self.assertEqual(artifact_kind(Path('state/agent_laboratory/report_refinement_review.txt')), 'review')

    def test_ai_402_is_preserved_before_native_retry_wrapper(self):
        class Rejected(Exception):
            status_code = 402
        async def rejected(**kwargs):
            raise Rejected('provider response')
        core = ModuleType('research_agent.inno.core')
        core.acompletion = rejected
        research = ModuleType('research_agent')
        inno = ModuleType('research_agent.inno')
        research.inno, inno.core = inno, core
        bridge = object.__new__(AIResearcherBridge)
        bridge.initial_budget = Budget(20, 1000, 1000, 10, 100, 10)
        bridge.usage = Usage()
        bridge.model, bridge.api_key = 'fake', 'fake'
        with patch.dict(sys.modules, {'research_agent': research, 'research_agent.inno': inno, 'research_agent.inno.core': core}), patch.dict(os.environ, {'API_BASE_URL': 'http://localhost'}):
            bridge._install_usage_adapter()
            with self.assertRaisesRegex(RuntimeError, 'Error code: 402'):
                asyncio.run(core.acompletion())

    def test_d2p_forward_stage_return_advances_past_skipped_phases(self):
        class Stage(Enum):
            GOAL = 1
            LITERATURE = 2
            PLAN = 3
        mapping = {'research_goal': (Stage.GOAL,), 'literature_review_goal': (Stage.LITERATURE,), 'hypothesis_plan': (Stage.PLAN,)}
        with tempfile.TemporaryDirectory() as directory:
            bridge = bridge_for(Path(directory), Stage.PLAN)
            bridge.stage = Stage
            bridge.completed.add('data_exploration')
            with patch.object(DataToPaperBridge, '_available_cards', return_value=[SimpleNamespace(capability_id='research_goal', available=True)]), patch.object(DataToPaperBridge, '_stages_for', side_effect=lambda c: mapping.get(c, ())):
                result = bridge.invoke('research_goal', None)
            self.assertIsNone(result.error)
            self.assertEqual(bridge._next_native(), 'hypothesis_plan')
            self.assertTrue(result.output.strip())
