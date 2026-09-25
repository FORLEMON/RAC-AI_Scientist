import json
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
import urllib.error
from unittest.mock import patch

from rac_ai_scientist.model_gateway import Account, BudgetExceeded, ModelGateway, ProviderError, Redactor, sse_response
from rac_ai_scientist.queue_runner import QueueRunner, ensure_episode_metadata, ensure_task_spec, run_logged
from rac_ai_scientist.benchmarks.scoring import DiscoveryJudge


class BudgetTests(unittest.TestCase):
    def account(self, **override):
        return Account(dict(max_wall_seconds=30, max_agent_calls=2, max_input_tokens=100000,
                            max_output_tokens=10000, max_provider_cost_usd=.01, **override),
                       {'input_usd_per_million': 2, 'output_usd_per_million': 4})

    def test_reservation_keeps_maximum_charge_inside_dollar_budget(self):
        account = self.account()
        inp, out = account.reserve({'messages': [{'role': 'user', 'content': 'hello'}], 'max_tokens': 10000})
        self.assertLess(out, 10000)
        self.assertLessEqual(account.cost_of(inp, out), .01)
        account.finish(inp, out)
        with self.assertRaises(BudgetExceeded):
            account.reserve({'messages': []})

    def test_failed_reservation_is_not_forwarded_or_counted(self):
        account = self.account()
        account.deadline = 0
        with self.assertRaises(BudgetExceeded):
            account.reserve({'messages': []})
        self.assertEqual(account.calls, 0)

    def test_calls_include_requests_even_before_a_response(self):
        account = self.account()
        account.reserve({'messages': [], 'max_tokens': 1})
        account.reserve({'messages': [], 'max_tokens': 1})
        with self.assertRaises(BudgetExceeded):
            account.reserve({'messages': [], 'max_tokens': 1})

    def test_litellm_null_token_limit_uses_bounded_default(self):
        account = self.account()
        inp, out = account.reserve({'messages': [], 'max_tokens': None})
        self.assertGreater(out, 0)
        self.assertLessEqual(account.cost_of(inp, out), .01)

    def test_provider_429_keeps_retry_guidance_without_fictitious_token_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            gateway = ModelGateway(base_url='https://example.test/v1', api_key='private', model='model',
                budget=self.account().budget, pricing=self.account().pricing, log_dir=Path(directory),
                bind='127.0.0.1', redactor=Redactor())
            error = urllib.error.HTTPError('https://example.test', 429, 'Rate limit',
                {'Retry-After':'12','x-ratelimit-limit-tokens':'10000'}, io.BytesIO(b'{"error":{"code":"RateLimitReached"}}'))
            with patch('rac_ai_scientist.model_gateway.urllib.request.urlopen', side_effect=error):
                with self.assertRaises(ProviderError) as caught:
                    gateway.complete({'model':'model','messages':[],'max_tokens':32})
            self.assertEqual(caught.exception.status,429)
            self.assertEqual(caught.exception.retry_after,'12')
            self.assertEqual(gateway.account.calls,1)
            self.assertEqual(gateway.account.cost,0)
            self.assertEqual(gateway.account.input_tokens,0)

    def test_stream_preserves_tools_reasoning_finish_and_usage(self):
        response = {'id': 'x', 'model': 'm', 'choices': [{'index': 0, 'message': {
            'role': 'assistant', 'content': None, 'reasoning_content': 'reason',
            'tool_calls': [{'id': 't', 'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}}]},
            'finish_reason': 'tool_calls'}], 'usage': {'prompt_tokens': 3, 'completion_tokens': 7}}
        chunks = list(sse_response(response))
        self.assertEqual(chunks[0]['choices'][0]['delta']['tool_calls'][0]['index'], 0)
        self.assertEqual(chunks[0]['choices'][0]['delta']['reasoning_content'], 'reason')
        self.assertEqual(chunks[1]['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(chunks[-1]['usage']['completion_tokens'], 7)

    def test_gpt5_judge_changes_transport_argument_only(self):
        judge = DiscoveryJudge.__new__(DiscoveryJudge)
        judge.model, judge.usage = 'gpt-5.4', {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}
        captured = {}
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='{}'))])
        judge.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        judge.chat(messages=[{'role': 'user', 'content': 'JSON'}], max_tokens=77, temperature=0)
        self.assertEqual(captured['max_completion_tokens'], 77)
        self.assertNotIn('max_tokens', captured)
        self.assertNotIn('temperature', captured)


class QueueFailureTests(unittest.TestCase):
    def test_existing_identical_read_only_task_spec_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'source.json', root / 'target.json'
            source.write_text('{"task":"same"}', encoding='utf-8')
            target.write_text('{"task":"same"}', encoding='utf-8')
            target.chmod(0o444)
            ensure_task_spec(source, target)
            self.assertEqual(target.read_text(encoding='utf-8'), source.read_text(encoding='utf-8'))

    def test_existing_different_task_spec_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'source.json', root / 'target.json'
            source.write_text('{"task":"expected"}', encoding='utf-8')
            target.write_text('{"task":"different"}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'differs from prepared input'):
                ensure_task_spec(source, target)

    def test_identical_episode_metadata_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.json'
            metadata = {'episode_id': 'e', 'host': 'agent_laboratory'}
            path.write_text(json.dumps(metadata), encoding='utf-8')
            path.chmod(0o444)
            with patch('rac_ai_scientist.queue_runner.write_json') as write:
                self.assertEqual(ensure_episode_metadata(path, {}, metadata), metadata)
            write.assert_not_called()

    def test_output_logs_redact_credentials_and_survive_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory)
            code = run_logged([sys.executable, '-c', "import sys;print('sensitive-key');print('rit_abc',file=sys.stderr);sys.exit(7)"],
                env=os.environ.copy(), timeout=10, log_dir=logs, stem='host', redact=Redactor(['sensitive-key']))
            self.assertEqual(code, 7)
            self.assertIn('[REDACTED]', (logs / 'host.stdout.log').read_text())
            self.assertNotIn('rit_abc', (logs / 'host.stderr.log').read_text())

    def test_timeout_calls_cleanup_and_keeps_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            cleaned = []
            code = run_logged([sys.executable, '-u', '-c', "import time;print('started');time.sleep(20)"], env=os.environ.copy(),
                timeout=.5, log_dir=Path(directory), stem='host', redact=Redactor(), cleanup=lambda: cleaned.append(True))
            self.assertEqual(code, 124)
            self.assertEqual(cleaned, [True])
            self.assertIn('started', (Path(directory) / 'host.stdout.log').read_text())

    @unittest.skipUnless(os.name == 'posix', 'controller lock uses Linux fcntl')
    def test_next_episode_runs_after_error_and_completed_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = QueueRunner.__new__(QueueRunner)
            runner.root = Path(directory)
            (runner.root / 'runs').mkdir()
            runner.rows = [{'episode_id': x} for x in ('previous', 'bad', 'next')]
            runner.state = {'episodes': {'previous': {'status': 'completed'}}}
            runner.redact = Redactor()
            runner.initialize = runner.persist = lambda: None
            runner.event = lambda *args, **kwargs: None
            seen = []
            def execute(row, index):
                seen.append(row['episode_id'])
                if row['episode_id'] == 'bad':
                    raise RuntimeError('host crashed')
            runner.execute_episode = execute
            runner.run()
            self.assertEqual(seen, ['bad', 'next'])
            self.assertEqual(runner.state['episodes']['bad']['status'], 'failed')

    def test_failed_host_still_attempts_scoring_after_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'prepared').mkdir()
            (root / 'prepared/task_spec.json').write_text('{}')
            runner = QueueRunner.__new__(QueueRunner)
            runner.root, runner.batch, runner.source = root, root / 'runs/batch', root / 'source'
            runner.batch.mkdir(parents=True)
            runner.redact, runner.state = Redactor(), {'episodes': {}}
            runner.settings = {'host_cpus': 4, 'host_memory': '6g', 'task_cpus': 4, 'task_memory': '8g', 'score_timeout': 1, 'pricing': {}}
            runner.env = {'AGENT_API_BASE': 'http://example', 'AGENT_API_KEY': 'secret'}
            runner.runtime_image = runner.scorer_image = 'image'
            runner.images, runner.bridge_ip = {'ark': 'image'}, '127.0.0.1'
            runner.rows = [{}]
            runner.persist = lambda: None
            runner.event = lambda *args, **kwargs: None
            runner.remove = lambda name: None
            phases = []
            class Runtime:
                token, name = 'runtime-secret', 'runtime'
                def __init__(self, workspace, *args, **kwargs):
                    workspace.mkdir(parents=True)
                def start(self):
                    return SimpleNamespace(url='http://runtime')
                def close(self):
                    phases.append('runtime_closed')
            class Gateway:
                token = 'gateway-secret'
                exhausted = threading.Event()
                account = SimpleNamespace(snapshot=lambda: {})
                def __init__(self, **kwargs):
                    pass
                def start(self):
                    return 'http://gateway'
                def close(self):
                    phases.append('gateway_closed')
            def execute(command, **kwargs):
                phases.append(kwargs['stem'])
                if kwargs['stem'] == 'host':
                    raise RuntimeError('host failed')
                (runner.batch / 'episodes/e/score.json').write_text('{"status":"invalid_submission","total_score":null}')
                return 2
            row = {'episode_id': 'e', 'host': 'ark', 'benchmark_id': 'corebench', 'condition': 'N0',
                   'budget': {'max_wall_seconds': 1, 'max_provider_cost_usd': 40, 'max_input_tokens': 10,
                              'max_output_tokens': 10, 'max_agent_calls': 1, 'max_hops': 1},
                   'model': 'm', 'sharednet_env_file': None, 'task_dir': 'prepared', 'scorer_source': 'private',
                   'scorer_dataset': None, 'seed': 0, 'task_id': 'task'}
            with patch('rac_ai_scientist.queue_runner.DockerTaskRuntime', Runtime), patch('rac_ai_scientist.queue_runner.ModelGateway', Gateway), patch('rac_ai_scientist.queue_runner.run_logged', execute):
                runner.execute_episode(row, 1)
            self.assertEqual(phases, ['host', 'runtime_closed', 'gateway_closed', 'scorer'])
            self.assertEqual(runner.state['episodes']['e']['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
