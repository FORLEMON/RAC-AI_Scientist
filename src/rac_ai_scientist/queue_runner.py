"""Two benchmark lanes per host, with scoring, isolation and CPU protection."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
import traceback

from .benchmarks.base import digest, write_json
from .config import load_config, validate_config
from .matrix import expand_matrix
from .model_gateway import ModelGateway, Redactor
from .sharednet import SharedNetInvite, load_sharednet_env
from .task_runtime import DockerTaskRuntime
from .resource_control import CpuGovernor


HOSTS = ('evo_scientist', 'ark', 'agent_laboratory')


def dotenv(path):
    result = {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        result[key.strip()] = value
    return result


def run_logged(command, *, env, timeout, log_dir, stem, redact, cleanup=None, cancelled=None, started=None, stage_ended=None):
    """Drain both streams continuously; a timeout removes the owned container."""
    log_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, start_new_session=(os.name == 'posix'))
    def drain(pipe, path):
        with path.open('w', encoding='utf-8', buffering=1) as stream:
            for line in iter(pipe.readline, b''):
                stream.write(redact(line.decode('utf-8', 'replace')))
        pipe.close()
    readers = []
    for name, pipe in (('stdout', proc.stdout), ('stderr', proc.stderr)):
        thread = threading.Thread(target=drain, args=(pipe, log_dir / f'{stem}.{name}.log'), daemon=True)
        thread.start()
        readers.append(thread)
    timed_out = False
    budget_stop = False
    deadline = time.monotonic() + timeout
    try:
        if started:
            started()
        while proc.poll() is None:
            if cancelled and cancelled():
                budget_stop = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                proc.wait(timeout=min(1, remaining))
            except subprocess.TimeoutExpired:
                pass
    finally:
        if stage_ended:
            stage_ended()
        if cleanup:
            cleanup()
        if proc.poll() is None:
            if os.name == 'posix':
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        proc.wait()
        for reader in readers:
            reader.join(timeout=15)
        if any(reader.is_alive() for reader in readers):
            raise RuntimeError('log reader failed to finish after process cleanup')
    return 124 if timed_out else 125 if budget_stop else proc.returncode


class QueueRunner:
    def __init__(self, root: Path, batch: str, *, execution_config: str | Path | None = None):
        if Path(batch).name != batch or batch in {'.', '..'}:
            raise ValueError('batch must be one path component')
        self.root = root.resolve()
        self.env = dotenv(self.root / '.env')
        self.redact = Redactor(v for k, v in self.env.items() if k.endswith(('_KEY', '_TOKEN', '_INVITE')))
        execution_path = Path(execution_config) if execution_config else self.root / 'configs/queues/execution.local.json'
        if not execution_path.is_absolute():
            execution_path = self.root / execution_path
        self.execution_path = execution_path.resolve()
        self.settings = load_config(self.execution_path)
        run_root = Path(self.settings.get('run_root', 'runs'))
        self.run_root = (run_root if run_root.is_absolute() else self.root / run_root).resolve()
        self.batch = self.run_root / batch
        self.batch.mkdir(parents=True, exist_ok=True)
        self.state_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.host_order = tuple(self.settings.get('host_order', HOSTS))
        if not self.host_order or len(set(self.host_order)) != len(self.host_order) or not set(self.host_order).issubset(HOSTS):
            raise ValueError('host_order must contain unique supported hosts')
        if self.settings['global_parallelism'] not in (1, 2, 3, 4):
            raise ValueError('global_parallelism must be between one and four')
        self.governor = None
        self.rows = []
        seen_rooms = set()
        configured_queues = self.settings.get('queue_files')
        queue_sources = []
        if configured_queues:
            for raw in configured_queues:
                path = Path(raw)
                queue_sources.append(path if path.is_absolute() else self.root / path)
        else:
            queue_sources = [self.root / f'configs/queues/{host}.smoke.json' for host in self.host_order]
        self.parallel_queue_lanes = bool(configured_queues)
        for queue_path in queue_sources:
            queue = load_config(queue_path.resolve())
            host = queue['host']
            if host not in self.host_order:
                raise ValueError(f'queue host {host!r} is not enabled by host_order')
            policy = queue.get('policy', {})
            if policy and (policy.get('max_parallel_episodes') != 1
                    or policy.get('score_after_each_episode') is not True
                    or policy.get('continue_on_episode_error') is not True
                    or policy.get('continue_on_scoring_error') is not True
                    or policy.get('retry_automatically') is not False):
                raise ValueError(f'unsafe queue policy in {queue_path}')
            queue_cpuset = queue.get('resources', {}).get('cpuset_cpus')
            if queue_cpuset is not None and not re.fullmatch(r'\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*', queue_cpuset):
                raise ValueError(f'invalid queue CPU set in {queue_path}')
            for item in queue['episodes']:
                row = dict(item, host=host, lane_id=queue['queue_id'], cpuset_cpus=queue_cpuset)
                config_path = self.root / row['config']
                config = load_config(config_path)
                blockers = [f for f in validate_config(config, self.root) if f.level in {'ERROR', 'BLOCKED'}]
                if blockers:
                    raise ValueError(str(blockers))
                expanded = next(x for x in expand_matrix(config, config_path) if x['episode_id'] == row['episode_id'])
                if expanded['config_sha256'] != row['config_sha256'] or digest(self.root / row['task_dir'] / 'task_spec.json') != row['task_spec_sha256']:
                    raise ValueError('queue configuration or prepared task changed since planning')
                # Scope Room envelopes to this execution attempt. Bootstrap
                # diagnostics from an earlier aborted launch must not become
                # scientific context for a new run of the same planned row.
                row['planned_episode_id'] = row['episode_id']
                row['episode_id'] += '__a' + hashlib.sha256(batch.encode()).hexdigest()[:8]
                row['budget'] = config['budget']
                row['model'] = config['model']['name']
                if row['sharednet_env_file']:
                    room = load_sharednet_env(self.root / row['sharednet_env_file'])
                    invite = SharedNetInvite.parse(room.get('SHAREDNET_INVITE', ''), room_id=room.get('SHAREDNET_ROOM_ID', ''))
                    if invite.room_id in seen_rooms:
                        raise ValueError('room reused across episodes')
                    seen_rooms.add(invite.room_id)
                    self.redact.values.add(invite.token)
                self.rows.append(row)
        self.status_path = self.batch / 'status.json'
        self.state = json.loads(self.status_path.read_text()) if self.status_path.exists() else {'batch': batch, 'status': 'prepared', 'global_parallelism': self.settings['global_parallelism'], 'host_order': self.host_order, 'episodes': {}}
        self.source = self.batch / 'controller-source'
        self.bridge_ip = None

    def event(self, event, **fields):
        record = {'at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'event': event, **fields}
        encoded = self.redact(json.dumps(record, ensure_ascii=False))
        with self.state_lock:
            with (self.batch / 'controller.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(encoded + '\n')
            print(encoded, flush=True)

    def persist(self):
        with self.state_lock:
            self.state['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
            temp = self.status_path.with_suffix('.tmp')
            temp.write_text(self.redact(json.dumps(self.state, indent=2)) + '\n', encoding='utf-8')
            temp.replace(self.status_path)

    def save_episode(self, episode_id, result):
        # Each worker owns its result; only immutable snapshots reach the
        # shared state, preventing JSON serialization races between lanes.
        with getattr(self, 'state_lock', threading.RLock()):
            self.state['episodes'][episode_id] = copy.deepcopy(result)
            self.persist()

    def resource_report(self, sample):
        with self.state_lock:
            self.state['resources'] = sample
            self.persist()
            with (self.batch / 'resources.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(sample)+'\n')

    def inspect_image(self, name):
        return subprocess.check_output(['docker', 'image', 'inspect', name, '--format', '{{.Id}}'], text=True, timeout=30).strip()

    def initialize(self):
        if not self.source.exists():
            shutil.copytree(self.root / 'src', self.source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            shutil.copy2(self.root / 'upstream.lock.json', self.batch / 'upstream.lock.json')
            shutil.copy2(self.root / 'benchmark.lock.json', self.batch / 'benchmark.lock.json')
        self.images = {host: self.inspect_image(self.settings['host_images'][host]) for host in self.host_order}
        self.scorer_image = self.inspect_image(self.settings['scorer_image'])
        self.runtime_image = self.inspect_image(self.settings['task_runtime_image'])
        network = json.loads(subprocess.check_output(['docker', 'network', 'inspect', 'bridge'], text=True, timeout=20))
        self.bridge_ip = network[0]['IPAM']['Config'][0]['Gateway']
        self.agent_api_base = self.env['AGENT_API_BASE']
        if relay := self.settings.get('agent_relay'):
            inspected = json.loads(subprocess.check_output(['docker', 'inspect', relay['container']], text=True, timeout=20))[0]
            health = (inspected.get('State', {}).get('Health') or {}).get('Status')
            if health not in (None, 'healthy'):
                raise RuntimeError(f"agent relay is not healthy: {health}")
            relay_env = dict(item.split('=', 1) for item in inspected['Config'].get('Env', []) if '=' in item)
            if relay_env.get('AZURE_AI_MODEL') != relay['expected_model']:
                raise RuntimeError('agent relay deployment does not match the requested model')
            networks = inspected.get('NetworkSettings', {}).get('Networks', {})
            if relay.get('network'):
                networks = {relay['network']: networks.get(relay['network'], {})}
            addresses = [value.get('IPAddress') for value in networks.values() if value.get('IPAddress')]
            if len(addresses) != 1:
                raise RuntimeError('agent relay must resolve to exactly one selected container network address')
            self.agent_api_base = f"http://{addresses[0]}:{int(relay.get('port', 8000))}/v1"
            self.state['agent_relay'] = {'container': relay['container'], 'model': relay['expected_model']}
        self.state.update(host_images=self.images, scorer_image=self.scorer_image, runtime_image=self.runtime_image,
                          pricing=self.settings['pricing'], source_tree_sha256=self.source_hash())
        if extension := self.settings.get('openhands_extensions'):
            original = self.root / extension['path']
            revision = subprocess.check_output(['git','-c',f'safe.directory={original}','-C',str(original),'rev-parse','HEAD'], text=True).strip()
            if revision != extension['revision']:
                raise ValueError('OpenHands public skills revision changed')
            frozen = self.batch / 'openhands-extensions'
            if not frozen.exists():
                shutil.copytree(original, frozen)
            self.state['openhands_extensions_revision'] = revision
        write_json(self.batch / 'frozen-plan.json', {'episodes': self.rows, 'execution': self.settings})
        self.persist()

    def source_hash(self):
        result = hashlib.sha256()
        for path in sorted(self.source.rglob('*.py')):
            result.update(path.relative_to(self.source).as_posix().encode())
            result.update(path.read_bytes())
        return result.hexdigest()

    def remove(self, name):
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=40)

    def base_command(self, name, *, cpuset_cpus=None):
        command = ['docker', 'run', '--rm', '--name', name, '--init', '--cpus', str(self.settings['host_cpus']),
                '--memory', self.settings['host_memory'], '--pids-limit', '1024',
                '--security-opt', 'no-new-privileges', '--cap-drop', 'ALL',
                '--cap-add', 'FOWNER', '--cap-add', 'DAC_OVERRIDE',
                '--mount', f'type=bind,source={self.source},target=/opt/integration/src,readonly',
                '--env', 'PYTHONPATH=/opt/integration/src', '--env', 'PYTHONUNBUFFERED=1',
                '--env', 'PYTHONDONTWRITEBYTECODE=1']
        if cpuset := (cpuset_cpus or self.settings.get('host_cpuset_cpus')):
            command[command.index('--memory'):command.index('--memory')] = ['--cpuset-cpus', cpuset]
        return command

    def execute_episode(self, row, index):
        episode = self.batch / 'episodes' / row['episode_id']
        logs = self.batch / 'logs' / row['episode_id']
        logs.mkdir(parents=True, exist_ok=True)
        if episode.exists():
            raise FileExistsError('episode already exists; automatic retries and room reuse are forbidden')
        runtime = gateway = None
        name = f'rac-smoke-{self.batch.name}-{index:02d}'
        result = {'status': 'running', 'host': row['host'], 'benchmark': row['benchmark_id'], 'condition': row['condition'],
                  'episode_dir': str(episode), 'log_dir': str(logs), 'started_at': time.time()}
        control = self.governor.register(row['episode_id'],index) if getattr(self,'governor',None) else None
        self.save_episode(row['episode_id'],result)
        self.event('episode_started', episode_id=row['episode_id'], index=index, total=len(self.rows))
        try:
            runtime = DockerTaskRuntime(episode / 'workspace', self.runtime_image, logs / 'task-runtime',
                wall_seconds=row['budget']['max_wall_seconds'], memory=self.settings['task_memory'],
                cpus=self.settings['task_cpus'],
                cpuset_cpus=row.get('cpuset_cpus') or self.settings.get('task_cpuset_cpus'), bind=self.bridge_ip)
            client = runtime.start()
            self.redact.values.add(runtime.token)
            result['task_container'] = runtime.name
            self.save_episode(row['episode_id'],result)
            gateway = ModelGateway(base_url=getattr(self, 'agent_api_base', self.env['AGENT_API_BASE']), api_key=self.env['AGENT_API_KEY'],
                model=row['model'], budget=row['budget'], pricing=self.settings['pricing'], log_dir=logs / 'model',
                bind=self.bridge_ip, redactor=self.redact)
            url = gateway.start()
            host_env = {**os.environ, 'AGENT_API_BASE': url, 'AGENT_API_KEY': gateway.token, 'AGENT_MODEL_NAME': row['model'],
                        'RAC_TASK_RUNTIME_URL': client.url, 'RAC_TASK_RUNTIME_TOKEN': runtime.token}
            passed = ['AGENT_API_BASE', 'AGENT_API_KEY', 'AGENT_MODEL_NAME', 'RAC_TASK_RUNTIME_URL', 'RAC_TASK_RUNTIME_TOKEN']
            if row['sharednet_env_file']:
                room = load_sharednet_env(self.root / row['sharednet_env_file'])
                host_env.update(room)
                passed += list(room)
            command = self.base_command(name, cpuset_cpus=row.get('cpuset_cpus'))
            if row['host'] == 'ark':
                conversations = logs / 'openhands'
                conversations.mkdir(exist_ok=True)
                command += ['--mount', f'type=bind,source={conversations},target=/root/.openhands']
                # Conda clones thousands of files. Keep host-only dependencies
                # on Docker's Linux disk, outside the bootstrap task runtime.
                # --rm also removes this anonymous volume after the episode.
                command += ['--mount', f'type=volume,target={episode / "workspace/.conda_env"}']
                if self.settings.get('openhands_extensions'):
                    command += ['--mount', f'type=bind,source={self.batch / "openhands-extensions"},target=/opt/openhands-extensions,readonly',
                        '--env','GIT_CONFIG_COUNT=3','--env','GIT_CONFIG_KEY_0=safe.directory','--env','GIT_CONFIG_VALUE_0=*',
                        '--env','GIT_CONFIG_KEY_1=url.file:///opt/openhands-extensions.insteadOf',
                        '--env','GIT_CONFIG_VALUE_1=https://github.com/OpenHands/extensions.git',
                        '--env','GIT_CONFIG_KEY_2=url.file:///opt/openhands-extensions.insteadOf',
                        '--env','GIT_CONFIG_VALUE_2=https://github.com/OpenHands/extensions']
            command += ['--mount', f'type=bind,source={episode},target={episode}',
                        '--mount', f'type=bind,source={self.root / row["task_dir"]},target=/input/task,readonly',
                        '--mount', f'type=bind,source={self.batch / "upstream.lock.json"},target=/opt/integration/upstream.lock.json,readonly',
                        '--mount', f'type=bind,source={self.batch / "benchmark.lock.json"},target=/opt/integration/benchmark.lock.json,readonly']
            for key in passed:
                command += ['--env', key]
            command += [self.images[row['host']], 'run-one', '--project-root', '/opt/integration', '--host', row['host'],
                        '--condition', row['condition'], '--task-dir', '/input/task', '--run-root', str(episode.parent),
                        '--episode-id', row['episode_id'], '--seed', str(row['seed']), '--model', row['model']]
            for key, flag in [('max_provider_cost_usd', 'max-cost-usd'), ('max_input_tokens', 'max-input-tokens'),
                              ('max_output_tokens', 'max-output-tokens'), ('max_agent_calls', 'max-agent-calls'),
                              ('max_wall_seconds', 'max-wall-seconds'), ('max_hops', 'max-hops')]:
                command += ['--' + flag, str(row['budget'][key])]
            result['host_exit_code'] = run_logged(command, env=host_env, timeout=row['budget']['max_wall_seconds'],
                log_dir=logs, stem='host', redact=self.redact, cleanup=lambda: self.remove(name),
                cancelled=lambda: gateway.exhausted.is_set() or (getattr(self,'stop_event',None) and self.stop_event.is_set()),
                started=(lambda:control.stage_start([name,runtime.name])) if control else None,
                stage_ended=control.stage_end if control else None)
        except Exception:
            result['host_exit_code'] = result.get('host_exit_code', -1)
            result['error'] = self.redact(traceback.format_exc())
            (logs / 'controller-error.log').write_text(result['error'], encoding='utf-8')
        finally:
            for resource in (runtime, gateway):
                if resource:
                    try:
                        resource.close()
                    except Exception as exc:
                        self.event('cleanup_error', episode_id=row['episode_id'], error=self.redact(str(exc)))
            if gateway:
                result['model_usage'] = gateway.account.snapshot()
        # Scoring happens after host/runtime termination, including failed runs.
        result['status'] = 'scoring'
        self.save_episode(row['episode_id'],result)
        try:
            episode.mkdir(parents=True, exist_ok=True)
            spec_source = self.root / row['task_dir'] / 'task_spec.json'
            shutil.copy2(spec_source, episode / 'task_spec.json')
            metadata_path = episode / 'episode.json'
            try:
                metadata = json.loads(metadata_path.read_text())
            except (OSError, ValueError):
                metadata = {'status': 'failed', 'reason': result.get('error', 'host did not produce metadata')}
            metadata.update(episode_id=row['episode_id'], benchmark_id=row['benchmark_id'], task_id=row['task_id'],
                task_spec_sha256=digest(spec_source), host=row['host'], condition=row['condition'])
            write_json(metadata_path, metadata)
            score_env = {**os.environ, **{k: v for k, v in self.env.items() if k.startswith('JUDGE_')}}
            command = self.base_command(name + '-score', cpuset_cpus=row.get('cpuset_cpus'))
            command += ['--mount', f'type=bind,source={episode},target={episode}',
                        '--mount', f'type=bind,source={self.root / row["scorer_source"]},target=/private/benchmark,readonly']
            if row['scorer_dataset']:
                command += ['--mount', f'type=bind,source={self.root / row["scorer_dataset"]},target=/private/dataset.json,readonly']
            for key in self.env:
                if key.startswith('JUDGE_'):
                    command += ['--env', key]
            command += ['--env', 'GIT_CONFIG_COUNT=1', '--env', 'GIT_CONFIG_KEY_0=safe.directory', '--env', 'GIT_CONFIG_VALUE_0=*',
                        self.scorer_image, 'score-episode', '--episode-dir', str(episode), '--benchmark', '/private/benchmark',
                        '--score-timeout', str(self.settings['score_timeout'])]
            if row['scorer_dataset']:
                command += ['--dataset', '/private/dataset.json']
            result['scorer_exit_code'] = run_logged(command, env=score_env, timeout=self.settings['score_timeout'] + 60,
                log_dir=logs, stem='scorer', redact=self.redact, cleanup=lambda: self.remove(name + '-score'),
                cancelled=(self.stop_event.is_set if getattr(self,'stop_event',None) else None),
                started=(lambda:control.stage_start([name+'-score'])) if control else None,
                stage_ended=control.stage_end if control else None)
            result['score'] = json.loads((episode / 'score.json').read_text())
            result['status'] = 'completed' if result['host_exit_code'] == 0 and result['score'].get('status') == 'scored' else 'failed'
        except Exception:
            result.update(status='failed', scoring_error=self.redact(traceback.format_exc()))
            self.event('scoring_error', episode_id=row['episode_id'], error=result['scoring_error'])
        for tree in (logs, episode / 'scoring'):
            if tree.exists():
                for file in tree.rglob('*'):
                    if file.is_file() and not file.is_symlink() and file.suffix in {'.json', '.jsonl', '.log'}:
                        text = file.read_text(encoding='utf-8', errors='replace')
                        cleaned = self.redact(text)
                        if cleaned != text:
                            file.write_text(cleaned, encoding='utf-8')
        result['finished_at'] = time.time()
        if control:
            self.governor.finish(control)
            result['cpu_pause_seconds'] = control.pause_seconds
        self.save_episode(row['episode_id'],result)
        self.event('episode_finished', episode_id=row['episode_id'], status=result['status'], host_exit=result.get('host_exit_code'),
                   score_status=result.get('score', {}).get('status'), score=result.get('score', {}).get('total_score'))

    def run_lanes(self):
        def lane(items):
            for index, row in items:
                if getattr(self,'stop_event',None) and self.stop_event.is_set():
                    return
                previous = self.state['episodes'].get(row['episode_id'])
                if previous:
                    self.event('episode_not_retried', episode_id=row['episode_id'], previous_status=previous['status'])
                    continue
                try:
                    self.execute_episode(row, index)
                except Exception:
                    self.save_episode(row['episode_id'],{'status':'failed','error':self.redact(traceback.format_exc())})
                    self.event('episode_controller_failed', episode_id=row['episode_id'])
                finally:
                    if getattr(self,'governor',None):
                        with self.governor.lock:
                            control=self.governor.episodes.get(row['episode_id'])
                            if control:self.governor.finish(control)
        settings=getattr(self,'settings',{})
        pool=ThreadPoolExecutor(max_workers=settings.get('global_parallelism',1))
        try:
            if getattr(self, 'parallel_queue_lanes', False):
                lanes = {}
                for index, row in enumerate(self.rows, 1):
                    lanes.setdefault(row['lane_id'], []).append((index, row))
                self.event('parallel_queues_started', lanes=len(lanes))
                futures = [pool.submit(lane, items) for items in lanes.values()]
                for future in futures:
                    future.result()
                self.event('parallel_queues_finished', lanes=len(lanes))
                return
            for host in getattr(self,'host_order',HOSTS):
                lanes={}
                for index,row in enumerate(self.rows,1):
                    if row.get('host','ark') == host:
                        lanes.setdefault(row.get('benchmark_id','single'),[]).append((index,row))
                self.event('host_group_started',host=host,lanes=len(lanes))
                futures=[pool.submit(lane,items) for items in lanes.values()]
                for future in futures:
                    future.result()
                self.event('host_group_finished',host=host)
        except BaseException:
            if getattr(self,'stop_event',None):self.stop_event.set()
            raise
        finally:
            pool.shutdown(wait=True,cancel_futures=True)

    def run(self):
        import fcntl
        run_root = getattr(self, 'run_root', self.root / 'runs')
        run_root.mkdir(parents=True, exist_ok=True)
        with (run_root / 'queue-global.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.initialize()
            self.state['status'] = 'running'
            self.persist()
            guard=getattr(self,'settings',{}).get('cpu_guard')
            if guard:
                self.governor=CpuGovernor(guard,self.event,self.resource_report)
                self.governor.start()
            try:
                self.run_lanes()
            finally:
                if getattr(self,'governor',None):self.governor.close()
            self.state['status'] = 'finished'
            self.persist()
            self.event('queue_finished', count=len(self.rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(Path.cwd()))
    parser.add_argument('--batch', default='smoke-20260923')
    parser.add_argument('--execution-config')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    runner = QueueRunner(Path(args.root), args.batch, execution_config=args.execution_config)
    if args.check:
        print(json.dumps({'episodes':len(runner.rows),'valid':True,'parallelism':runner.settings['global_parallelism'],'host_order':runner.host_order}))
    else:
        runner.run()


if __name__ == '__main__':
    main()
