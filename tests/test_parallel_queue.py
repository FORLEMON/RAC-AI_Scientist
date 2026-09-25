import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rac_ai_scientist.queue_runner import HOSTS, QueueRunner
from rac_ai_scientist.model_gateway import Redactor
from rac_ai_scientist.resource_control import CpuGovernor, EpisodeResources


class ParallelQueueTests(unittest.TestCase):
    def test_parallel_controller_uses_batch_lock_with_explicit_cpu_isolation(self):
        runner = QueueRunner.__new__(QueueRunner)
        runner.root = Path('/repo')
        runner.run_root = Path('/runs')
        runner.batch = Path('/runs/evo-r1-r3')
        runner.settings = {
            'controller_lock_scope': 'batch',
            'allow_parallel_controllers': True,
        }
        runner.rows = [{'cpuset_cpus': '8-13'}]

        self.assertEqual(runner.controller_lock_path(), Path('/runs/queue-evo-r1-r3.lock'))

    def test_parallel_controller_requires_opt_in_and_cpu_isolation(self):
        runner = QueueRunner.__new__(QueueRunner)
        runner.root = Path('/repo')
        runner.run_root = Path('/runs')
        runner.batch = Path('/runs/evo-r1-r3')
        runner.settings = {'controller_lock_scope': 'batch'}
        runner.rows = [{'cpuset_cpus': '8-13'}]
        with self.assertRaisesRegex(ValueError, 'allow_parallel_controllers'):
            runner.controller_lock_path()

        runner.settings['allow_parallel_controllers'] = True
        runner.rows = [{'cpuset_cpus': None}]
        with self.assertRaisesRegex(ValueError, 'explicit CPU set'):
            runner.controller_lock_path()

    def test_global_controller_lock_remains_the_default(self):
        runner = QueueRunner.__new__(QueueRunner)
        runner.root = Path('/repo')
        runner.run_root = Path('/runs')
        runner.settings = {}

        self.assertEqual(runner.controller_lock_path(), Path('/runs/queue-global.lock'))

    def test_four_named_lanes_run_in_parallel_and_keep_each_condition_order(self):
        runner = QueueRunner.__new__(QueueRunner)
        runner.settings = {'global_parallelism': 4}
        runner.host_order = ('evo_scientist', 'agent_laboratory')
        runner.parallel_queue_lanes = True
        runner.state_lock = threading.RLock()
        runner.stop_event = threading.Event()
        runner.state = {'episodes': {}}
        runner.redact = Redactor()
        runner.governor = None
        runner.persist = lambda: None
        runner.event = lambda *args, **kwargs: None
        runner.rows = [
            {'lane_id': lane, 'host': host, 'benchmark_id': 'discoverybench',
             'condition': condition, 'episode_id': f'{lane}/{condition}'}
            for lane, host in (
                ('evo-ses', 'evo_scientist'), ('evo-worldbank', 'evo_scientist'),
                ('lab-ses', 'agent_laboratory'), ('lab-worldbank', 'agent_laboratory'))
            for condition in ('N0', 'R1', 'R2', 'R3')
        ]
        barrier = threading.Barrier(4)
        starts = []
        active = set()
        peak = [0]

        def execute(row, index):
            with runner.state_lock:
                starts.append(row.copy())
                active.add(row['lane_id'])
                peak[0] = max(peak[0], len(active))
            if row['condition'] == 'N0':
                barrier.wait(timeout=3)
            time.sleep(.005)
            with runner.state_lock:
                active.remove(row['lane_id'])
            if row['lane_id'] == 'evo-worldbank' and row['condition'] == 'R1':
                raise RuntimeError('one cell failed')

        runner.execute_episode = execute
        runner.run_lanes()
        self.assertEqual(peak[0], 4)
        self.assertEqual(len(starts), 16)
        for lane in ('evo-ses', 'evo-worldbank', 'lab-ses', 'lab-worldbank'):
            self.assertEqual([row['condition'] for row in starts if row['lane_id'] == lane],
                             ['N0', 'R1', 'R2', 'R3'])
        self.assertEqual(runner.state['episodes']['evo-worldbank/R1']['status'], 'failed')

    def test_host_command_uses_shared_six_cpu_affinity(self):
        runner = QueueRunner.__new__(QueueRunner)
        runner.settings = {'host_cpus': 6, 'host_cpuset_cpus': '2-7', 'host_memory': '4g'}
        runner.source = Path('/source')
        command = runner.base_command('test')
        self.assertEqual(command[command.index('--cpus') + 1], '6')
        self.assertEqual(command[command.index('--cpuset-cpus') + 1], '2-7')
        self.assertIn('DAC_OVERRIDE', command)

    def test_lane_cpu_affinity_overrides_global_default(self):
        runner = QueueRunner.__new__(QueueRunner)
        runner.settings = {'host_cpus': 6, 'host_cpuset_cpus': '2-7', 'host_memory': '4g'}
        runner.source = Path('/source')
        command = runner.base_command('test', cpuset_cpus='20-25')
        self.assertEqual(command[command.index('--cpuset-cpus') + 1], '20-25')

    def test_two_lanes_preserve_condition_order_host_barrier_and_continue_after_error(self):
        runner=QueueRunner.__new__(QueueRunner)
        runner.settings={'global_parallelism':2}
        runner.host_order=HOSTS
        runner.state_lock=threading.RLock()
        runner.stop_event=threading.Event()
        runner.state={'episodes':{}}
        runner.redact=Redactor()
        runner.governor=None
        runner.persist=lambda:None
        runner.event=lambda *args,**kwargs:None
        runner.rows=[{'host':h,'benchmark_id':b,'condition':c,'episode_id':f'{h}/{b}/{c}'}
            for h in HOSTS for b in ('discovery','core') for c in ('N0','R1','R3')]
        events=[]
        active=set()
        peak=[0]
        gate={h:threading.Barrier(2) for h in HOSTS}
        def execute(row,index):
            with runner.state_lock:
                active.add(row['episode_id'])
                peak[0]=max(peak[0],len(active))
                events.append(('start',row.copy()))
            if row['condition']=='N0':gate[row['host']].wait(timeout=3)
            time.sleep(.01)
            with runner.state_lock:
                events.append(('score_end',row.copy()))
                active.remove(row['episode_id'])
            if row['host']==HOSTS[0] and row['benchmark_id']=='core' and row['condition']=='R1':
                raise RuntimeError('simulated scoring failure')
        runner.execute_episode=execute
        runner.run_lanes()
        self.assertEqual(peak[0],2)
        starts=[row for kind,row in events if kind=='start']
        self.assertEqual(len(starts),18)
        for h in HOSTS:
            for b in ('discovery','core'):
                self.assertEqual([r['condition'] for r in starts if r['host']==h and r['benchmark_id']==b],['N0','R1','R3'])
        for first,second in zip(HOSTS,HOSTS[1:]):
            last=max(i for i,(kind,row) in enumerate(events) if kind=='score_end' and row['host']==first)
            next_start=min(i for i,(kind,row) in enumerate(events) if kind=='start' and row['host']==second)
            self.assertLess(last,next_start)
        self.assertEqual(runner.state['episodes'][f'{HOSTS[0]}/core/R1']['status'],'failed')

    def test_parallel_status_writers_produce_complete_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=QueueRunner.__new__(QueueRunner)
            runner.state_lock=threading.RLock()
            runner.state={'episodes':{}}
            runner.status_path=Path(directory)/'status.json'
            runner.redact=Redactor()
            threads=[threading.Thread(target=lambda key=k:[runner.save_episode(key,{'step':i}) for i in range(15)]) for k in ('one','two')]
            for thread in threads:thread.start()
            for thread in threads:thread.join()
            state=json.loads(runner.status_path.read_text())
            self.assertEqual(state['episodes'],{'one':{'step':14},'two':{'step':14}})


class CpuGovernorTests(unittest.TestCase):
    def test_overload_pauses_only_later_episode_then_resumes_on_recovery(self):
        calls=[]
        def docker(command,**kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0,stdout='true\n',stderr='')
        settings=dict(pause_above_percent=90,overload_seconds=20,resume_below_percent=70,
            recovery_seconds=10,max_pause_seconds=20,cooldown_seconds=20,sample_seconds=5)
        events=[]
        guard=CpuGovernor(settings,lambda event,**data:events.append(event),lambda x:None)
        first=guard.register('first',1);second=guard.register('second',4)
        first.stage_start(['host-first','task-first'])
        second.stage_start(['host-second','task-second'])
        with patch('rac_ai_scientist.resource_control.subprocess.run',side_effect=docker):
            self.assertEqual(guard.tick(95,0),[])
            self.assertEqual(guard.tick(95,20),['second'])
            self.assertEqual(guard.tick(50,25),['second'])
            self.assertEqual(guard.tick(50,35),[])
        self.assertEqual(events,['cpu_paused','cpu_resumed'])
        self.assertEqual([x for x in calls if x[1]=='pause'],[['docker','pause','host-second'],['docker','pause','task-second']])
        self.assertEqual(second.pause_seconds,15)

    def test_single_task_is_not_paused_and_pause_is_bounded(self):
        settings=dict(pause_above_percent=90,overload_seconds=0,resume_below_percent=70,
            recovery_seconds=10,max_pause_seconds=20,cooldown_seconds=20,sample_seconds=5)
        guard=CpuGovernor(settings,lambda *args,**kwargs:None,lambda x:None)
        first=guard.register('one',1);first.stage_start(['one'])
        with patch('rac_ai_scientist.resource_control.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='true')) as docker:
            self.assertEqual(guard.tick(100,0),[])
            docker.assert_not_called()
            second=guard.register('two',2);second.stage_start(['two'])
            self.assertEqual(guard.tick(100,1),['two'])
            self.assertEqual(guard.tick(100,21),[])


if __name__=='__main__':unittest.main()
