"""Explicit ARK startup recovery alongside an already running smoke batch."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import traceback

from .queue_runner import QueueRunner
from .resource_control import CpuGovernor


class AdmissionGate:
    """Require a real first model response and sustained machine headroom."""
    def __init__(self, cpu_limit=50, seconds=60, minimum_memory_mb=6144):
        self.cpu_limit, self.seconds, self.minimum_memory_mb = cpu_limit, seconds, minimum_memory_mb
        self.low_since = None

    def observe(self, sample, model_responded, now):
        healthy = (model_responded and 'windows_probe_error' not in sample and sample['cpu_percent'] < self.cpu_limit
                   and sample['wsl_available_memory_mb'] >= self.minimum_memory_mb)
        self.low_since = (now if self.low_since is None else self.low_since) if healthy else None
        return self.low_since is not None and now - self.low_since >= self.seconds


def validate_startup_recovery(parent, rows):
    """Do not silently reuse rooms or budgets from a previous scientific run."""
    old_rows = {x['episode_id']: x for x in json.loads((parent/'frozen-plan.json').read_text())['episodes']}
    state = json.loads((parent/'status.json').read_text())
    for row in rows:
        matching = [eid for eid, old in old_rows.items()
                    if old.get('planned_episode_id', eid) == row['planned_episode_id']]
        if len(matching) != 1:
            raise ValueError('recovery requires one matching parent episode')
        eid = matching[0]
        episode = state['episodes'].get(eid, {})
        usage = json.loads((parent/'logs'/eid/'model/usage.json').read_text())
        stderr = (parent/'logs'/eid/'host.stderr.log').read_text()
        if (episode.get('status') != 'failed' or episode.get('host_exit_code') != 1
                or usage.get('calls') != 0 or usage.get('budget_cost_usd') != 0
                or 'episode directory already exists:' not in stderr):
            raise ValueError('only zero-call ARK directory-startup failures can be recovered here')
        row['recovery_of'] = eid


class ArkRecoveryRunner(QueueRunner):
    def __init__(self, root, batch, parent_batch):
        if Path(parent_batch).name != parent_batch or parent_batch in {'.','..',batch}:
            raise ValueError('parent batch must be a different safe path component')
        super().__init__(root,batch)
        self.rows = [row for row in self.rows if row['host'] == 'ark']
        self.parent = self.root/'runs'/parent_batch
        validate_startup_recovery(self.parent,self.rows)
        self.host_order = ('ark',)
        # New containers only; the two existing Agent Laboratory jobs continue.
        self.settings.update(host_order=['ark'],host_cpus=1,task_cpus=2,host_memory='3g',task_memory='2g')
        self.settings['admission'] = dict(cpu_below_percent=50,stable_seconds=60,minimum_available_memory_mb=6144)
        self.state.update(host_order=['ark'],supplement_to=parent_batch,admitted_parallelism=1,
                          admission=self.settings['admission'])
        self.admission = AdmissionGate()
        self.second_slot = threading.Event()

    def resource_report(self,sample):
        if not self.second_slot.is_set():
            # A completed response proves the first ARK passed startup. No
            # model messages, credentials or source datasets are inspected.
            responded = any((self.batch/'logs'/row['episode_id']/'model/000001.response.json').is_file()
                            for row in self.rows)
            if self.admission.observe(sample,responded,time.monotonic()):
                with self.state_lock:
                    self.state['admitted_parallelism'] = 2
                self.event('ark_second_slot_admitted',cpu_percent=sample['cpu_percent'],
                           available_memory_mb=sample['wsl_available_memory_mb'])
                self.second_slot.set()
        super().resource_report(sample)

    def lane(self,items):
        for index,row in items:
            if self.stop_event.is_set():return
            if row['episode_id'] in self.state['episodes']:
                self.event('episode_not_retried',episode_id=row['episode_id'])
                continue
            try:
                self.execute_episode(row,index)
            except Exception:
                self.save_episode(row['episode_id'],{'status':'failed','host':'ark',
                    'benchmark':row['benchmark_id'],'condition':row['condition'],
                    'error':self.redact(traceback.format_exc())})
                self.event('episode_controller_failed',episode_id=row['episode_id'])
            finally:
                with self.governor.lock:
                    control=self.governor.episodes.get(row['episode_id'])
                    if control:self.governor.finish(control)

    def run_lanes(self):
        lanes={}
        for index,row in enumerate(self.rows,1):
            lanes.setdefault(row['benchmark_id'],[]).append((index,row))
        first,second=list(lanes.values())
        with ThreadPoolExecutor(max_workers=2) as pool:
            future=pool.submit(self.lane,first)
            self.event('ark_first_lane_started',benchmark=first[0][1]['benchmark_id'])
            try:
                while not self.second_slot.wait(1):
                    if self.stop_event.is_set():return
                    if future.done():
                        future.result()
                        self.event('ark_second_lane_serial_fallback')
                        break
                next_future=pool.submit(self.lane,second)
                future.result()
                next_future.result()
            except BaseException:
                self.stop_event.set()
                raise

    def run(self):
        import fcntl
        # This is the user-authorized extra ARK queue, not a second copy of
        # the main 18-episode runner. One recovery per parent is allowed.
        with (self.parent/'ark-recovery.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            record=self.parent/'ark-recovery.json'
            if record.exists():
                raise FileExistsError('this parent already has an ARK recovery; inspect it before any rerun')
            validate_startup_recovery(self.parent,self.rows)
            self.initialize()
            record.write_text(json.dumps({'batch':self.batch.name,'parent_batch':self.parent.name,
                'reason':'user requested parallel recovery of zero-call directory-startup failures',
                'created_at':time.time()},indent=2)+'\n')
            self.state['status']='running'
            self.persist()
            self.governor=CpuGovernor(self.settings['cpu_guard'],self.event,self.resource_report)
            self.governor.start()
            try:
                self.run_lanes()
            finally:
                self.governor.close()
            self.state['status']='finished'
            self.persist()
            self.event('queue_finished',count=len(self.rows))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default=str(Path.cwd()))
    parser.add_argument('--batch',required=True)
    parser.add_argument('--parent-batch',required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    runner=ArkRecoveryRunner(Path(args.root),args.batch,args.parent_batch)
    if args.check:
        print(json.dumps({'valid':True,'episodes':len(runner.rows),'hosts':list(runner.host_order),
                          'parent':args.parent_batch,'initial_parallelism':1,'maximum_parallelism':2}))
    else:runner.run()


if __name__=='__main__':main()
