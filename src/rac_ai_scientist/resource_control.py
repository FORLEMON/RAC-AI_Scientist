"""CPU telemetry and bounded suspension of one of two experiment slots."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
import time


class ResourceSampler:
    def __init__(self):
        self.previous = None

    def sample(self):
        values = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
        total, idle = sum(values), values[3] + values[4]
        linux_cpu = 0.0
        if self.previous and total > self.previous[0]:
            linux_cpu = 100 * (1 - (idle-self.previous[1])/(total-self.previous[0]))
        self.previous = (total, idle)
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        report = {'at': time.time(), 'wsl_cpu_percent': round(linux_cpu, 2),
                  'wsl_available_memory_mb': int(memory['MemAvailable'].split()[0])/1024}
        # WSL CPU alone misses load from Windows applications. Interop keeps
        # this short read-only probe on the same user's Windows session.
        executable = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
        script = "(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter \"Name='_Total'\" -ErrorAction Stop).PercentProcessorTime | ConvertTo-Json -Compress"
        try:
            response = subprocess.run([executable,'-NoProfile','-NonInteractive','-WindowStyle','Hidden','-Command',script],
                capture_output=True,timeout=12,check=True)
            report['windows_cpu_percent'] = float(response.stdout.decode('utf-8-sig').strip())
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            report['windows_probe_error'] = type(exc).__name__
        report['cpu_percent'] = max(linux_cpu, report.get('windows_cpu_percent', 0))
        return report


class EpisodeResources:
    def __init__(self, episode_id, index):
        self.episode_id, self.index = episode_id, index
        self.containers = []
        self.paused_at = None
        self.pause_seconds = 0.0
        self.lock = threading.RLock()

    def stage_start(self, containers):
        with self.lock:
            self.containers = list(containers)

    def pause(self, now):
        with self.lock:
            if not self.containers or self.paused_at is not None:
                return False
            # A Docker client may have started before its container exists.
            # Do not freeze half of an episode during that short interval.
            for name in self.containers:
                probe = subprocess.run(['docker','inspect','--format','{{.State.Running}}',name],capture_output=True,text=True,timeout=8)
                if probe.returncode or probe.stdout.strip() != 'true':
                    return False
            paused = []
            try:
                for name in self.containers:
                    subprocess.run(['docker','pause',name],capture_output=True,check=True,timeout=8)
                    paused.append(name)
            except subprocess.SubprocessError:
                if paused:
                    subprocess.run(['docker','unpause',*paused],capture_output=True,timeout=10)
                return False
            self.paused_at = now
            return True

    def resume(self, now=None):
        with self.lock:
            if self.paused_at is None:
                return False
            response = subprocess.run(['docker','unpause',*self.containers],capture_output=True,timeout=10)
            if response.returncode:
                # Exit/cleanup can race with the governor. Only an existing
                # container still paused makes this a failed resume.
                for name in self.containers:
                    state = subprocess.run(['docker','inspect','--format','{{.State.Paused}}',name],capture_output=True,text=True,timeout=8)
                    if state.returncode == 0 and state.stdout.strip() == 'true':
                        raise RuntimeError('failed to resume experiment container '+name)
            self.pause_seconds += max(0, (time.monotonic() if now is None else now)-self.paused_at)
            self.paused_at = None
            return True

    def stage_end(self):
        with self.lock:
            self.resume()
            self.containers = []


class CpuGovernor:
    def __init__(self, settings, event, report):
        self.settings, self.event, self.report = settings, event, report
        self.episodes = {}
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.hot_since = self.cool_since = None
        self.last_resume = float('-inf')

    def register(self, episode_id, index):
        item = EpisodeResources(episode_id,index)
        with self.lock:
            self.episodes[episode_id] = item
        return item

    def finish(self, item):
        with self.lock:
            item.stage_end()
            self.episodes.pop(item.episode_id,None)

    def tick(self, cpu, now):
        with self.lock:
            active = [x for x in self.episodes.values() if x.containers]
            paused = next((x for x in active if x.paused_at is not None), None)
            if paused:
                self.cool_since = (self.cool_since if self.cool_since is not None else now) if cpu < self.settings['resume_below_percent'] else None
                recovered = self.cool_since is not None and now-self.cool_since >= self.settings['recovery_seconds']
                # Bound pauses so sockets and native command timeouts remain
                # responsive. Wall-clock budgets continue to include pauses.
                if recovered or now-paused.paused_at >= self.settings['max_pause_seconds'] or len(active) < 2:
                    paused.resume(now)
                    self.last_resume, self.hot_since, self.cool_since = now, None, None
                    self.event('cpu_resumed',episode_id=paused.episode_id,total_pause_seconds=round(paused.pause_seconds,2))
            elif len(active) >= 2 and cpu >= self.settings['pause_above_percent']:
                self.hot_since = self.hot_since if self.hot_since is not None else now
                if now-self.hot_since >= self.settings['overload_seconds'] and now-self.last_resume >= self.settings['cooldown_seconds']:
                    candidate = max(active,key=lambda item:item.index)
                    if candidate.pause(now):
                        self.event('cpu_paused',episode_id=candidate.episode_id,cpu_percent=round(cpu,2),containers=candidate.containers)
            else:
                self.hot_since = None
            return [x.episode_id for x in active if x.paused_at is not None]

    def start(self):
        def loop():
            sampler = ResourceSampler()
            while not self.stop_event.is_set():
                try:
                    sample = sampler.sample()
                    sample['paused_episodes'] = self.tick(sample['cpu_percent'],time.monotonic())
                    self.report(sample)
                except Exception as exc:
                    self.event('cpu_monitor_error',error=str(exc))
                self.stop_event.wait(self.settings['sample_seconds'])
        self.thread = threading.Thread(target=loop,name='cpu-governor',daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=30)
        with self.lock:
            for item in list(self.episodes.values()):
                item.stage_end()
