import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rac_ai_scientist.ark_recovery import AdmissionGate, validate_startup_recovery
from rac_ai_scientist.cli import _empty_external_workspace
from rac_ai_scientist.benchmarks.base import copy_prepared
from test_benchmark_adapters import discovery_fixture
from rac_ai_scientist.benchmarks import get_adapter


class ArkVolumeTests(unittest.TestCase):
    def test_only_empty_mounted_ark_directory_is_accepted_and_inputs_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            episode=root/'episode';workspace=episode/'workspace';volume=workspace/'.conda_env'
            volume.mkdir(parents=True)
            source=root/'source';discovery_fixture(source)
            prepared=root/'prepared'
            get_adapter('discoverybench').prepare(source,prepared,task_id='topic/metadata_0/0/0',split='train')
            self.assertFalse(_empty_external_workspace(episode,'ark','http://runtime'))
            with patch.object(Path,'is_mount',return_value=True):
                self.assertTrue(_empty_external_workspace(episode,'ark','http://runtime'))
                self.assertFalse(_empty_external_workspace(episode,'evo_scientist','http://runtime'))
                self.assertFalse(_empty_external_workspace(episode,'ark',None))
                with self.assertRaises(FileExistsError):copy_prepared(prepared,workspace,allow_empty=True)
                copy_prepared(prepared,workspace,allow_empty=True,allow_empty_mounts=('.conda_env',))
                self.assertTrue((workspace/'data/data.csv').is_file())
                self.assertFalse(any(volume.iterdir()))
                self.assertFalse(_empty_external_workspace(episode,'ark','http://runtime'))
                with self.assertRaises(FileExistsError):
                    copy_prepared(prepared,workspace,allow_empty=True,allow_empty_mounts=('.conda_env',))

    def test_existing_volume_data_or_episode_metadata_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode=Path(temporary);volume=episode/'workspace/.conda_env';volume.mkdir(parents=True)
            with patch.object(Path,'is_mount',return_value=True):
                (volume/'old').write_text('prior dependency state')
                self.assertFalse(_empty_external_workspace(episode,'ark','http://runtime'))
                (volume/'old').unlink()
                (episode/'episode.json').write_text('{}')
                self.assertFalse(_empty_external_workspace(episode,'ark','http://runtime'))


class AdmissionTests(unittest.TestCase):
    def test_second_slot_needs_response_and_continuously_low_cpu_and_memory_headroom(self):
        gate=AdmissionGate()
        sample={'cpu_percent':10,'wsl_available_memory_mb':10000}
        self.assertFalse(gate.observe(sample,False,0))
        self.assertFalse(gate.observe(sample,True,10))
        self.assertFalse(gate.observe(dict(sample,cpu_percent=50),True,60))
        self.assertFalse(gate.observe(sample,True,65))
        self.assertFalse(gate.observe(dict(sample,wsl_available_memory_mb=3000),True,120))
        self.assertFalse(gate.observe(sample,True,125))
        self.assertFalse(gate.observe(sample,True,184))
        self.assertTrue(gate.observe(sample,True,185))

    def test_recovery_rejects_any_prior_model_usage(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent=Path(temporary)
            (parent/'frozen-plan.json').write_text(json.dumps({'episodes':[{'episode_id':'old','planned_episode_id':'plan'}]}))
            (parent/'status.json').write_text(json.dumps({'episodes':{'old':{'status':'failed','host_exit_code':1}}}))
            logs=parent/'logs/old';(logs/'model').mkdir(parents=True)
            (logs/'host.stderr.log').write_text('episode directory already exists: example')
            usage=logs/'model/usage.json';usage.write_text('{"calls":0,"budget_cost_usd":0}')
            rows=[{'planned_episode_id':'plan'}]
            validate_startup_recovery(parent,rows)
            self.assertEqual(rows[0]['recovery_of'],'old')
            usage.write_text('{"calls":1,"budget_cost_usd":0.01}')
            with self.assertRaises(ValueError):validate_startup_recovery(parent,rows)


if __name__=='__main__':unittest.main()
