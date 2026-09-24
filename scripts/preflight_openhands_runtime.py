"""One real OpenHands tool call against the isolated task runtime."""
from pathlib import Path
import json
import os
import subprocess
import sys
import argparse
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rac_ai_scientist.queue_runner import dotenv, run_logged
from rac_ai_scientist.model_gateway import ModelGateway, Redactor
from rac_ai_scientist.task_runtime import DockerTaskRuntime


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tools-only',action='store_true')
    args=parser.parse_args()
    settings=json.loads((ROOT/'configs/queues/execution.local.json').read_text())
    values=dotenv(ROOT/'.env')
    output=ROOT/('runs/preflight/openhands-runtime-'+time.strftime('%Y%m%d-%H%M%S'))
    output.mkdir(parents=True,exist_ok=True)
    print('Preflight logs: '+str(output),flush=True)
    workspace=output/'workspace'
    bind=json.loads(subprocess.check_output(['docker','network','inspect','bridge']))[0]['IPAM']['Config'][0]['Gateway']
    runtime=DockerTaskRuntime(workspace,settings['task_runtime_image'],output/'task-runtime',wall_seconds=300,bind=bind)
    redact=Redactor([values['AGENT_API_KEY']])
    gateway=ModelGateway(base_url=values['AGENT_API_BASE'],api_key=values['AGENT_API_KEY'],model=values['AGENT_MODEL_NAME'],
        bind=bind,redactor=redact,log_dir=output/'gateway',pricing=settings['pricing'],
        budget=dict(max_wall_seconds=300,max_agent_calls=8,max_input_tokens=100000,max_output_tokens=16000,max_provider_cost_usd=.5))
    name='rac-openhands-tool-preflight'
    try:
        client=runtime.start()
        client.request('begin', {'episode_id':'native-transport-preflight','task_spec_sha256':None,'wall_seconds':300})
        base=gateway.start()
        redact.values.add(runtime.token)
        env={**os.environ,'RAC_TASK_RUNTIME_URL':client.url,'RAC_TASK_RUNTIME_TOKEN':runtime.token,
             'LLM_BASE_URL':base,'LLM_MODEL':'openai/'+values['AGENT_MODEL_NAME'],'LLM_API_KEY':gateway.token}
        code="""import subprocess,sys
from rac_ai_scientist.task_runtime.hooks import openhands_interpreter
sys.exit(subprocess.call([openhands_interpreter(),'-m','rac_ai_scientist.task_runtime.openhands_entry','--headless','--json','--override-with-envs','-t','Use the terminal tool once to execute: printf runtime-ok > transport-ok.txt . Then finish immediately. This is only a tool connectivity check.']))
"""
        if args.tools_only:
            inner = """import os
from pathlib import Path
from rac_ai_scientist.task_runtime.client import RuntimeClient
from rac_ai_scientist.task_runtime.openhands_entry import install
from openhands.sdk.skills import load_public_skills
from openhands.tools.terminal.impl import TerminalExecutor
from openhands.tools.terminal.definition import TerminalAction
runtime=RuntimeClient(os.environ['RAC_TASK_RUNTIME_URL'],os.environ['RAC_TASK_RUNTIME_TOKEN'])
install(runtime)
skills=load_public_skills()
assert skills, 'local public skills did not load'
tool=TerminalExecutor(working_dir=Path.cwd())
observation=tool(TerminalAction(command='printf runtime-ok > transport-ok.txt'))
print('Public skills and native terminal tool passed',len(skills))
"""
            code="import subprocess,sys; from rac_ai_scientist.task_runtime.hooks import openhands_interpreter; sys.exit(subprocess.call([openhands_interpreter(),'-c',"+repr(inner)+"]))"
        command=['docker','run','--rm','--name',name,'--cpus','4','--memory','6g','--cap-drop','ALL','--cap-add','FOWNER',
            '--mount',f'type=bind,source={workspace},target={workspace}',
            '--mount',f'type=bind,source={ROOT/"src"},target=/opt/integration/src,readonly',
            '--env','PYTHONPATH=/opt/integration/src','--env','PYTHONDONTWRITEBYTECODE=1',
            '--env','OPENHANDS_SUPPRESS_BANNER=1','--workdir',str(workspace)]
        command += ['--mount',f'type=bind,source={ROOT / settings["openhands_extensions"]["path"]},target=/opt/openhands-extensions,readonly',
            '--env','GIT_CONFIG_COUNT=3','--env','GIT_CONFIG_KEY_0=safe.directory','--env','GIT_CONFIG_VALUE_0=*',
            '--env','GIT_CONFIG_KEY_1=url.file:///opt/openhands-extensions.insteadOf','--env','GIT_CONFIG_VALUE_1=https://github.com/OpenHands/extensions.git',
            '--env','GIT_CONFIG_KEY_2=url.file:///opt/openhands-extensions.insteadOf','--env','GIT_CONFIG_VALUE_2=https://github.com/OpenHands/extensions']
        for key in ('RAC_TASK_RUNTIME_URL','RAC_TASK_RUNTIME_TOKEN','LLM_BASE_URL','LLM_MODEL','LLM_API_KEY'):
            command += ['--env',key]
        command += ['--entrypoint','python',settings['host_images']['ark'],'-c',code]
        rc=run_logged(command,env=env,timeout=280,log_dir=output,stem='host',redact=redact,
            cleanup=lambda:subprocess.run(['docker','rm','-f',name],capture_output=True),cancelled=gateway.exhausted.is_set)
        passed=rc==0 and (workspace/'transport-ok.txt').is_file() and (workspace/'transport-ok.txt').read_text()=='runtime-ok'
        print(json.dumps({'exit':rc,'runtime_tool_passed':passed,'usage':gateway.account.snapshot()}),flush=True)
        if not passed:raise RuntimeError('OpenHands tool preflight failed; inspect runs/preflight/openhands-runtime')
    finally:
        gateway.close()
        runtime.close()


if __name__=='__main__':
    main()
