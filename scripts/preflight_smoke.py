"""Minimal real connectivity checks before starting the 18-episode queue."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rac_ai_scientist.queue_runner import dotenv, run_logged
from rac_ai_scientist.model_gateway import ModelGateway, Redactor


def main():
    values = dotenv(ROOT / '.env')
    settings = json.loads((ROOT / 'configs/queues/execution.local.json').read_text())
    output = ROOT / 'runs/preflight/linux'
    output.mkdir(parents=True, exist_ok=True)
    network = json.loads(subprocess.check_output(['docker', 'network', 'inspect', 'bridge'], text=True))
    bind = network[0]['IPAM']['Config'][0]['Gateway']
    redactor = Redactor(v for k, v in values.items() if k.endswith(('_KEY', '_INVITE', '_TOKEN')))
    gateway = ModelGateway(base_url=values['AGENT_API_BASE'], api_key=values['AGENT_API_KEY'], model=values['AGENT_MODEL_NAME'],
        budget=dict(max_wall_seconds=180, max_agent_calls=3, max_input_tokens=10000, max_output_tokens=256, max_provider_cost_usd=.1),
        pricing=settings['pricing'], log_dir=output / 'gateway', bind=bind, redactor=redactor)
    try:
        base = gateway.start()
        process_env = {**os.environ, 'AGENT_API_BASE': base, 'AGENT_API_KEY': gateway.token,
                       'AGENT_MODEL_NAME': values['AGENT_MODEL_NAME']}
        code = """import json,os,urllib.request
from openai import OpenAI
c=OpenAI(api_key=os.environ['AGENT_API_KEY'],base_url=os.environ['AGENT_API_BASE'],max_retries=0,timeout=120)
r=c.chat.completions.create(model=os.environ['AGENT_MODEL_NAME'],messages=[{'role':'user','content':'Reply only OK.'}],max_tokens=32,stream=True,stream_options={'include_usage':True})
chunks=list(r)
assert any(x.choices and x.choices[0].delta.content for x in chunks)
print(json.dumps({'agent_stream':True,'usage_reported':any(x.usage for x in chunks)}))
with urllib.request.urlopen('https://www.sharednet.ai',timeout=20) as response: print(json.dumps({'sharednet_https_status':response.status}))
"""
        command = ['docker', 'run', '--rm', '--name', 'rac-connectivity-agent', '--env', 'AGENT_API_BASE', '--env', 'AGENT_API_KEY',
                   '--env', 'AGENT_MODEL_NAME', '--entrypoint', 'python', settings['host_images']['agent_laboratory'], '-c', code]
        exit_code = run_logged(command, env=process_env, timeout=150, log_dir=output, stem='agent', redact=redactor,
                              cleanup=lambda: subprocess.run(['docker','rm','-f','rac-connectivity-agent'],capture_output=True))
        if exit_code:
            raise RuntimeError('agent container connectivity failed; inspect runs/preflight/linux/agent.stderr.log')
    finally:
        gateway.close()
    code = """import json
from pathlib import Path
from rac_ai_scientist.benchmarks.scoring import DiscoveryJudge
from rac_ai_scientist.benchmarks.upstream import assert_revision
j=DiscoveryJudge()
r=j.chat(messages=[{'role':'user','content':'Return a JSON object with ok set to true.'}],max_tokens=64)
assert json.loads(r.choices[0].message.content)['ok'] is True
assert_revision(Path('/private/discovery'),'discoverybench')
assert_revision(Path('/private/core'),'corebench')
print(json.dumps({'judge':j.model,'usage':j.usage,'benchmark_revisions':True}))
"""
    env = {**os.environ, **{k:v for k,v in values.items() if k.startswith('JUDGE_')}}
    command = ['docker','run','--rm','--name','rac-connectivity-judge','--env','PYTHONPATH=/opt/integration/src',
               '--env','GIT_CONFIG_COUNT=1','--env','GIT_CONFIG_KEY_0=safe.directory','--env','GIT_CONFIG_VALUE_0=*',
               '--mount',f'type=bind,source={ROOT / "src"},target=/opt/integration/src,readonly',
               '--mount',f'type=bind,source={ROOT / "upstreams/discoverybench"},target=/private/discovery,readonly',
               '--mount',f'type=bind,source={ROOT / "upstreams/hal_harness"},target=/private/core,readonly']
    for key in values:
        if key.startswith('JUDGE_'):
            command += ['--env',key]
    command += ['--entrypoint','python',settings['scorer_image'],'-c',code]
    exit_code = run_logged(command,env=env,timeout=180,log_dir=output,stem='judge',redact=redactor,
                          cleanup=lambda: subprocess.run(['docker','rm','-f','rac-connectivity-judge'],capture_output=True))
    if exit_code:
        raise RuntimeError('judge/scorer connectivity failed; inspect runs/preflight/linux/judge.stderr.log')
    (output/'result.json').write_text(json.dumps({'agent':True,'judge':True,'sharednet_https':True,'scorer_revisions':True})+'\n')
    print('Linux container preflight passed: agent, judge, SharedNet HTTPS, frozen benchmark sources.',flush=True)


if __name__ == '__main__':
    main()
