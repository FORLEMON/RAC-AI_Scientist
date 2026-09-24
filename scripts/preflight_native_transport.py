"""Exercise the actual host model libraries and WSL file-copy permissions."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rac_ai_scientist.model_gateway import ModelGateway, Redactor
from rac_ai_scientist.queue_runner import dotenv, run_logged


def main():
    settings = json.loads((ROOT / 'configs/queues/execution.local.json').read_text())
    values = dotenv(ROOT / '.env')
    output = ROOT / 'runs/preflight/native-transport'
    output.mkdir(parents=True, exist_ok=True)
    bind = json.loads(subprocess.check_output(['docker', 'network', 'inspect', 'bridge']))[0]['IPAM']['Config'][0]['Gateway']
    redact = Redactor([values['AGENT_API_KEY']])
    gateway = ModelGateway(base_url=values['AGENT_API_BASE'], api_key=values['AGENT_API_KEY'],
        model=values['AGENT_MODEL_NAME'], bind=bind, redactor=redact, log_dir=output/'gateway',
        budget=dict(max_wall_seconds=600, max_agent_calls=6, max_input_tokens=100000,
                    max_output_tokens=8192, max_provider_cost_usd=.20), pricing=settings['pricing'])
    clients = {
        'ark': "import litellm; r=litellm.completion(model='openai/'+os.environ['AGENT_MODEL_NAME'],api_key=os.environ['AGENT_API_KEY'],messages=[{'role':'user','content':'Reply only OK.'}],max_tokens=None,timeout=90); assert r.choices[0].message.content",
        'agent_laboratory': "from openai import OpenAI; c=OpenAI(api_key=os.environ['AGENT_API_KEY'],base_url=os.environ['AGENT_API_BASE'],max_retries=0); r=c.chat.completions.create(model=os.environ['AGENT_MODEL_NAME'],messages=[{'role':'user','content':'Reply only OK.'}],max_tokens=32); assert r.choices[0].message.content",
        'evo_scientist': "from langchain_openai import ChatOpenAI; c=ChatOpenAI(model=os.environ['AGENT_MODEL_NAME'],api_key=os.environ['AGENT_API_KEY'],base_url=os.environ['AGENT_API_BASE'],max_tokens=32,max_retries=0); r=c.invoke('Reply only OK.'); assert r.content",
    }
    try:
        base = gateway.start()
        env = {**os.environ, 'AGENT_API_BASE':base, 'OPENAI_API_BASE':base, 'OPENAI_BASE_URL':base,
               'AGENT_API_KEY':gateway.token, 'AGENT_MODEL_NAME':values['AGENT_MODEL_NAME']}
        for host, code in clients.items():
            command = ['docker','run','--rm','--name','rac-native-transport-'+host,'--cap-drop','ALL','--cap-add','FOWNER',
                '--mount',f'type=bind,source={output},target=/check']
            for key in ('AGENT_API_BASE','OPENAI_API_BASE','OPENAI_BASE_URL','AGENT_API_KEY','AGENT_MODEL_NAME'):
                command += ['--env', key]
            code = "import os,shutil,pathlib; shutil.copy2('/etc/hostname','/check/"+host+".copy'); " + code + "; print('native model client and file copy passed')"
            command += ['--entrypoint','python',settings['host_images'][host],'-c',code]
            rc = run_logged(command,env=env,timeout=150,log_dir=output,stem=host,redact=redact,
                cleanup=lambda h=host: subprocess.run(['docker','rm','-f','rac-native-transport-'+h],capture_output=True))
            if rc:
                raise RuntimeError(f'{host} native transport failed; see {output}/{host}.stderr.log')
            print(host+': native client passed',flush=True)
        (output/'result.json').write_text(json.dumps({'hosts':list(clients),'passed':True,'usage':gateway.account.snapshot()})+'\n')
    finally:
        gateway.close()


if __name__ == '__main__':
    main()
