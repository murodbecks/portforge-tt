"""CHIA cluster configuration and pinned SSH jobs; no SSH aliases or cloud creation."""
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time

from chia.cluster.config import build_config, load_raw_config
from runner import safe_file, write_json


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load(path):
    raw = load_raw_config(str(path))
    native = build_config(raw)
    endpoints = {}
    for role in ('tt', 'reference'):
        options = dict(raw['porting'][role])
        node = native.node_types[role]
        if len(node.compatible_ips) != 1:
            raise ValueError('select exactly one host per role; edit compatible_ips to replace it')
        host = node.compatible_ips[0]
        auth = native.get_ssh_auth(host)
        if not re.fullmatch(r'[A-Za-z0-9_.:-]+', host) or host.startswith('-'):
            raise ValueError('invalid host')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', auth.ssh_user) or auth.ssh_user.startswith('-'):
            raise ValueError('invalid SSH user')
        if not auth.ssh_private_key:
            raise ValueError('explicit SSH key path required')
        for key in ('work_dir', 'python'):
            if not Path(options[key]).is_absolute() or '\n' in options[key]:
                raise ValueError('runtime paths must be absolute')
        if role == 'tt' and 'weights_by_model' in options:
            from models import model_spec
            weights = options['weights_by_model']
            if not isinstance(weights, dict) or not weights:
                raise ValueError('weights_by_model must map model keys to checkpoint paths')
            for model, location in weights.items():
                model_spec(model)
                if (not isinstance(location, str) or not Path(location).is_absolute()
                        or '..' in Path(location).parts or '\n' in location):
                    raise ValueError('checkpoint paths must be absolute without traversal')
        endpoints[role] = dict(options, host=host, user=auth.ssh_user,
            key=str(Path(auth.ssh_private_key).expanduser()), proxy=auth.ssh_proxy_command,
            resources=node.resources)
    if endpoints['reference']['executor'] not in ('direct', 'slurm'):
        raise ValueError('reference executor must be direct or slurm')
    return endpoints


def ssh(endpoint, command, data=None, timeout=90):
    args = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
            '-i', endpoint['key'], '-l', endpoint['user']]
    for key, option in (('known_hosts', 'UserKnownHostsFile'), ('host_key_alias', 'HostKeyAlias'), ('proxy', 'ProxyCommand')):
        if endpoint.get(key):
            value = str(Path(endpoint[key]).expanduser()) if key == 'known_hosts' else endpoint[key]
            args += ['-o', option + '=' + value]
    args += ['--', endpoint['host'], command]
    result = subprocess.run(args, input=data, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-1500:])
    return result.stdout


def upload(endpoint, directory, files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name, data in files.items():
            safe_file('/tmp/portforge-upload', name)
            data = data.encode() if isinstance(data, str) else data
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o600
            archive.addfile(info, io.BytesIO(data))
    ssh(endpoint, 'mkdir -p ' + shlex.quote(directory) + ' && tar -xf - -C ' + shlex.quote(directory), stream.getvalue(), 300)


def activated(run):
    return json.loads((Path(run) / 'cluster.json').read_text())


def remote(config, run, request, timeout=90):
    endpoint = config['tt']
    root = endpoint['work_dir'] + '/' + Path(run).name
    command = shlex.join([endpoint['python'], root + '/harness/runner.py', root])
    return json.loads(ssh(endpoint, command, json.dumps(request).encode(), timeout))


def reference_job(endpoint, directory, command, gpu, record):
    """Pin a submitted job to its endpoint; reconnect never silently submits twice."""
    q = shlex.quote
    script = '#!/bin/bash\n'
    if endpoint['executor'] == 'slurm':
        settings = endpoint['slurm']
        script += (f'#SBATCH --partition={settings["partition"]}\n#SBATCH --nodes=1\n'
                   '#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=12\n#SBATCH --mem=24G\n'
                   '#SBATCH --time=06:00:00\n#SBATCH --output=launcher.log\n')
        if gpu:
            script += '#SBATCH --gres=gpu:1\n'
    script += ('set -euo pipefail\ncd ' + q(directory) + '\n'
               'exec 9>.job.lock\nflock -n 9 || exit 73\n'
               "trap 'rc=$?; echo \"$rc\" > exit-code.tmp; mv exit-code.tmp exit-code.txt' EXIT\n"
               'export OMP_NUM_THREADS=12 PYTHONNOUSERSITE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1\n'
               'export PORTFORGE_REFERENCE_EXECUTION=' + q(endpoint['executor']) + '\n'
               'export HF_HOME=' + q(endpoint['hf_home']) + '\n')
    if gpu and endpoint['executor'] == 'direct':
        # Separate controllers share this dedicated GPU; serialize measurements
        # and model loading across campaigns, not merely within one Ray runtime.
        script += 'exec 8>' + q(endpoint['work_dir'] + '/.gpu-reference.lock') + '\nflock 8\n'
    if endpoint.get('pythonpath'):
        script += 'export PYTHONPATH=' + q(endpoint['pythonpath']) + '\n'
    record = Path(record)
    signature = identity(endpoint)
    if record.exists():
        saved = json.loads(record.read_text())
        if saved['endpoint_sha256'] != signature or saved['directory'] != directory:
            raise ValueError('job endpoint changed; refuse polling a different machine')
    else:
        upload(endpoint, directory, {'job.sh': script + command + '\n'})
        # Save intent before submission: an ambiguous SSH failure needs inspection.
        saved = {'directory': directory, 'endpoint_sha256': signature, 'submission': 'intent'}
        write_json(record, saved)
        launch = ('sbatch --parsable job.sh' if endpoint['executor'] == 'slurm'
                  else 'nohup timeout --kill-after=30s 21600 bash job.sh </dev/null >launcher.log 2>&1 & echo $!')
        job = ssh(endpoint, 'cd ' + q(directory) + ' && { ' + launch + '; }').decode().strip()
        if not job.isdigit():
            raise RuntimeError('invalid remote job ID; submission intent preserved')
        saved.update(job=job, submission='submitted')
        write_json(record, saved)
    if saved['submission'] != 'submitted':
        raise RuntimeError('ambiguous previous submission; inspect remote job before retrying')
    failures, end = 0, time.monotonic() + 7 * 3600
    while time.monotonic() < end:
        try:
            state = ssh(endpoint, f'if test -f {q(directory + "/exit-code.txt")}; then cat {q(directory + "/exit-code.txt")}; else echo pending; fi', timeout=45).decode().strip()
            failures = 0
        except (RuntimeError, subprocess.TimeoutExpired):
            failures += 1
            if failures >= 6:
                raise RuntimeError('reference transport unavailable; job record preserved')
            time.sleep(10)
            continue
        if state != 'pending':
            if state != '0':
                raise RuntimeError('reference failed: ' + ssh(endpoint, 'tail -c 1800 ' + q(directory + '/launcher.log')).decode())
            return
        time.sleep(10)
    raise TimeoutError('reference job deadline; inspect saved remote job')
