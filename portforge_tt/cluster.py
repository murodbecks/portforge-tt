"""Cluster inventory and pinned SSH jobs; no SSH aliases or cloud creation.

The inventory parser reads the small subset of the (CHIA-compatible) cluster
YAML schema PortForge actually uses — ``auth`` (global and per-IP SSH
credentials), ``available_node_types`` (one host per role with its resource
counts) and the ``porting`` section — with ``${VAR}`` environment expansion
in values, matching the semantics the CHIA loader provided. Everything else
in the file is ignored, so an existing cluster.yaml keeps working unchanged.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time

import yaml

if __package__:  # installed as part of the portforge_tt package
    from .runner import safe_file, write_json
else:  # flat: a frozen run harness or a direct script execution
    from runner import safe_file, write_json


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _expand_env_vars(data):
    """Recursively expand ${VAR} references in values, leaving $VAR as-is."""
    import re
    if isinstance(data, str):
        return re.sub(r'\$\{([^}]+)\}', lambda m: os.environ.get(m.group(1), m.group(0)), data)
    if isinstance(data, dict):
        return {_expand_env_vars(k): _expand_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    return data


def load_raw_config(path):
    """Load the inventory YAML and expand environment variables in values."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"cluster inventory not found: {path}")
    with open(path) as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("cluster inventory must be a YAML mapping")
    return _expand_env_vars(raw)


def _ssh_auth(raw, host):
    """Resolve SSH credentials for *host*: per-IP override, else the globals."""
    auth = raw.get('auth') or {}
    override = (auth.get('overrides') or {}).get(host) or {}
    user = override.get('ssh_user', auth.get('ssh_user'))
    key = override.get('ssh_private_key', auth.get('ssh_private_key'))
    proxy = override.get('ssh_proxy_command', auth.get('ssh_proxy_command'))
    if not user:
        raise ValueError(f"no SSH user configured for {host}")
    return user, key, proxy


def load(path):
    raw = load_raw_config(str(path))
    node_types = raw.get('available_node_types') or {}
    endpoints = {}
    for role in ('tt', 'reference'):
        options = dict((raw.get('porting') or {}).get(role) or {})
        if not options:
            raise ValueError(f"porting.{role} section missing from the inventory")
        node = node_types.get(role) or {}
        hosts = node.get('compatible_ips') or []
        if len(hosts) != 1:
            raise ValueError('select exactly one host per role; edit compatible_ips to replace it')
        host = hosts[0]
        user, key, proxy = _ssh_auth(raw, host)
        if not re.fullmatch(r'[A-Za-z0-9_.:-]+', host) or host.startswith('-'):
            raise ValueError('invalid host')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', user) or user.startswith('-'):
            raise ValueError('invalid SSH user')
        if not key:
            raise ValueError('explicit SSH key path required')
        for key_name in ('work_dir', 'python'):
            if not Path(options[key_name]).is_absolute() or '\n' in options[key_name]:
                raise ValueError('runtime paths must be absolute')
        if role == 'tt' and 'weights_by_model' in options:
            if __package__:
                from .models import model_spec
            else:
                from models import model_spec
            weights = options['weights_by_model']
            if not isinstance(weights, dict) or not weights:
                raise ValueError('weights_by_model must map model keys to checkpoint paths')
            for model, location in weights.items():
                model_spec(model)
                if (not isinstance(location, str) or not Path(location).is_absolute()
                        or '..' in Path(location).parts or '\n' in location):
                    raise ValueError('checkpoint paths must be absolute without traversal')
        endpoints[role] = dict(options, host=host, user=user,
            key=str(Path(key).expanduser()), proxy=proxy,
            resources=node.get('resources') or {})
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
        # and model loading across campaigns, not merely within one controller.
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
