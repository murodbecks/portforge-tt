"""TT host: isolated jobs, exclusive card leases, phase deadlines and quarantine.

Protocol: python runner.py ROOT < request.json. Only the controller calls this.
No reset operation exists. A quarantined card needs an explicit healthy preflight.
Host RSS is sampled outside the sandbox once per second, including descendants
that create a session. Summing RSS double-counts shared pages; this is a
conservative observed limit, not a cgroup hard cap or a peak between samples.
Trusted runtime.json host_memory_limit_bytes defaults to 64 GiB (1–128 GiB).
Trusted full_timeout_seconds defaults to six hours (bounded to 1–24 hours).
This changes only the full job's outer deadline, not phase or initialization limits.
Only Linux /proc provides RSS accounting; non-Linux unit-test runs report null.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from models import model_spec

METAL = PYTHON = HEALTH = LEASES = None
MOUNTS, CARD_ALLOWLIST = (), ()
WEIGHTS_BY_MODEL = {}
HOST_MEMORY_LIMIT_BYTES = 64 * 1024**3
FULL_TIMEOUT_SECONDS = 21600
PCI_PATTERN = r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]'


def full_job_timeout(config):
    """Read the bounded outer deadline from trusted machine configuration."""
    value = config.get('full_timeout_seconds', 21600)
    if type(value) is not int or not 3600 <= value <= 86400:
        raise ValueError('full_timeout_seconds must be an integer between 3600 and 86400')
    return value


def configure(config):
    """Load trusted host configuration; candidates never supply mounts or paths."""
    global METAL, PYTHON, HEALTH, LEASES, MOUNTS, CARD_ALLOWLIST, WEIGHTS_BY_MODEL
    global HOST_MEMORY_LIMIT_BYTES, FULL_TIMEOUT_SECONDS
    memory_limit = config.get('host_memory_limit_bytes', 64 * 1024**3)
    if type(memory_limit) is not int or not 1024**3 <= memory_limit <= 128 * 1024**3:
        raise ValueError('host_memory_limit_bytes must be an integer between 1 and 128 GiB')
    full_timeout = full_job_timeout(config)
    def absolute(value):
        if not isinstance(value, str) or not Path(value).is_absolute() or '..' in Path(value).parts:
            raise ValueError('runtime paths must be absolute without traversal')
        return Path(value)
    paths = [absolute(config[key]) for key in ('metal', 'python', 'health_dir', 'lease_dir')]
    weights = config.get('weights_by_model', {})
    if not isinstance(weights, dict):
        raise ValueError('weights_by_model must map registry keys to paths')
    weights = dict(weights)
    if not weights:
        raise ValueError('configure at least one checkpoint')
    for name in weights:
        model_spec(name)
    weights = {name: absolute(value) for name, value in weights.items()}
    mounts, allowlist = config.get('runtime_mounts', []), config.get('cards', [])
    if not isinstance(mounts, list) or not isinstance(allowlist, list):
        raise ValueError('runtime_mounts and cards must be lists')
    mounts = tuple(absolute(path) for path in mounts)
    if any(not isinstance(pci, str) or not re.fullmatch(PCI_PATTERN, pci) for pci in allowlist):
        raise ValueError('invalid PCI card address')
    HOST_MEMORY_LIMIT_BYTES = memory_limit
    FULL_TIMEOUT_SECONDS = full_timeout
    METAL, PYTHON, HEALTH, LEASES = paths
    WEIGHTS_BY_MODEL = weights
    MOUNTS, CARD_ALLOWLIST = mounts, tuple(allowlist)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def safe_file(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or not path.parts:
        raise ValueError('unsafe relative path')
    target = Path(root) / path
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('path escapes workspace')
    return target


def validate_log_request(job, stream, offset_bytes, limit_bytes):
    if not isinstance(job, str) or not re.fullmatch('[a-zA-Z0-9_-]{1,80}', job):
        raise ValueError('invalid job ID')
    if stream not in ('stdout', 'stderr'):
        raise ValueError('stream must be stdout or stderr')
    if type(offset_bytes) is not int or not 0 <= offset_bytes < 2**63:
        raise ValueError('offset_bytes must be a nonnegative signed 64-bit integer')
    if type(limit_bytes) is not int or not 1 <= limit_bytes <= 65536:
        raise ValueError('limit_bytes must be an integer in 1..65536')


def read_job_log(root, request):
    """Read only existing job stdout/stderr; offsets count bytes, not characters."""
    job, stream = request['job'], request.get('stream', 'stdout')
    offset, limit = request.get('offset_bytes', 0), request.get('limit_bytes', 65536)
    validate_log_request(job, stream, offset, limit)
    handles = []
    try:
        # Directory-relative, no-follow opens reject symlinks even during replacement.
        handles.append(os.open(root, os.O_RDONLY | os.O_DIRECTORY))
        for part in ('jobs', job):
            handles.append(os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=handles[-1]))
        handles.append(os.open(stream + '.log', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                               dir_fd=handles[-1]))
        info = os.fstat(handles[-1])
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('log must be a regular file')
        os.lseek(handles[-1], offset, os.SEEK_SET)
        data = os.read(handles[-1], min(limit, max(0, info.st_size - offset)))
        return {'job': job, 'stream': stream, 'offset_bytes': offset,
                'next_offset_bytes': offset + len(data), 'size_bytes': info.st_size,
                'eof': offset + len(data) >= info.st_size,
                'text': data.decode('utf-8', errors='replace')}
    finally:
        for handle in reversed(handles):
            os.close(handle)


def cards(device_dir=Path('/dev/tenstorrent'), sysfs_dir=Path('/sys/class/tenstorrent')):
    found = {}
    if not device_dir.is_dir():
        return found
    for node in device_dir.iterdir():
        if node.name.isdigit():
            pci = (sysfs_dir / ('tenstorrent!' + node.name) / 'device').resolve().name
            if re.fullmatch(PCI_PATTERN, pci) and (not CARD_ALLOWLIST or pci in CARD_ALLOWLIST):
                found[pci] = node.name
    return found


def health(pci):
    path = HEALTH / (pci.replace(':', '_') + '.json')
    return json.loads(path.read_text()) if path.exists() else {'state': 'unknown', 'pci': pci}


def mark(pci, state, job):
    write_json(HEALTH / (pci.replace(':', '_') + '.json'),
               {'pci': pci, 'state': state, 'job': job, 'checked_at': time.time()})


def checkpoint(model):
    spec = model_spec(model)
    if model not in WEIGHTS_BY_MODEL:
        raise ValueError('no checkpoint configured for model ' + model)
    path = WEIGHTS_BY_MODEL[model]
    if spec['weight_index'] and not path.is_dir():
        raise ValueError('sharded checkpoint requires a directory')
    return path, '/weights' if path.is_dir() else '/weights/' + path.name


def sandbox(root, job, pci, node, model):
    weights, target = checkpoint(model)
    args = ['bwrap', '--die-with-parent', '--new-session', '--unshare-all', '--clearenv']
    for path in ('/usr', '/bin', '/lib', '/lib64', '/sys', '/etc/ld.so.cache', '/etc/hosts'):
        args += ['--ro-bind', path, path]
    args += ['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
             '--ro-bind', str(METAL), str(METAL), '--ro-bind', str(METAL), '/opt/tt-metal',
             '--dir', '/weights', '--bind', str(job / 'work'), '/work',
             '--bind', str(job / 'output'), '/output', '--ro-bind', str(job / 'input'), '/input',
             '--ro-bind', str(root / 'harness/evaluate.py'), '/evaluate.py',
             '--ro-bind', str(root / 'harness/portforge_eval.py'), '/portforge_eval.py',
             '--ro-bind', str(root / 'harness/models.py'), '/models.py',
             '--ro-bind', str(root / 'harness/models.json'), '/models.json',
             '--dev-bind', '/dev/tenstorrent/' + node, '/dev/tenstorrent/' + node]
    if weights.is_dir():
        for filename in model_spec(model)['files']:
            if (weights / filename).is_file():
                args += ['--ro-bind', str((weights / filename).resolve()), '/weights/' + filename]
    else:
        args += ['--ro-bind', str(weights.resolve()), target]
    for path in MOUNTS:
        args += ['--ro-bind', str(path), str(path)]
    for alias in (str(METAL), '/opt/tt-metal'):
        args += ['--tmpfs', alias + '/generated']
        # A Git worktree uses a pointer file, not a .git directory. Hide both
        # representations without requiring a directory mount over a file.
        args += (['--ro-bind', '/dev/null', alias + '/.git'] if (METAL / '.git').is_file()
                 else ['--tmpfs', alias + '/.git'])
    for path in ('/dev/hugepages', '/dev/hugepages-1G'):
        if Path(path).exists():
            args += ['--bind' if path.startswith('/dev') else '--ro-bind', path, path]
    env = {'PATH': str(PYTHON.parent) + ':/usr/bin:/bin', 'HOME': '/work', 'PYTHONPATH': '/work:' + str(METAL),
           'TT_METAL_HOME': str(METAL), 'TT_METAL_CACHE': '/work/.cache', 'TT_VISIBLE_DEVICES': pci,
           'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '12', 'HF_HUB_OFFLINE': '1'}
    for key, value in env.items():
        args += ['--setenv', key, value]
    return args + ['--chdir', '/work']


def proc_snapshot(proc=Path('/proc')):
    """Read Linux process identity and RSS; disappearing processes are normal."""
    if not proc.is_dir():
        return None  # Unit-test hosts may be macOS; production requires Linux.
    page_bytes = os.sysconf('SC_PAGE_SIZE')
    found = {}
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            # comm can contain whitespace and parentheses; fields follow its last ')'.
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            found[int(path.name)] = dict(state=fields[0], ppid=int(fields[1]),
                pgrp=int(fields[2]), starttime=int(fields[19]), rss=max(0, int(fields[21])) * page_bytes)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def owned_processes(snapshot, leader, known):
    """Track a launched group and descendants by (pid, starttime), never PID alone."""
    owned = {pid: row for pid, row in snapshot.items() if known.get(pid) == row['starttime']}
    # Only extend group membership while its original leader is still identified.
    if leader in owned:
        owned.update({pid: row for pid, row in snapshot.items() if row['pgrp'] == leader})
    while True:
        children = {pid: row for pid, row in snapshot.items() if row['ppid'] in owned}
        if children.keys() <= owned.keys():
            break
        owned.update(children)
    known.update({pid: row['starttime'] for pid, row in owned.items()})
    return owned


def terminate_job(process, known):
    """Stop our launched group plus verified session-changing descendants."""
    def signal_owned(sig):
        # Popen's unreaped child cannot have its PID reused while it is running.
        if process.poll() is None:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        try:
            snapshot = proc_snapshot()
            owned = owned_processes(snapshot, process.pid, known) if snapshot is not None else {}
        except (OSError, ValueError, IndexError):
            owned = {}  # Still terminate the launched group if /proc becomes unreadable.
        for pid, row in owned.items():
            if row['state'] == 'Z':
                continue
            try:
                # Recheck identity immediately before signaling an individual PID.
                fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[19]) == row['starttime']:
                    os.kill(pid, sig)
            except (FileNotFoundError, ProcessLookupError):
                pass
    signal_owned(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # A leader can exit before its detached descendants; never leave them running.
    signal_owned(signal.SIGKILL)
    if process.poll() is None:
        process.wait(timeout=10)


def failure_kind(code, text, timeout=False, host_memory_limit=False):
    if host_memory_limit:
        return 'host_memory_limit'
    if timeout or code in (124, 137, -9):
        return 'timeout'
    # Normal TT startup prints firmware/open-device information. A later Python
    # exception must not turn successful initialization logs into a device fault.
    trace = 'Traceback (most recent call last):'
    failure = text.rsplit(trace, 1)[-1].lower() if trace in text else text.lower()
    device_fault = (('open_device' in failure and trace in text)
                    or '0xffffffff' in failure
                    or re.search(r'physical core[^\n]*(?:fail|error|timeout|timed out|not found)', failure)
                    or re.search(r'firmware[^\n]*(?:fail|error|mismatch)|(?:fail|error)[^\n]*firmware', failure))
    if code and device_fault:
        return 'device_initialization'
    return 'candidate_error' if code else None


def execute(args, job, total, monitor, stage='smoke'):
    start = changed = time.monotonic()
    sequence, progress, timed_out = None, None, False
    memory_exceeded, peak_rss, known = False, None, {}
    with (job / 'stdout.log').open('wb') as out, (job / 'stderr.log').open('wb') as err:
        process = subprocess.Popen(args, stdout=out, stderr=err, start_new_session=True)
        try:
            snapshot = proc_snapshot()
            if snapshot is None and sys.platform.startswith('linux'):
                raise OSError('Linux /proc is required for the host memory watchdog')
            if snapshot is not None and process.pid in snapshot:
                known[process.pid] = snapshot[process.pid]['starttime']
            while process.poll() is None:
                snapshot = proc_snapshot()
                if snapshot is not None:
                    owned = owned_processes(snapshot, process.pid, known)
                    rss = sum(row['rss'] for row in owned.values())
                    peak_rss = max(peak_rss or 0, rss)
                    if rss > HOST_MEMORY_LIMIT_BYTES:
                        memory_exceeded = True
                        terminate_job(process, known)
                        break
                if monitor:
                    try:
                        p = json.loads((job / 'output/progress.json').read_text())
                        if p['sequence'] != sequence:
                            sequence, progress, changed = p['sequence'], p, time.monotonic()
                            write_json(job / 'progress.json', p)
                    except (OSError, ValueError, KeyError):
                        pass
                phase = (progress or {}).get('phase', '')
                initial = progress is None or phase in ('open_device', 'load', 'create_backend',
                                                        'load_synchronize', 'import_backend', 'import_runtime', 'verify_checkpoint')
                limit = (600 if initial else (300 if stage == 'full' else 180)) if monitor else total
                if time.monotonic() - start > total or time.monotonic() - changed > limit:
                    timed_out = True
                    terminate_job(process, known)
                    break
                time.sleep(1)
        except BaseException:
            terminate_job(process, known)
            raise
    timed_out = not memory_exceeded and (timed_out or process.returncode in (124, 137, -9))
    return {'returncode': 125 if memory_exceeded else (124 if timed_out else process.returncode), 'timeout': timed_out,
            'host_memory_limit': memory_exceeded, 'peak_host_rss_bytes': peak_rss,
            'host_memory_limit_bytes': HOST_MEMORY_LIMIT_BYTES,
            'host_memory_accounting': 'sampled_sum_rss_shared_pages_counted_per_process',
            'progress': progress, 'seconds': time.monotonic() - start}


def main(root, request):
    root = Path(root).resolve()
    if request['kind'] == 'log':
        return read_job_log(root, request)
    configure(json.loads((root / 'harness/runtime.json').read_text()))
    if request['kind'] == 'health':
        return {'cards': [health(p) for p in cards()]}
    name = request['job']
    if not re.fullmatch('[a-zA-Z0-9_-]{1,80}', name):
        raise ValueError('invalid job ID')
    job = root / 'jobs' / name
    if request['kind'] == 'status':
        path = job / 'progress.json'
        return json.loads(path.read_text()) if path.exists() else {'phase': 'queued_or_starting'}
    model = request.get('model')
    if not isinstance(model, str) or not model:
        raise ValueError('request must name a registry model')
    _, weight_target = checkpoint(model)
    for folder in ('work', 'input', 'output'):
        (job / folder).mkdir(parents=True, exist_ok=True)
    for name, content in request.get('files', {}).items():
        target = safe_file(job / 'work', name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    public_stage = request.get('public_input_stage', '')
    if public_stage and (request['kind'] != 'run' or public_stage not in ('smoke', 'bringup')):
        raise ValueError('public_input_stage requires run and smoke/bringup')
    input_stage = request.get('stage') if request['kind'] == 'evaluate' else public_stage
    if input_stage:
        if input_stage not in ('smoke', 'bringup', 'full'):
            raise ValueError('invalid stage')
        reference = root / 'references' / model / input_stage / 'input'
        if (reference / 'suite.json').exists():
            prepared = json.loads((reference / 'suite.json').read_text())
            if prepared.get('model_key') != model:
                raise ValueError('reference suite model differs from requested checkpoint')
        for path in reference.iterdir():
            if path.name not in ('suite.json', 'config.json', 'inputs.npz', 'checkpoint_receipt.json'):
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError('public input must be a regular file')
            (job / 'input' / path.name).write_bytes(path.read_bytes())
    if request['kind'] == 'evaluate':
        stage = request['stage']
        if stage not in ('smoke', 'bringup', 'full'):
            raise ValueError('invalid stage')
        precision = request.get('precision', 'bf16')
        if precision not in ('bf16', 'fp32', 'fp16', 'bfp8_b'):
            raise ValueError('invalid candidate precision')
        command = [str(PYTHON), '/evaluate.py', 'candidate', '--input', '/input', '--output', '/output',
                   '--weights', weight_target, '--precision', precision]
        total = {'smoke': 900, 'bringup': 1200, 'full': FULL_TIMEOUT_SECONDS}[stage]
    elif request['kind'] == 'preflight':
        command = [str(PYTHON), '-c', "import os,torch,ttnn; "
            f"assert not os.path.exists({str(Path.home() / '.ssh')!r}); "
            f"assert not os.path.exists({str(root / 'references')!r}); "
            "d=ttnn.open_device(device_id=0); "
            "x=ttnn.from_torch(torch.ones(1,1,32,32),device=d,layout=ttnn.TILE_LAYOUT); "
            "assert torch.equal(ttnn.to_torch(ttnn.add(x,x)),torch.full((1,1,32,32),2.)); "
            "ttnn.close_device(d); print('ISOLATION_DEVICE_OK')"]
        total = 120
    else:
        command = ['/bin/bash', '--noprofile', '--norc', '-c', request['command']]
        total = min(max(int(request.get('timeout', 120)), 1), 300)
    started = time.monotonic()
    lease = None
    LEASES.mkdir(parents=True, exist_ok=True)
    while not lease:
        available = [(p, n) for p, n in cards().items()
                     if (request['kind'] == 'preflight' and p == request.get('pci'))
                     or (request['kind'] != 'preflight' and health(p)['state'] == 'healthy')]
        if not available or time.monotonic() - started > 300:
            return {'returncode': 75, 'failure': 'no_healthy_or_free_card'}
        for pci, node in available:
            handle = (LEASES / ('device-' + pci.replace(':', '_') + '.lock')).open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if request['kind'] != 'preflight' and health(pci)['state'] != 'healthy':
                    handle.close()
                    continue
                lease = handle
                break
            except BlockingIOError:
                handle.close()
        if not lease:
            time.sleep(1)
    try:
        write_json(job / 'identity.json', {'pci': pci, 'model': model, 'started': time.time()})
        result = execute(['timeout', '--kill-after=10s', str(total)] + sandbox(root, job, pci, node, model) + command,
                         job, total + 2, request['kind'] == 'evaluate', request.get('stage', 'smoke'))
        result.update(pci=pci, model=model, job=job.name,
                      stdout=(job / 'stdout.log').read_text(errors='replace')[-12000:],
                      stderr=(job / 'stderr.log').read_text(errors='replace')[-12000:])
        result['failure'] = failure_kind(result['returncode'], result['stdout'] + result['stderr'], result['timeout'],
                                         result.get('host_memory_limit', False))
        hashing_timeout = result['failure'] == 'timeout' and (result.get('progress') or {}).get('phase') == 'verify_checkpoint'
        if not hashing_timeout and (result['failure'] in ('timeout', 'device_initialization', 'host_memory_limit')
                                    or (request['kind'] == 'preflight' and result['returncode'])):
            mark(pci, 'quarantined', job.name)
        elif request['kind'] == 'preflight' and result['returncode'] == 0:
            mark(pci, 'healthy', job.name)
        write_json(job / 'result.json', result)
        return result
    finally:
        lease.close()


if __name__ == '__main__':
    print(json.dumps(main(sys.argv[1], json.load(sys.stdin))))
