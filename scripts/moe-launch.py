#!/usr/bin/env python3
"""Pick a llama-server configuration for a MoE model with CPU/GPU expert placement, then print, save or run it.

The rules come from measurements on this fork (see docs/moe-hybrid-strata-port-plan.md):
  - pinned expert memory, when the experts fit in available RAM: cache fills become DMA instead of staged copies
  - --moe-expert-cache-size auto: measures the configuration, draft included, so the cache never oversubscribes
  - an MTP sidecar, when present, runs entirely on the GPU with 4 drafts
  - hybrid decide mode only when the cache cannot hold a verification pass's routes; large caches fill faster

  python scripts/moe-launch.py --model MODEL-00001-of-0000N.gguf [--mtp SIDECAR.gguf] [--save cfg.json] [--run|--bench]
"""
import argparse, ctypes, glob, json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gguf-py'))


def shards(model):
    m = re.match(r'(.*)-(\d{5})-of-(\d{5})\.gguf$', str(model))
    if not m:
        return [Path(model)]
    return [Path(f'{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf') for i in range(1, int(m.group(3)) + 1)]


def model_facts(model):
    from gguf import GGUFReader
    facts = {'expert_bytes': 0, 'host_bytes': 0, 'total_bytes': 0, 'n_layer': 0, 'n_expert': 0, 'n_expert_used': 0, 'arch': None}
    for path in shards(model):
        reader = GGUFReader(str(path))
        for field in reader.fields.values():
            name = field.name
            if name == 'general.architecture':
                facts['arch'] = bytes(field.parts[-1]).decode()
            for key, suffix in (('n_layer', '.block_count'), ('n_expert', '.expert_count'),
                                ('n_expert_used', '.expert_used_count')):
                if name.endswith(suffix) and not facts[key]:
                    facts[key] = int(field.parts[-1][0])
        for tensor in reader.tensors:
            facts['total_bytes'] += int(tensor.n_bytes)
            if re.search(r'\.ffn_(up|down|gate|gate_up)_(ch|)exps\.weight$', tensor.name):
                facts['expert_bytes'] += int(tensor.n_bytes)
            elif tensor.name.endswith('token_embd.weight'):
                facts['host_bytes'] += int(tensor.n_bytes)   # embedding lookups stay in host memory
    return facts


def gpu_memory():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free,name', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
        total, free, name = [x.strip() for x in out.split(',', 2)]
        return int(total) << 20, int(free) << 20, name
    except Exception:
        return 0, 0, 'unknown'


def host_memory():
    if os.name == 'nt':
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong), ('total', ctypes.c_ulonglong),
                        ('avail', ctypes.c_ulonglong), ('total_page', ctypes.c_ulonglong), ('avail_page', ctypes.c_ulonglong),
                        ('total_virtual', ctypes.c_ulonglong), ('avail_virtual', ctypes.c_ulonglong),
                        ('avail_extended', ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.total, status.avail
    info = dict(line.split(':', 1) for line in open('/proc/meminfo'))
    return int(info['MemTotal'].split()[0]) << 10, int(info['MemAvailable'].split()[0]) << 10


def find_mtp(model):
    folder = Path(model).parent
    found = sorted(glob.glob(str(folder / 'mtp-*exp-q4_K*.gguf'))) or sorted(glob.glob(str(folder / 'mtp-*.gguf')))
    return Path(found[0]) if found else None


def plan(args):
    facts = model_facts(args.model)
    vram_total, vram_free, gpu_name = gpu_memory()
    ram_total, ram_avail = host_memory()
    cores = os.cpu_count() or 8
    mtp = Path(args.mtp) if args.mtp else (None if args.no_mtp else find_mtp(args.model))
    mtp_bytes = mtp.stat().st_size if mtp and mtp.exists() else 0
    drafts = args.drafts if mtp else 0
    GiB = 1 << 30

    dense_bytes = facts['total_bytes'] - facts['expert_bytes'] - facts['host_bytes']
    per_slot = facts['expert_bytes'] // max(1, facts['n_expert'])      # one expert in every layer
    # Rough slot estimate, only to choose the policy; the server measures the real count with 'auto'.
    reserve = dense_bytes + mtp_bytes + 2 * GiB   # compute buffers and KV; matches 'auto' on Qwen IQ3_S
    est_slots = max(0, (vram_free - reserve) // max(1, per_slot))
    if args.slots:
        est_slots = args.slots
    routes_per_pass = facts['n_expert_used'] * (drafts + 1)
    hybrid = args.hybrid == 'on' or (args.hybrid == 'auto' and est_slots < routes_per_pass)

    pinned_mib = 0
    if ram_avail > facts['expert_bytes'] + facts['host_bytes'] + 6 * GiB:
        pinned_mib = (facts['expert_bytes'] + facts['expert_bytes'] // 50 + GiB) >> 20

    threads = str(min(8, cores))
    cmd = [str(Path(args.server)), '-m', str(args.model), '-ngl', 'all', '--cpu-moe',
           '--moe-expert-cache-size', str(args.slots) if args.slots else 'auto', '-c', str(args.ctx), '-np', '1', '-t', threads, '-tb', threads,
           '--load-mode', 'mmap', '--host', args.host, '--port', str(args.port)]
    if pinned_mib:
        cmd += ['--moe-expert-cache-host-pinned-mb', str(pinned_mib)]
    if mtp:
        cmd += ['-md', str(mtp), '-ngld', 'all', '--spec-type', 'draft-mtp', '--spec-draft-n-max', str(drafts)]
    env = {}
    if hybrid:
        env.update({'LLAMA_MOE_ADMIT_SHADOW': '1', 'LLAMA_MOE_ADMIT_DOORBELL': '1', 'LLAMA_MOE_HYBRID_CPU': '1',
                    'LLAMA_MOE_HYBRID_Q8_1': '1', 'LLAMA_MOE_HYBRID_DECIDE': '1',
                    'LLAMA_MOE_HYBRID_PROMOTE': '4', 'LLAMA_MOE_HYBRID_PROMOTE_MIN': '1',
                    'LLAMA_MOE_HYBRID_CPU_THREADS': str(max(4, min(16, cores - int(threads))))})
    report = {
        'gpu': gpu_name, 'vram_free_gib': round(vram_free / GiB, 1), 'ram_available_gib': round(ram_avail / GiB, 1),
        'cores': cores, 'arch': facts['arch'], 'layers': facts['n_layer'], 'experts': facts['n_expert'],
        'experts_used': facts['n_expert_used'], 'expert_gib': round(facts['expert_bytes'] / GiB, 2),
        'dense_gib': round(dense_bytes / GiB, 2), 'host_lookup_gib': round(facts['host_bytes'] / GiB, 2), 'per_slot_mib': round(per_slot / 2**20, 1),
        'estimated_slots': int(est_slots), 'routes_per_pass': routes_per_pass, 'mtp': str(mtp) if mtp else None,
        'pinned_mib': pinned_mib, 'policy': 'hybrid decide (cache too small for verification passes)' if hybrid
                      else 'pinned LRU cache (fills over PCIe beat the CPU here)',
    }
    if not pinned_mib:
        report['warning'] = 'experts do not fit in available RAM; cache fills stay pageable and slow'
    return {'command': cmd, 'env': env, 'report': report}


def bench(config, port):
    env = dict(os.environ, **config['env'])
    env['PATH'] = str(ROOT / 'toolchain/cuda/bin/x64') + os.pathsep + env.get('PATH', '')
    proc = subprocess.Popen(config['command'], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        for _ in range(900):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1)
        lt, gt, bar = chr(60), chr(62), chr(124)
        im, en = lt + bar + 'im_start' + bar + gt, lt + bar + 'im_end' + bar + gt
        prompt = im + 'user\nWrite a short paragraph about the history of the bicycle.' + en + '\n' + im + 'assistant\n'
        body = json.dumps({'prompt': prompt, 'n_predict': 128, 'temperature': 0, 'cache_prompt': False}).encode()
        speeds = []
        for _ in range(2):
            req = urllib.request.Request(f'http://127.0.0.1:{port}/completion', data=body,
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=3600) as r:
                speeds.append(json.loads(r.read())['timings']['predicted_per_second'])
        return speeds[-1]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except Exception:
            proc.kill()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='first shard of the model')
    ap.add_argument('--mtp', help='MTP sidecar (default: an mtp-*.gguf next to the model)')
    ap.add_argument('--no-mtp', action='store_true')
    ap.add_argument('--drafts', type=int, default=4)
    ap.add_argument('--hybrid', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--slots', type=int, help='fixed expert cache slots instead of auto (simulates a smaller GPU)')
    ap.add_argument('--ctx', type=int, default=8192)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--server', default=str(ROOT / 'build/hybrid-baseline/bin' /
                                           ('llama-server.exe' if os.name == 'nt' else 'llama-server')))
    ap.add_argument('--save', help='write the configuration as JSON')
    ap.add_argument('--run', action='store_true', help='start the server with this configuration')
    ap.add_argument('--bench', action='store_true', help='start it, time a 128 token reply, stop')
    args = ap.parse_args()

    config = plan(args)
    print(json.dumps(config['report'], indent=1))
    print('\nenvironment:', ' '.join(f'{k}={v}' for k, v in config['env'].items()) or '(none)')
    print('command:', ' '.join(config['command']))
    if args.save:
        Path(args.save).write_text(json.dumps(config, indent=1))
        print('saved', args.save)
    if args.bench:
        print('decode tok/s:', round(bench(config, args.port), 2))
    elif args.run:
        env = dict(os.environ, **config['env'])
        os.execvpe(config['command'][0], config['command'], env) if os.name != 'nt' else \
            sys.exit(subprocess.call(config['command'], env=env))


if __name__ == '__main__':
    main()
