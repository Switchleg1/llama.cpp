#!/usr/bin/env python3
"""Pick a llama-server configuration for a MoE model with CPU/GPU expert placement, then print, save or run it.

The rules come from measurements on this fork (see docs/moe-hybrid-strata-port-plan.md):
  - pinned expert memory, when the experts fit in available RAM: cache fills become DMA instead of staged copies
  - --moe-expert-cache-size auto: measures the configuration, draft included, to size the cache
  - an MTP sidecar, when present, runs entirely on the GPU with 4 drafts
  - hybrid decide mode only when the cache cannot hold a verification pass's routes; large caches fill faster

  python scripts/moe-launch.py --model MODEL-00001-of-0000N.gguf --mtp SIDECAR.gguf --tune --save cfg.json
  python scripts/moe-launch.py --config cfg.json --run

Tuning tests policies and draft counts, then smaller caches below the loaded fit.
The saved winner is specific to the benchmark prompt and current hardware.
"""
import argparse, copy, ctypes, glob, itertools, json, os, re, shlex, signal, socket, statistics, subprocess, sys, time, urllib.request
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
    if not facts['expert_bytes'] or not facts['n_expert'] or not facts['n_expert_used']:
        raise ValueError('model has no recognized routed expert weights or metadata')
    vram_total, vram_free, gpu_name = gpu_memory()
    ram_total, ram_avail = host_memory()
    cores = os.cpu_count() or 8
    mtp = Path(args.mtp) if args.mtp else (None if args.no_mtp else find_mtp(args.model))
    mtp_bytes = mtp.stat().st_size if mtp else 0
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
    hybrid_supported = drafts + 1 <= 8 and routes_per_pass <= 64
    if hybrid and not hybrid_supported:
        raise ValueError('experimental hybrid supports at most 8 rows and 64 routes per pass')

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
    # Measured: GPU-side sampling and decode/draft overlap shorten the host gap between rounds (same greedy output).
    if not args.no_overlap:
        cmd += ['--backend-sampling', '--decode-overlap', '--decode-boundary-overlap']
    env = {}
    if hybrid:
        cmd += ['--moe-expert-split', 'on', '--moe-expert-split-threads', str(max(4, cores - 1)),
                '--moe-expert-promote', '64:2']
    report = {
        'gpu': gpu_name, 'vram_free_gib': round(vram_free / GiB, 1), 'ram_available_gib': round(ram_avail / GiB, 1),
        'cores': cores, 'arch': facts['arch'], 'layers': facts['n_layer'], 'experts': facts['n_expert'],
        'experts_used': facts['n_expert_used'], 'expert_gib': round(facts['expert_bytes'] / GiB, 2),
        'dense_gib': round(dense_bytes / GiB, 2), 'host_lookup_gib': round(facts['host_bytes'] / GiB, 2), 'per_slot_mib': round(per_slot / 2**20, 1),
        'estimated_slots': int(est_slots), 'routes_per_pass': routes_per_pass, 'mtp': str(mtp) if mtp else None,
        'drafts': drafts,
        'pinned_mib': pinned_mib, 'hybrid_supported': hybrid_supported,
        'policy': 'hybrid decide (estimated cache too small for verification passes)' if hybrid
                      else 'LRU cache (heuristic; use --tune to measure)',
    }
    if not pinned_mib:
        report['warning'] = 'experts do not fit in available RAM; cache fills stay pageable and slow'
    return {'command': cmd, 'env': env, 'report': report}


def launch_env(config):
    # Diagnostic gates inherited from a shell must not change the measured configuration.
    prefixes = ('LLAMA_MOE_HYBRID_', 'LLAMA_MOE_ADMIT_')
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes) and
           k not in ('LLAMA_MOE_ROUTE_DUMP', 'LLAMA_MOE_AUTO_PIN', 'GGML_CUDA_SYNC_DEBUG')}
    env.update(config['env'])
    env['PATH'] = str(ROOT / 'toolchain/cuda/bin/x64') + os.pathsep + env.get('PATH', '')
    return env


def confirmation_prompts(prompt):
    records = '\n'.join(f'Record {i}: temperature {18 + i % 13} C, pressure {1000 + i % 19} hPa, station {i % 7}.'
                        for i in range(96))
    return [prompt, 'Write a Python function that merges two sorted lists without modifying either input. Explain its complexity.',
            'Summarize the trends and temperature range in these observations:\n' + records]


def bench(config, args, label='bench', prompts=None):
    host = '127.0.0.1' if args.host == '0.0.0.0' else args.host
    # Refuse an occupied port before starting, so health checks cannot target another server.
    with socket.socket() as probe:
        probe.bind((host, args.port))
    url = f'http://{host}:{args.port}'
    log_path = Path(f'{args.log}-{label}.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def post(endpoint, body):
        req = urllib.request.Request(url + endpoint, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=3600) as response:
            return json.loads(response.read())

    with log_path.open('wb') as log:
        # Own process group on Windows, so Ctrl+Break can stop the server gracefully (end-of-run logs, profiles).
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        proc = subprocess.Popen(config['command'], env=launch_env(config), stdout=log, stderr=subprocess.STDOUT,
                                creationflags=flags)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited with {proc.returncode}; see {log_path}')
                try:
                    with urllib.request.urlopen(url + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(1)
            else:
                raise TimeoutError(f'server startup timed out; see {log_path}')
            workloads = []
            context = int(config['command'][config['command'].index('-c') + 1])
            for text in prompts if prompts is not None else [args.prompt]:
                prompt = post('/apply-template', {'messages': [{'role': 'user', 'content': text}]})['prompt']
                prompt_tokens = len(post('/tokenize', {'content': prompt, 'add_special': True})['tokens'])
                if prompt_tokens + 128 > context:
                    raise ValueError(f'benchmark prompt plus output exceeds context {context}; see {log_path}')
                timings = []
                for index in range(args.repeats + 1):
                    result = post('/completion', {'prompt': prompt, 'n_predict': 128, 'temperature': 0,
                                                  'seed': 1234, 'cache_prompt': False, 'ignore_eos': True})
                    timing = result['timings']
                    if timing['predicted_n'] != 128 or timing['predicted_per_second'] <= 0:
                        raise RuntimeError(f'incomplete 128-token benchmark; see {log_path}')
                    if index:
                        timings.append(timing)
                workloads.append({'prompt': text, 'prompt_tokens': prompt_tokens, 'timings': timings,
                                  'median_tok_s': statistics.median(t['predicted_per_second'] for t in timings)})
            slots = config['command'][config['command'].index('--moe-expert-cache-size') + 1]
            if slots == 'auto':
                match = re.search(r'MoE expert cache auto: (\d+) slots', log_path.read_text(errors='replace'))
                actual_slots = int(match.group(1)) if match else None
            else:
                actual_slots = int(slots)
            return {'median_tok_s': statistics.harmonic_mean(w['median_tok_s'] for w in workloads),
                    'score': 'harmonic mean of per-prompt median decode tok/s', 'workloads': workloads,
                    'timings': workloads[0]['timings'], 'log': str(log_path), 'cache_slots': actual_slots}
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGINT)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def tuning_candidates(config, args):
    draft_counts = args.tune_drafts if config['report']['mtp'] else [0]
    for drafts in draft_counts:
        base = copy.copy(args)
        base.hybrid = 'off'
        base.mtp = config['report']['mtp'] if drafts else None
        base.no_mtp = not drafts
        base.drafts = drafts
        lru = plan(base)
        lru['report']['policy'] = 'LRU cache'
        yield f'd{drafts}-lru', lru
        if lru['report']['hybrid_supported']:
            base.hybrid = 'on'
            hybrid = plan(base)
            for budget in (16, 64):
                candidate = copy.deepcopy(hybrid)
                cmd = candidate['command']
                cmd[cmd.index('--moe-expert-promote') + 1] = f'{budget}:2'
                candidate['report'] = copy.deepcopy(lru['report'])
                candidate['report']['policy'] = f'expert split, promote {budget}:2'
                yield f'd{drafts}-split-{budget}', candidate


def sweep_regressed(score, state, margin):
    if score is None:
        state['regressions'] = 0
        return False
    state['regressions'] = state['regressions'] + 1 if score < state['best'] * (1 - margin) else 0
    state['best'] = max(state['best'], score)
    return state['regressions'] >= 2


def tune(config, args):
    results = []
    stopped_sweeps = []
    winner = None
    best = 0

    def measure(label, candidate):
        nonlocal winner, best
        print('measuring', label, flush=True)
        try:
            measurement = bench(candidate, args, label)
            candidate['report']['measured_cache_slots'] = measurement['cache_slots']
            results.append(dict(measurement, candidate=label, configuration=copy.deepcopy(candidate)))
            speed = measurement['median_tok_s']
            print(f'  median {speed:.2f} tok/s', flush=True)
            if speed > best:
                winner, best = candidate, speed
            return speed
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            results.append({'candidate': label, 'error': str(error), 'configuration': copy.deepcopy(candidate)})
            print('  failed:', error, flush=True)

    draft_state = {'best': 0, 'regressions': 0}
    groups = itertools.groupby(tuning_candidates(config, args), key=lambda item: item[1]['report']['drafts'])
    for drafts, candidates in groups:
        scores = [score for label, candidate in candidates if (score := measure(label, candidate)) is not None]
        if sweep_regressed(max(scores) if scores else None, draft_state, args.regression_margin):
            stopped_sweeps.append({'parameter': 'drafts', 'after': drafts, 'best_tok_s': draft_state['best']})
            print('draft sweep stopped after two regressions; moving to cache sweep', flush=True)
            break
    if winner is None:
        raise RuntimeError('all tuning candidates failed; inspect the server logs')

    initial_winner = copy.deepcopy(winner)
    initial_result = max((r for r in results if 'median_tok_s' in r), key=lambda r: r['median_tok_s'])
    cap = initial_result['cache_slots']
    cache_sizes = []
    if args.tune_cache and cap is not None:
        # Stay below the successfully loaded fit. Do not grow from the heuristic slot estimate.
        cache_sizes = sorted({max(1, int(cap * fraction) // 8 * 8) for fraction in (0.9, 0.8)}, reverse=True)
        cache_sizes = [slots for slots in cache_sizes if slots < cap]
        cache_state = {'best': best, 'regressions': 0}
        tested_sizes = []
        for slots in cache_sizes:
            tested_sizes.append(slots)
            base = copy.copy(args)
            base.slots = slots
            base.tune_drafts = [initial_winner['report']['drafts']]
            scores = []
            for label, candidate in tuning_candidates(initial_winner, base):
                score = measure(f's{slots}-{label}', candidate)
                if score is not None:
                    scores.append(score)
            if sweep_regressed(max(scores) if scores else None, cache_state, args.regression_margin):
                stopped_sweeps.append({'parameter': 'cache_slots', 'after': slots, 'best_tok_s': cache_state['best']})
                print('cache sweep stopped after two regressions; moving to confirmation', flush=True)
                break
        cache_sizes = tested_sizes
    elif args.tune_cache:
        print('cache sweep skipped: server did not report a measured auto slot count', flush=True)

    confirmation = []
    if args.confirm_finalists:
        finalists = sorted((r for r in results if 'median_tok_s' in r),
                           key=lambda r: r['median_tok_s'], reverse=True)[:args.confirm_finalists]
        confirmed_winner = None
        confirmed_best = 0
        for result in finalists:
            label = 'confirm-' + result['candidate']
            candidate = copy.deepcopy(result['configuration'])
            print('confirming', result['candidate'], flush=True)
            try:
                measurement = bench(candidate, args, label, confirmation_prompts(args.prompt))
                candidate['report']['measured_cache_slots'] = measurement['cache_slots']
                confirmation.append(dict(measurement, candidate=result['candidate'], configuration=copy.deepcopy(candidate)))
                speed = measurement['median_tok_s']
                print(f'  aggregate {speed:.2f} tok/s', flush=True)
                if speed > confirmed_best:
                    confirmed_winner, confirmed_best = candidate, speed
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                confirmation.append({'candidate': result['candidate'], 'error': str(error), 'configuration': candidate})
                print('  failed:', error, flush=True)
        if confirmed_winner is None:
            raise RuntimeError('all finalists failed confirmation; no confirmed configuration can be saved')
        winner = confirmed_winner

    winner['tuning'] = {'results': results, 'prompt': args.prompt, 'repeats': args.repeats,
                        'confirmation': confirmation, 'confirmed': bool(confirmation),
                        'regression_margin': args.regression_margin, 'stopped_sweeps': stopped_sweeps,
                        'draft_counts': args.tune_drafts if config['report']['mtp'] else [0],
                        'n_predict': 128, 'ignore_eos': True,
                        'cache_sweep': {'enabled': args.tune_cache, 'loaded_slot_cap': cap, 'tested_sizes': cache_sizes,
                                        'drafts': initial_winner['report']['drafts']},
                        'scope': 'policy/draft grid, smaller caches, then multi-prompt confirmation of finalists; decode speed only, not a quality test'}
    return winner


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--model', help='first shard of the model')
    source.add_argument('--config', help='replay a saved local configuration without retuning')
    ap.add_argument('--mtp', help='MTP sidecar (default: an mtp-*.gguf next to the model)')
    ap.add_argument('--no-mtp', action='store_true')
    ap.add_argument('--drafts', type=int, default=4)
    ap.add_argument('--no-overlap', action='store_true',
                    help='omit the experimental --backend-sampling/--decode-overlap/--decode-boundary-overlap flags')
    ap.add_argument('--hybrid', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--slots', type=int, help='fixed expert cache slots instead of auto (simulates a smaller GPU)')
    ap.add_argument('--ctx', type=int, default=8192)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--server', default=str(ROOT / 'build/hybrid-baseline/bin' /
                                           ('llama-server.exe' if os.name == 'nt' else 'llama-server')))
    ap.add_argument('--save', help='write the configuration as JSON')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='start the server with this configuration')
    mode.add_argument('--bench', action='store_true', help='warm up, measure repeated replies, stop')
    ap.add_argument('--tune', action='store_true', help='measure policies and draft counts; can combine with --run')
    ap.add_argument('--tune-cache', action=argparse.BooleanOptionalAction, default=True,
                    help='after --tune, test smaller caches below the successfully loaded fit (default: enabled)')
    ap.add_argument('--tune-drafts', type=int, nargs='+', default=[0, 2, 4],
                    help='draft counts tested by --tune (0 disables the draft model; default: 0 2 4)')
    ap.add_argument('--regression-margin', type=float, default=0.05,
                    help='stop a parameter sweep after two regressions below its best speed by this fraction (default: 0.05)')
    ap.add_argument('--confirm-finalists', type=int, default=2,
                    help='top candidates checked on three prompts, including longer input (0 disables; max 3)')
    ap.add_argument('--repeats', type=int, default=3, help='measured replies after one warmup per prompt')
    ap.add_argument('--startup-timeout', type=int, default=900)
    ap.add_argument('--log', default=str(ROOT / 'build/moe-launch'), help='server log path prefix')
    ap.add_argument('--prompt', default='Write a short paragraph about the history of the bicycle.')
    args = ap.parse_args()
    if args.drafts < 1 or args.ctx < 1 or args.repeats < 1 or args.startup_timeout < 1:
        ap.error('drafts, context, repeats and startup timeout must be positive')
    if not 0 < args.regression_margin < 1:
        ap.error('regression-margin must be between 0 and 1')
    if not 0 <= args.confirm_finalists <= 3:
        ap.error('confirm-finalists must be between 0 and 3')
    if args.slots is not None and args.slots < 1:
        ap.error('slots must be positive')
    if args.mtp and args.no_mtp:
        ap.error('--mtp and --no-mtp are mutually exclusive')
    if any(d < 0 or d > 7 for d in args.tune_drafts):
        ap.error('tuning draft counts must be between 0 and 7')
    args.tune_drafts = list(dict.fromkeys(args.tune_drafts))
    if args.tune and (args.hybrid != 'auto' or args.bench or args.config):
        ap.error('--tune requires a model, --hybrid auto, and no --bench')

    if args.config:
        config = json.loads(Path(args.config).read_text())
        command = config['command']
        if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
            ap.error('saved command must be a nonempty list of strings')
        if not isinstance(config['env'], dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                         for k, v in config['env'].items()):
            ap.error('saved environment must contain string keys and values')
        args.host = command[command.index('--host') + 1]
        args.port = int(command[command.index('--port') + 1])
    else:
        config = plan(args)
    if args.tune:
        config = tune(config, args)
    print(json.dumps(config['report'], indent=1))
    print('\nenvironment:', ' '.join(f'{k}={v}' for k, v in config['env'].items()) or '(none)')
    print('command:', subprocess.list2cmdline(config['command']) if os.name == 'nt' else shlex.join(config['command']))
    if args.save:
        Path(args.save).write_text(json.dumps(config, indent=1))
        print('saved', args.save)
    if args.bench:
        measurement = bench(config, args)
        print('median decode tok/s:', round(measurement['median_tok_s'], 2))
        print('server log:', measurement['log'])
    elif args.run:
        env = launch_env(config)
        os.execvpe(config['command'][0], config['command'], env) if os.name != 'nt' else \
            sys.exit(subprocess.call(config['command'], env=env))


if __name__ == '__main__':
    main()
