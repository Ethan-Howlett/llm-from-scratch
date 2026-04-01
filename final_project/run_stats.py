import os
import subprocess
import json
import csv
import itertools
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

# Upper bound so we do not spawn dozens of CPU-heavy worker processes by mistake.
MAX_CONCURRENT_CAP = 16
# Reserve this much MiB on the target GPU for driver / display / fragmentation.
VRAM_HEADROOM_MIB = 1024
# Rough peak VRAM (MiB) per training process at batch_size≈8; scaled by actual max batch in PARAM_GRID.
MODEL_BASE_MIB = {
    'gpt2-small (124M)': 5500,
    'gpt2-medium (355M)': 12000,
    'gpt2-large (774M)': 22000,
    'gpt2-xl (1558M)': 38000,
}


def _nvidia_smi_gpu_free_mib() -> list[tuple[int, int]]:
    """Return [(gpu_index, memory.free_MiB), ...] from nvidia-smi. Empty if unavailable."""
    try:
        proc = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.free',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0:
        return []
    rows: list[tuple[int, int]] = []
    for line in (proc.stdout or '').strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            try:
                rows.append((int(parts[0]), int(parts[1])))
            except ValueError:
                continue
    return rows


def _target_gpu_free_mib(gpus: list[tuple[int, int]]) -> int | None:
    """Free MiB on the GPU PyTorch will use as cuda:0 (CUDA_VISIBLE_DEVICES) or physical GPU 0."""
    if not gpus:
        return None
    spec = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if spec:
        first = spec.split(',')[0].strip()
        if first.isdigit():
            want = int(first)
            for idx, free in gpus:
                if idx == want:
                    return free
    return gpus[0][1]


def _estimate_mib_per_run(param_grid: dict) -> int:
    models = param_grid.get('model_size', ['gpt2-medium (355M)'])
    batches = param_grid.get('batch_size', [8])
    max_batch = max(int(b) for b in batches)
    worst_model = max(models, key=lambda m: MODEL_BASE_MIB.get(m, 14_000))
    base = MODEL_BASE_MIB.get(worst_model, 14_000)
    # Activations scale roughly with batch; anchor at batch 8.
    scale = max(0.75, min(2.25, max_batch / 8.0))
    return max(2048, int(base * scale))


def max_concurrent_workers(param_grid: dict) -> int:
    """
    Worker count from env, or from nvidia-smi free VRAM and PARAM_GRID worst-case estimate.

    Override: RUN_STATS_MAX_WORKERS=<int>
    """
    override = os.environ.get('RUN_STATS_MAX_WORKERS', '').strip()
    if override:
        try:
            n = int(override)
            return max(1, min(n, MAX_CONCURRENT_CAP))
        except ValueError:
            print(f"[run_stats] Ignoring invalid RUN_STATS_MAX_WORKERS={override!r}", flush=True)

    per_run = _estimate_mib_per_run(param_grid)
    gpus = _nvidia_smi_gpu_free_mib()
    free = _target_gpu_free_mib(gpus)
    if free is None:
        print(
            "[run_stats] nvidia-smi unavailable or no GPU; using max_workers=1 "
            "(set RUN_STATS_MAX_WORKERS to override)",
            flush=True,
        )
        return 1

    budget = max(0, free - VRAM_HEADROOM_MIB)
    n = max(1, budget // per_run)
    n = min(n, MAX_CONCURRENT_CAP)
    print(
        f"[run_stats] VRAM: gpu free≈{free} MiB, headroom={VRAM_HEADROOM_MIB} MiB, "
        f"est. per run≈{per_run} MiB → max_workers={n}",
        flush=True,
    )
    return n

# PARAM_GRID = {
#     'model_size': ['gpt2-small (124M)', 'gpt2-medium (355M)', 'gpt2-large (774M)'],
#     'batch_size': [3, 5, 8, 13],
#     'epochs': [3, 5, 8, 13],
#     'lr': [1e-5, 5e-5, 1e-4],
#     'drop_rate': [0.0, 0.05, 0.1],
#     'weight_decay': [0.05, 0.1, 0.2],
#     'balance_dataset': [0, 1]
# }

PARAM_GRID = {
    'model_size': ['gpt2-small (124M)', 'gpt2-medium (355M)'],
    'batch_size': [3, 5, 8],
    'epochs': [3, 5, 8],
    'lr': [1e-5, 5e-5, 1e-4],
    'drop_rate': [0.0, 0.05, 0.1],
    'weight_decay': [0.05, 0.1, 0.2],
    'balance_dataset': [0, 1]
}


def permutations(grid: dict):
    keys = list(grid.keys())
    for vals in itertools.product(*grid.values()):
        yield dict(zip(keys, vals))


def run_test(params: dict) -> dict:
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    cmd = ['python3', 'finetune_testing.py']
    for key, val in params.items():
        cmd += [f'--{key}', str(val)]

    print(f"[run_stats] Starting subprocess: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Parse results for the JSON
    try:
        for line in result.stdout.splitlines():
            if line.startswith('RESULTS_JSON:'):
                return json.loads(line[len('RESULTS_JSON:'):])
    except json.JSONDecodeError:
        print(f"[run_stats] JSON parse error (exit {result.returncode}). stdout tail:\n{result.stdout[-1500:]}", flush=True)
        return None

    if result.returncode != 0:
        err_tail = (result.stderr or '')[-1500:]
        print(f"[run_stats] finetune_testing exited {result.returncode} (no RESULTS_JSON). stderr tail:\n{err_tail}", flush=True)
    else:
        print(f"[run_stats] finetune_testing exited 0 but no RESULTS_JSON line. stdout tail:\n{result.stdout[-1500:]}", flush=True)

    return None


def write_row_to_csv(metrics: dict, csv_path: str):
    fieldnames = ['item', 'model_size', 'epochs', 'batch_size', 'lr', 'drop_rate', 'weight_decay', 'balanced',
                  'train_accuracy', 'val_accuracy', 'test_accuracy', 'train_loss', 'val_loss', 'test_loss', 'time_minutes']
    row = {
        'item': metrics.get('item', ''),
        'model_size': metrics.get('model_size', ''),
        'epochs': metrics.get('epochs', ''),
        'batch_size': metrics.get('batch_size', ''),
        'lr': metrics.get('lr', ''),
        'drop_rate': metrics.get('drop_rate', ''),
        'weight_decay': metrics.get('weight_decay', ''),
        'balanced': metrics.get('balanced', ''),
        'train_accuracy': metrics.get('train_accuracy', ''),
        'val_accuracy': metrics.get('val_accuracy', ''),
        'test_accuracy': metrics.get('test_accuracy', ''),
        'train_loss': metrics.get('train_loss', ''),
        'val_loss': metrics.get('val_loss', ''),
        'test_loss': metrics.get('test_loss', ''),
        'time_minutes': metrics.get('time_minutes', ''),
    }
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)


def get_next_item_number(csv_path: str) -> int:
    """Count existing data rows in the CSV to determine the next item number."""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            row_count = sum(1 for _ in reader) - 1  # subtract header
            return max(row_count, 0) + 1
    except FileNotFoundError:
        return 1


if __name__ == "__main__":
    # Required on Linux when multiprocessing uses spawn/forkserver (default in Python 3.14+):
    # worker processes re-import this module; without this guard they recurse into the pool.
    csv_path = 'training-data.csv'
    all_permutations = list(permutations(PARAM_GRID))
    total_runs = len(all_permutations)
    max_workers = max_concurrent_workers(PARAM_GRID)

    item_counter_lock = threading.Lock()
    next_item = get_next_item_number(csv_path)

    print(
        f"[run_stats] Sweep: {total_runs} run(s), max_workers={max_workers}, "
        f"appending to {csv_path!r}, next item id = {next_item}",
        flush=True,
    )

    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_test, params): params for params in all_permutations}
        for future in as_completed(futures):
            params = futures[future]
            try:
                metrics = future.result()
                completed += 1
                if metrics:
                    with item_counter_lock:
                        metrics['item'] = next_item
                        item_id = next_item
                        next_item += 1
                        write_row_to_csv(metrics, csv_path)
                    tm = float(metrics.get('time_minutes') or 0)
                    print(
                        f"[run_stats] {completed}/{total_runs} saved item={item_id} "
                        f"val_acc={metrics.get('val_accuracy')} test_acc={metrics.get('test_accuracy')} "
                        f"time_min={tm:.2f} "
                        f"| {params}",
                        flush=True,
                    )
                else:
                    print(
                        f"[run_stats] {completed}/{total_runs} finished with no metrics (see worker logs above) | {params}",
                        flush=True,
                    )
            except Exception as e:
                completed += 1
                print(f"[run_stats] {completed}/{total_runs} error for {params}: {e}", flush=True)
                continue

    print(f"[run_stats] Done. Completed {completed}/{total_runs} job(s). CSV: {csv_path!r}", flush=True)
