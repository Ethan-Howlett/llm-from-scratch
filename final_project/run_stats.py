import argparse
import csv
import json
import os
import subprocess
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Upper bound so we do not spawn dozens of CPU-heavy worker processes by mistake.
MAX_CONCURRENT_CAP = 16
# Reserve this much MiB on the target GPU for driver / display / fragmentation.
VRAM_HEADROOM_MIB = 1024
# Rough peak VRAM (MiB) per training process at batch_size≈8; scaled by actual max batch.
MODEL_BASE_MIB = {
    'gpt2-small (124M)': 5500,
    'gpt2-medium (355M)': 12000,
    'gpt2-large (774M)': 22000,
    'gpt2-xl (1558M)': 38000,
}


def script_dir() -> Path:
    return Path(__file__).resolve().parent


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

REFERENCE = {
    'model_size': 'gpt2-medium (355M)',
    'batch_size': 8,
    'epochs': 5,
    'lr': 5e-5,
    'drop_rate': 0.0,
    'weight_decay': 0.1,
    'balance_dataset': 0,
}

SWEEPS = {
    'model_size': ['gpt2-medium (355M)'],
    'batch_size': [3, 5, 8, 13],
    'epochs': [3, 4, 5, 8],
    'lr': [1e-4, 1e-5, 1e-6, 5e-5, 5e-6],
    'drop_rate': [0.0, 0.05],
    'weight_decay': [0.1, 0.2],
    'balance_dataset': [0, 1],
}

SEEDS = [random.randint(1, 1000) for _ in range(30)]

# CSV / CLI use 'balanced'; sweep tag from OFAT is 'balance_dataset'.
SWEEP_PARAM_TO_CSV_COL: dict[str, str] = {'balance_dataset': 'balanced'}


def csv_column_for_sweep(sweep_name: str) -> str:
    return SWEEP_PARAM_TO_CSV_COL.get(sweep_name, sweep_name)


def generate_ofat_runs(reference: dict, sweeps: dict, seeds: list[int]) -> list[dict]:
    """One-factor-at-a-time: vary each factor while holding others at reference.

    Reference-level runs (sweep='reference') are shared across all factor analyses.
    Non-reference levels are tagged with sweep=<factor_name>.
    Every condition is repeated once per seed for replication.
    """
    runs: list[dict] = []
    for seed in seeds:
        run = dict(reference)
        run['seed'] = seed
        run['sweep'] = 'reference'
        runs.append(run)
    for factor, levels in sweeps.items():
        for level in levels:
            if level == reference[factor]:
                continue
            params = dict(reference)
            params[factor] = level
            for seed in seeds:
                run = dict(params)
                run['seed'] = seed
                run['sweep'] = factor
                runs.append(run)
    return runs


_NON_CLI_KEYS = {'sweep'}


def _run_test_job(payload: tuple[dict, str]) -> dict | None:
    """Worker: run finetune_testing in script_dir. Picklable top-level for ProcessPoolExecutor."""
    params, cwd = payload
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    finetune = Path(cwd) / 'finetune_testing.py'
    cmd = ['python3', str(finetune)]
    for key, val in params.items():
        if key not in _NON_CLI_KEYS:
            cmd += [f'--{key}', str(val)]

    print(f"\033[32m[run_stats] Starting subprocess: {' '.join(cmd)}\033[0m", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)

    try:
        for line in result.stdout.splitlines():
            if line.startswith('RESULTS_JSON:'):
                return json.loads(line[len('RESULTS_JSON:'):])
    except json.JSONDecodeError:
        print(f"\033[31m[run_stats] JSON parse error (exit {result.returncode}). stdout tail:\n{result.stdout[-1500:]}\033[0m", flush=True)
        return None

    if result.returncode != 0:
        err_tail = (result.stderr or '')[-1500:]
        print(f"\033[31m[run_stats] finetune_testing exited {result.returncode} (no RESULTS_JSON). stderr tail:\n{err_tail}\033[0m", flush=True)
    else:
        print(f"\033[31m[run_stats] finetune_testing exited 0 but no RESULTS_JSON line. stdout tail:\n{result.stdout[-1500:]}\033[0m", flush=True)

    return None


CSV_FIELDNAMES = [
    'item', 'sweep', 'seed',
    'model_size', 'epochs', 'batch_size', 'lr', 'drop_rate', 'weight_decay', 'balanced',
    'train_accuracy', 'val_accuracy', 'test_accuracy',
    'train_loss', 'val_loss', 'test_loss',
    'time_minutes',
]


def ensure_csv_header(csv_path: str):
    """Write the header row if the file does not exist or is empty."""
    try:
        with open(csv_path, 'r') as f:
            if f.readline().strip():
                return
    except FileNotFoundError:
        pass
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writeheader()


def truncate_csv_fresh(csv_path: str):
    """Remove existing data; write header only. Next row item should be 1."""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writeheader()


def write_row_to_csv(metrics: dict, csv_path: str):
    row = {field: metrics.get(field, '') for field in CSV_FIELDNAMES}
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        writer.writerow(row)


def get_next_item_number(csv_path: str) -> int:
    """Next item id: max existing item + 1, or 1 if file missing / no data rows."""
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            mx = 0
            any_row = False
            for row in reader:
                any_row = True
                raw = row.get('item', '') or ''
                try:
                    mx = max(mx, int(float(raw)))
                except (ValueError, TypeError):
                    continue
            if not any_row or mx == 0:
                return 1
            return mx + 1
    except FileNotFoundError:
        return 1


def run_experiments(csv_path: str, project_dir: str, *, fresh: bool) -> None:
    all_runs = generate_ofat_runs(REFERENCE, SWEEPS, SEEDS)
    total_runs = len(all_runs)

    vram_grid = {
        'model_size': list({REFERENCE['model_size']} | set(SWEEPS.get('model_size', []))),
        'batch_size': list({REFERENCE['batch_size']} | set(SWEEPS.get('batch_size', []))),
    }
    max_workers = max_concurrent_workers(vram_grid)

    if fresh:
        next_item = 1
    else:
        ensure_csv_header(csv_path)
        next_item = get_next_item_number(csv_path)

    # Mutable counter for the main thread (lock protects concurrent writes).
    item_seq = [next_item]
    item_counter_lock = threading.Lock()

    ref_levels = len(SEEDS)
    non_ref_levels = sum(
        len([v for v in levels if v != REFERENCE[factor]])
        for factor, levels in SWEEPS.items()
    )
    print(
        f"[run_stats] OFAT sweep: {total_runs} run(s) "
        f"({ref_levels} reference + {non_ref_levels} non-ref levels × {len(SEEDS)} seeds), "
        f"max_workers={max_workers}, csv={csv_path!r}, next item id = {next_item}",
        flush=True,
    )

    completed = 0
    payloads = [(params, project_dir) for params in all_runs]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_test_job, p): p[0] for p in payloads}
        for future in as_completed(futures):
            params = futures[future]
            try:
                metrics = future.result()
                completed += 1
                if metrics:
                    with item_counter_lock:
                        metrics['item'] = item_seq[0]
                        metrics['sweep'] = params.get('sweep', '')
                        item_id = item_seq[0]
                        item_seq[0] += 1
                        write_row_to_csv(metrics, csv_path)
                    tm = float(metrics.get('time_minutes') or 0)
                    print(
                        f"\033[34m[run_stats] {completed}/{total_runs} saved item={item_id} "
                        f"sweep={params.get('sweep')} seed={params.get('seed')} "
                        f"test_acc={metrics.get('test_accuracy')} "
                        f"time_min={tm:.2f}\033[0m",
                        flush=True,
                    )
                else:
                    print(
                        f"\033[31m[run_stats] {completed}/{total_runs} no metrics | {params}\033[0m",
                        flush=True,
                    )
            except Exception as e:
                completed += 1
                print(f"\033[31m[run_stats] {completed}/{total_runs} error for {params}: {e}\033[0m", flush=True)
                continue

    print(f"[run_stats] Done. {completed}/{total_runs} job(s). CSV: {csv_path!r}", flush=True)


def _load_training_df(csv_path: str) -> pd.DataFrame | None:
    p = Path(csv_path)
    if not p.is_file():
        print(f"[run_stats] Analysis: CSV not found: {csv_path!r}", flush=True)
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        print('[run_stats] Analysis: CSV has no data rows.', flush=True)
        return None
    num_cols = [
        'item', 'seed', 'epochs', 'batch_size', 'lr', 'drop_rate', 'weight_decay', 'balanced',
        'train_accuracy', 'val_accuracy', 'test_accuracy',
        'train_loss', 'val_loss', 'test_loss', 'time_minutes',
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _factor_level_row(row: pd.Series) -> str:
    if row.get('sweep') == 'reference':
        return 'reference'
    sw = row.get('sweep')
    col = csv_column_for_sweep(str(sw)) if pd.notna(sw) else ''
    if not col or col not in row.index:
        return ''
    v = row.get(col)
    return str(v) if pd.notna(v) else ''


def run_analysis(csv_path: str, out_dir: str) -> None:
    """Load CSV; generate plots of mean test_accuracy per sweep (no statistical tests)."""
    os.makedirs(out_dir, exist_ok=True)
    df = _load_training_df(csv_path)
    if df is None:
        return

    df = df.copy()
    df['factor_level'] = df.apply(_factor_level_row, axis=1)
    _plot_mean_test_accuracy_by_balanced(df, out_dir=out_dir)
    # Default threshold for "noticeable" can be overridden by CLI.
    min_delta = float(os.environ.get('RUN_STATS_MIN_DELTA', '0.01'))
    _save_test_accuracy_sweep_plots(df, out_dir=out_dir, min_delta=min_delta)


def _plot_mean_test_accuracy_by_balanced(df: pd.DataFrame, *, out_dir: str) -> None:
    """Bar chart: mean test_accuracy for balanced=0 vs balanced=1 (±SE)."""
    if 'balanced' not in df.columns:
        print('[run_stats] Skip balanced plot: no balanced column.', flush=True)
        return
    sub = df.dropna(subset=['test_accuracy', 'balanced'])
    if sub.empty:
        print('[run_stats] Skip balanced plot: no test_accuracy / balanced rows.', flush=True)
        return

    agg = (
        sub.groupby('balanced', dropna=False)
        .agg(mean=('test_accuracy', 'mean'), std=('test_accuracy', 'std'), n=('test_accuracy', 'count'))
        .reset_index()
    )
    agg['se'] = agg.apply(
        lambda r: (float(r['std']) / np.sqrt(float(r['n']))) if r['n'] and r['n'] > 1 and pd.notna(r['std']) else 0.0,
        axis=1,
    )
    # Order: unbalanced (0) then balanced (1) when both exist
    try:
        agg['_sort'] = agg['balanced'].map(lambda x: (0 if float(x) == 0 else 1 if float(x) == 1 else 2, x))
        agg = agg.sort_values('_sort').drop(columns=['_sort'])
    except Exception:
        agg = agg.sort_values('balanced')

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    x = np.arange(len(agg))
    colors = ['#4a8f6f', '#c75d5d']
    ax.bar(
        x,
        agg['mean'],
        yerr=agg['se'],
        capsize=5,
        color=[colors[i % len(colors)] for i in range(len(agg))],
        edgecolor='#333',
        linewidth=0.8,
    )

    def _label(bal) -> str:
        try:
            b = int(float(bal))
            return 'Unbalanced' if b == 0 else 'Balanced' if b == 1 else str(bal)
        except Exception:
            return str(bal)

    ax.set_xticks(x)
    ax.set_xticklabels([_label(v) for v in agg['balanced'].tolist()])
    ax.set_ylabel('mean test_accuracy (±SE)')
    ax.set_xlabel('Dataset balancing (balanced flag)')
    ax.set_ylim(0.0, 1.05)
    ax.set_title('Mean test accuracy by balanced vs not balanced')
    for j, (_, r) in enumerate(agg.iterrows()):
        ax.text(
            j,
            float(r['mean']) + float(r['se']) + 0.02,
            f"n={int(r['n'])}",
            ha='center',
            fontsize=9,
        )
    fig.tight_layout()
    out = Path(out_dir) / 'mean_test_accuracy_by_balanced.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[run_stats] Wrote {out}", flush=True)


def _save_test_accuracy_sweep_plots(df: pd.DataFrame, *, out_dir: str, min_delta: float) -> None:
    """
    For each sweep, plot mean test_accuracy (and mean val_accuracy when available) vs the varied factor level.
    Only save plots when the change is noticeable:
      max_level |mean(level) - mean(reference)| >= min_delta
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref = df[df['sweep'] == 'reference'].dropna(subset=['test_accuracy'])
    if ref.empty:
        print('[run_stats] No reference rows; cannot compute baseline for filtering.', flush=True)
        return
    ref_mean = float(ref['test_accuracy'].mean())
    ref_n = int(ref['test_accuracy'].notna().sum())
    ref_std = float(ref['test_accuracy'].std(ddof=1)) if ref_n > 1 else 0.0
    ref_se = ref_std / np.sqrt(ref_n) if ref_n > 1 else 0.0

    ref_val_mean: float | None = None
    ref_val_se = 0.0
    ref_val_n = 0
    if 'val_accuracy' in ref.columns and ref['val_accuracy'].notna().any():
        rv = ref['val_accuracy'].dropna()
        ref_val_n = int(len(rv))
        if ref_val_n:
            ref_val_mean = float(rv.mean())
            ref_val_std = float(rv.std(ddof=1)) if ref_val_n > 1 else 0.0
            ref_val_se = ref_val_std / np.sqrt(ref_val_n) if ref_val_n > 1 else 0.0

    sweep_names = [s for s in sorted(df['sweep'].dropna().unique()) if s != 'reference']
    kept = 0
    for sweep_name in sweep_names:
        fac_col = csv_column_for_sweep(sweep_name)
        sub = df[df['sweep'] == sweep_name].dropna(subset=['test_accuracy', fac_col])
        if sub.empty:
            continue

        # Aggregate mean and standard error (SE = sd/sqrt(n)).
        agg = (
            sub.groupby(fac_col, dropna=False)
            .agg(
                mean=('test_accuracy', 'mean'),
                std=('test_accuracy', 'std'),
                n=('test_accuracy', 'count'),
                val_mean=('val_accuracy', 'mean'),
                val_std=('val_accuracy', 'std'),
                val_n=('val_accuracy', lambda s: int(s.notna().sum())),
            )
            .reset_index()
        )
        agg['se'] = agg.apply(lambda r: (r['std'] / np.sqrt(r['n'])) if r['n'] and r['n'] > 1 else 0.0, axis=1)
        agg['val_se'] = agg.apply(
            lambda r: (float(r['val_std']) / np.sqrt(float(r['val_n'])))
            if r['val_n'] and r['val_n'] > 1 and pd.notna(r['val_std'])
            else 0.0,
            axis=1,
        )
        agg['delta_vs_ref'] = agg['mean'] - ref_mean

        max_abs_delta = float(np.nanmax(np.abs(agg['delta_vs_ref'].to_numpy())))
        if not (max_abs_delta >= min_delta):
            continue

        kept += 1

        # Sort x-axis by numeric where possible.
        x_vals = agg[fac_col].tolist()
        try:
            order = np.argsort([float(x) for x in x_vals])
        except Exception:
            order = np.argsort([str(x) for x in x_vals])
        agg = agg.iloc[order].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        dodge = 0.12
        show_val = ref_val_mean is not None and ref_val_n > 0
        show_val_alts = bool(show_val and len(agg) and int(agg['val_n'].sum()) > 0)

        # Reference first (same OFAT baseline for every sweep); square marker, not connected to sweep line.
        ref_label = 'reference'
        try:
            # Show the baseline value for this factor, e.g. epochs=5, lr=5e-05, balanced=1.
            ref_key = sweep_name
            if ref_key in REFERENCE:
                ref_label = f"{fac_col}={REFERENCE[ref_key]}"
        except Exception:
            ref_label = 'reference'
        ax.errorbar(
            [0 - dodge],
            [ref_mean],
            yerr=[ref_se],
            fmt='s',
            color='C0',
            capsize=5,
            markersize=7,
            label=f'{ref_label} test (n={ref_n})',
            zorder=4,
        )
        if show_val:
            ax.errorbar(
                [0 + dodge],
                [ref_val_mean],
                yerr=[ref_val_se],
                fmt='s',
                color='C1',
                capsize=5,
                markersize=7,
                label=f'{ref_label} val (n={ref_val_n})',
                zorder=4,
            )
        if len(agg) > 0:
            xs = np.arange(1, 1 + len(agg))
            ax.errorbar(
                xs - dodge,
                agg['mean'],
                yerr=agg['se'],
                fmt='o-',
                capsize=4,
                linewidth=1.5,
                markersize=6,
                color='C0',
                label=f'{sweep_name} test',
                zorder=2,
            )
            if show_val_alts:
                ax.errorbar(
                    xs + dodge,
                    agg['val_mean'],
                    yerr=agg['val_se'],
                    fmt='s--',
                    capsize=4,
                    linewidth=1.5,
                    markersize=6,
                    color='C1',
                    label=f'{sweep_name} val',
                    zorder=2,
                )
        tick_x = np.arange(0.0, 1.0 + len(agg))
        ax.set_xticks(tick_x)
        xtick = [ref_label] + [str(v) for v in agg[fac_col].tolist()]
        ax.set_xticklabels(xtick, rotation=22, ha='right')
        ax.set_ylabel('mean accuracy (±SE)')
        ax.set_ylim(0.0, 1.05)
        ax.set_title(
            f'test vs val accuracy vs {fac_col} (sweep={sweep_name}; baseline + alternatives; min_delta={min_delta})'
        )
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        out_path = out / f'sweep_{sweep_name}__test_accuracy.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[run_stats] Wrote {out_path}", flush=True)

    print(f"[run_stats] Plots saved for {kept}/{len(sweep_names)} sweeps (min_delta={min_delta}).", flush=True)


def parse_args() -> argparse.Namespace:
    sd = script_dir()
    p = argparse.ArgumentParser(
        description='OFAT finetune sweep, CSV logging, and analysis focused on val vs test accuracy.',
    )
    mx = p.add_mutually_exclusive_group()
    mx.add_argument(
        '--test',
        action='store_true',
        help='Run full OFAT training sweep and append rows to the CSV, then run analysis.',
    )
    mx.add_argument(
        '--test-fresh',
        action='store_true',
        help='Clear the CSV (header only), run full OFAT sweep with item starting at 1, then analysis.',
    )
    p.add_argument(
        '--csv',
        type=Path,
        default=None,
        help=f'Path to training CSV (default: {sd / "training-data.csv"})',
    )
    p.add_argument(
        '--out-dir',
        type=Path,
        default=None,
        help=f'Output directory for charts and summary CSV (default: {sd / "stats_output"})',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sd = script_dir()
    project_dir = str(sd)
    csv_path = str(args.csv) if args.csv is not None else str(sd / 'training-data.csv')
    out_dir = str(args.out_dir) if args.out_dir is not None else str(sd / 'stats_output')

    if args.test_fresh:
        truncate_csv_fresh(csv_path)
        run_experiments(csv_path, project_dir, fresh=True)
    elif args.test:
        run_experiments(csv_path, project_dir, fresh=False)

    run_analysis(csv_path, out_dir)


if __name__ == '__main__':
    main()
