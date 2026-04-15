"""
Run full finetuning vs LoRA via subprocesses on `gpt_class_finetune.py`, collect
FINETUNE_METRICS_JSON lines, and save comparison figures (similar workflow to run_stats.py).

Parallel runs: use two GPUs (--gpu-full / --gpu-lora) or pass --sequential for one GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS_PREFIX = "FINETUNE_METRICS_JSON:"
MAX_COMPARE_WORKERS = 4


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def _gpu_count() -> int:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 0
    if proc.returncode != 0:
        return 0
    return sum(1 for line in (proc.stdout or "").splitlines() if line.startswith("GPU "))


def _run_compare_job(
    payload: tuple[str, str, str | None, str, list[str]],
) -> dict | None:
    """Worker: cwd, python_exe, cuda_visible_devices (or None), run_label, extra_cli_args."""
    project_dir, python_exe, cuda_device, run_label, extra_args = payload
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    if cuda_device is not None and str(cuda_device).strip() != "":
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device).strip()

    script_path = Path(project_dir) / "gpt_class_finetune.py"
    cmd = [
        python_exe,
        str(script_path),
        "--emit-metrics-json",
        "--skip-plots",
        *extra_args,
    ]
    print(f"\033[32m[compare_finetuning:{run_label}] {' '.join(cmd)}\033[0m", flush=True)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=project_dir,
    )
    for line in (result.stdout or "").splitlines():
        if line.startswith(METRICS_PREFIX):
            try:
                metrics = json.loads(line[len(METRICS_PREFIX) :])
                metrics["_run_label"] = run_label
                metrics["_returncode"] = int(result.returncode)
                if result.returncode != 0:
                    print(
                        f"\033[33m[compare_finetuning:{run_label}] exit {result.returncode} "
                        f"(still parsed metrics)\033[0m",
                        flush=True,
                    )
                return metrics
            except json.JSONDecodeError:
                break

    err_tail = (result.stderr or "")[-2000:]
    out_tail = (result.stdout or "")[-2000:]
    print(
        f"\033[31m[compare_finetuning:{run_label}] No {METRICS_PREFIX} line. "
        f"exit={result.returncode}\n--- stderr ---\n{err_tail}\n--- stdout ---\n{out_tail}\033[0m",
        flush=True,
    )
    return None


def _shared_cli(
    *,
    model_size: str,
    num_epochs: int,
    batch_size: int,
) -> list[str]:
    return [
        "--model-size",
        model_size,
        "--num-epochs",
        str(num_epochs),
        "--batch-size",
        str(batch_size),
    ]


def plot_summary_dashboard(
    full: dict,
    lora: dict,
    *,
    outfile: Path,
) -> None:
    """Side-by-side final metrics, wall time, and end-of-run validation loss."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")

    labels = ["Train", "Val", "Test"]
    x = np.arange(len(labels))
    w = 0.35
    full_scores = [
        full["final_train_accuracy"],
        full["final_val_accuracy"],
        full["final_test_accuracy"],
    ]
    lora_scores = [
        lora["final_train_accuracy"],
        lora["final_val_accuracy"],
        lora["final_test_accuracy"],
    ]
    axes[0].bar(x - w / 2, full_scores, width=w, label="Full fine-tune", color="#2c5282")
    axes[0].bar(x + w / 2, lora_scores, width=w, label="LoRA", color="#c05621")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Final accuracy")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    t_x = np.arange(2)
    t_heights = [full["time_minutes"], lora["time_minutes"]]
    axes[1].bar(t_x, t_heights, color=["#2c5282", "#c05621"], width=0.55)
    axes[1].set_xticks(t_x, ["Full", "LoRA"])
    axes[1].set_ylabel("Minutes")
    axes[1].set_title("Training wall time")
    axes[1].grid(axis="y", alpha=0.3)
    t_max = max(full["time_minutes"], lora["time_minutes"], 0.05)
    y1_hi = axes[1].get_ylim()[1]
    for i, t in enumerate(t_heights):
        axes[1].text(i, min(t + 0.04 * t_max, 0.96 * y1_hi), f"{t:.2f}", ha="center", fontsize=9)

    full_vl = full["val_losses"][-1] if full["val_losses"] else float("nan")
    lora_vl = lora["val_losses"][-1] if lora["val_losses"] else float("nan")
    vl_x = np.arange(2)
    axes[2].bar(vl_x, [full_vl, lora_vl], color=["#2c5282", "#c05621"], width=0.55)
    axes[2].set_xticks(vl_x, ["Full", "LoRA"])
    axes[2].set_ylabel("Validation loss")
    axes[2].set_title("Last logged val loss (same step)")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Full fine-tuning vs LoRA — summary", fontsize=12, y=1.02)
    plt.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_finetuning] Wrote {outfile}", flush=True)


def plot_confusion_matrix(cm: list[list[int]], labels: list[str], *, title: str, outfile: Path) -> None:
    """Save a confusion-matrix heatmap."""
    cm_arr = np.array(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_labels = [l.replace("_", "\n") for l in labels]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)

    thresh = cm_arr.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_arr[i, j] > thresh else "black")

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_finetuning] Wrote {outfile}", flush=True)


def plot_gap_curves(full: dict, lora: dict, *, outfile: Path) -> None:
    """Val-loss gap (full − LoRA) over training steps; val-acc gap over epochs."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=False, layout="constrained")

    s_full = np.array(full["loss_eval_steps"], dtype=float)
    s_lora = np.array(lora["loss_eval_steps"], dtype=float)
    vf = np.array(full["val_losses"], dtype=float)
    vl = np.array(lora["val_losses"], dtype=float)
    n = min(len(s_full), len(s_lora), len(vf), len(vl))
    if n == 0:
        plt.close(fig)
        return
    steps = s_full[:n]
    gap_loss = vf[:n] - vl[:n]
    ax1.plot(steps, gap_loss, color="#553c9a", linewidth=1.5, label="val loss (full − LoRA)")
    ax1.axhline(0.0, color="gray", linewidth=0.8)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Δ val loss")
    ax1.set_title("Validation loss gap along training")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.ticklabel_format(style="plain", axis="x", useOffset=False)

    ef = np.array(full["val_accs"], dtype=float)
    el = np.array(lora["val_accs"], dtype=float)
    m = min(len(ef), len(el))
    if m > 0:
        epochs = np.arange(1, m + 1)
        ax2.plot(epochs, ef[:m] - el[:m], color="#276749", linewidth=1.5, marker="o", markersize=4)
        ax2.axhline(0.0, color="gray", linewidth=0.8)
        ax2.set_xticks(epochs)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Δ val accuracy (full − LoRA)")
    ax2.set_title("Per-epoch validation accuracy gap")
    ax2.grid(alpha=0.3)

    plt.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_finetuning] Wrote {outfile}", flush=True)

def plot_compare_full_vs_lora(epochs_loss, examples_loss, full_train_loss, full_val_loss, lora_train_loss,
    lora_val_loss, epochs_acc, examples_acc, full_train_acc, full_val_acc, lora_train_acc, lora_val_acc, outfile="compare-full-vs-lora.pdf"):
    """One figure: loss (both methods) and accuracy (both methods).

    Uses a single x-axis per panel (training step / epoch). A second top axis for
    \"examples seen\" misaligns tick marks with the curves because points are
    plotted in step/epoch space, not example space.
    """
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=False, layout="constrained")
    ax_loss.plot(epochs_loss, full_train_loss, label="Full train", linestyle="-", color="C0")
    ax_loss.plot(epochs_loss, full_val_loss, label="Full val", linestyle="--", color="C0")
    ax_loss.plot(epochs_loss, lora_train_loss, label="LoRA train", linestyle="-", color="C1")
    ax_loss.plot(epochs_loss, lora_val_loss, label="LoRA val", linestyle="--", color="C1")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_xlabel("Training step (eval checkpoints)")
    ax_loss.legend(loc="best", fontsize=8)
    ax_loss.set_title("Training / validation loss")
    ax_loss.ticklabel_format(style="plain", axis="x", useOffset=False)
    if examples_loss:
        ex0, ex1 = int(examples_loss[0]), int(examples_loss[-1])
        ax_loss.text(
            0.02,
            0.04,
            f"Examples seen: {ex0} … {ex1} (same checkpoints)",
            transform=ax_loss.transAxes,
            fontsize=7,
            verticalalignment="bottom",
            color="0.35",
        )
    ax_acc.plot(epochs_acc, full_train_acc, label="Full train", linestyle="-", color="C0")
    ax_acc.plot(epochs_acc, full_val_acc, label="Full val", linestyle="--", color="C0")
    ax_acc.plot(epochs_acc, lora_train_acc, label="LoRA train", linestyle="-", color="C1")
    ax_acc.plot(epochs_acc, lora_val_acc, label="LoRA val", linestyle="--", color="C1")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.legend(loc="best", fontsize=8)
    ax_acc.set_title("Training / validation accuracy")
    ax_acc.set_xticks(list(epochs_acc))
    if examples_acc:
        ax_acc.text(
            0.02,
            0.04,
            f"Cumulative examples after each epoch: {int(examples_acc[0])} … {int(examples_acc[-1])}",
            transform=ax_acc.transAxes,
            fontsize=7,
            verticalalignment="bottom",
            color="0.35",
        )
    plt.savefig(outfile)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    sd = script_dir()
    p = argparse.ArgumentParser(
        description="Compare full finetuning vs LoRA by driving gpt_class_finetune.py (two runs).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=sd / "compare_finetune_output",
        help="Directory for figures and JSON metrics dumps.",
    )
    p.add_argument(
        "--model-size",
        type=str,
        default="gpt2-medium (355M)",
        help="Passed through to gpt_class_finetune.py",
    )
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter for subprocesses",
    )
    p.add_argument(
        "--gpu-full",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES for the full fine-tune job (empty = inherit env).",
    )
    p.add_argument(
        "--gpu-lora",
        type=str,
        default="1",
        help="CUDA_VISIBLE_DEVICES for the LoRA job (empty = inherit env).",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Process pool size; reduced to 1 if only one GPU is detected (unless --force-parallel).",
    )
    p.add_argument(
        "--sequential",
        action="store_true",
        help="Always run the two trainings one after another on this machine.",
    )
    p.add_argument(
        "--force-parallel",
        action="store_true",
        help="Use --max-workers even if only one GPU is reported (OOM risk).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sd = script_dir()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared = _shared_cli(
        model_size=args.model_size,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )
    full_args = [*shared, "--trainable-layers", "all"]
    lora_args = [
        *shared,
        "--trainable-layers",
        "lora",
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
    ]

    ngpu = _gpu_count()
    parallel = not args.sequential
    workers = max(1, min(int(args.max_workers), MAX_COMPARE_WORKERS))
    # One physical GPU cannot safely run two default VRAM-heavy finetunes at once.
    if (
        parallel
        and ngpu == 1
        and workers > 1
        and not args.force_parallel
    ):
        print(
            "[compare_finetuning] Single GPU detected; running sequentially "
            "(set --force-parallel to try both runs at once, or use two GPUs via "
            "--gpu-full / --gpu-lora).",
            flush=True,
        )
        parallel = False

    job_full = (str(sd), args.python, args.gpu_full or None, "full", full_args)
    job_lora = (str(sd), args.python, args.gpu_lora or None, "lora", lora_args)
    jobs = [job_full, job_lora]

    if not parallel:
        results = [_run_compare_job(j) for j in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            future_map = {ex.submit(_run_compare_job, j): j for j in jobs}
            for fut in as_completed(future_map):
                results.append(fut.result())

    by_label: dict[str, dict] = {}
    for m in results:
        if m and "_run_label" in m:
            by_label[m["_run_label"]] = m

    if "full" not in by_label or "lora" not in by_label:
        print("[compare_finetuning] Missing metrics for one or both runs; aborting plots.", flush=True)
        sys.exit(1)

    full_m = by_label["full"]
    lora_m = by_label["lora"]

    (out_dir / "metrics_full.json").write_text(json.dumps(full_m, indent=2), encoding="utf-8")
    (out_dir / "metrics_lora.json").write_text(json.dumps(lora_m, indent=2), encoding="utf-8")

    # Align loss curves to common x for the shared helper (same schedule → same lengths).
    n_steps = min(
        len(full_m["loss_eval_steps"]),
        len(lora_m["loss_eval_steps"]),
        len(full_m["train_losses"]),
        len(lora_m["train_losses"]),
    )
    steps = full_m["loss_eval_steps"][:n_steps]
    ex_loss = full_m["loss_eval_examples"][:n_steps]
    f_tr = full_m["train_losses"][:n_steps]
    f_va = full_m["val_losses"][:n_steps]
    l_tr = lora_m["train_losses"][:n_steps]
    l_va = lora_m["val_losses"][:n_steps]

    n_ep = min(len(full_m["train_accs"]), len(lora_m["train_accs"]))
    epochs_axis = list(range(1, n_ep + 1))
    f_te = full_m["epoch_end_examples"][:n_ep]
    # Use full's example counts for twiny (LoRA will match closely on same data).
    f_ta = full_m["train_accs"][:n_ep]
    f_va_ep = full_m["val_accs"][:n_ep]
    l_ta = lora_m["train_accs"][:n_ep]
    l_va_ep = lora_m["val_accs"][:n_ep]

    compare_pdf = out_dir / "compare-full-vs-lora.png"
    plot_compare_full_vs_lora(steps, ex_loss, f_tr, f_va, l_tr, l_va, epochs_axis, list(f_te), f_ta, f_va_ep, l_ta, l_va_ep, outfile=str(compare_pdf))


    plot_summary_dashboard(full_m, lora_m, outfile=out_dir / "compare-summary.png")
    plot_gap_curves(full_m, lora_m, outfile=out_dir / "compare-gaps.png")

    for label, metrics in [("full", full_m), ("lora", lora_m)]:
        cm = metrics.get("confusion_matrix")
        cm_labels = metrics.get("confusion_matrix_labels")
        if cm and cm_labels:
            plot_confusion_matrix(
                cm, cm_labels,
                title=f"Confusion Matrix — {label.upper()} fine-tuning",
                outfile=out_dir / f"confusion_matrix_{label}.png",
            )

    print(f"[compare_finetuning] Done. Outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
