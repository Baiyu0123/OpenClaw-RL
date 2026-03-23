#!/usr/bin/env python3
"""
OpenClaw-RL Training Monitor
Parses training log, extracts metrics, saves CSV, and plots curves.
Run periodically or once after training to generate plots.

Usage:
    python monitor_training.py [--log LOG_FILE] [--out OUTPUT_DIR]
"""
import re
import json
import argparse
import os
from collections import defaultdict

def parse_dict_from_log(line):
    """Extract python dict from a log line."""
    match = re.search(r"\{.*\}", line)
    if match:
        try:
            s = match.group(0).replace("'", '"')
            return json.loads(s)
        except:
            return None
    return None

def parse_log(log_path):
    """Parse the training log file and extract all metrics."""
    rollout_metrics = []  # per-rollout
    train_metrics = []    # per-train-step
    perf_metrics = []     # per-step timing
    dropped_info = []     # sample dropping

    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            # Rollout metrics: rollout/raw_reward, log_probs, advantages
            if 'actor.py:519' in line and 'rollout' in line:
                d = parse_dict_from_log(line)
                step_match = re.search(r'rollout (\d+):', line)
                if d and step_match:
                    d['rollout_id'] = int(step_match.group(1))
                    rollout_metrics.append(d)

            # Train step metrics: loss, pg_loss, kl, etc.
            elif 'actor.py:755' in line and 'step' in line:
                d = parse_dict_from_log(line)
                step_match = re.search(r'step (\d+):', line)
                if d and step_match:
                    d['step'] = int(step_match.group(1))
                    train_metrics.append(d)

            # Perf metrics from rollout manager
            elif 'rollout.py:1190' in line and 'perf' in line:
                d = parse_dict_from_log(line)
                step_match = re.search(r'perf (\d+):', line)
                if d and step_match:
                    d['rollout_id'] = int(step_match.group(1))
                    perf_metrics.append(d)

            # Perf metrics from train actor
            elif 'train_metric_utils.py' in line and 'perf' in line:
                d = parse_dict_from_log(line)
                step_match = re.search(r'perf (\d+):', line)
                if d and step_match:
                    d['rollout_id'] = int(step_match.group(1))
                    perf_metrics.append(d)

            # Dropped samples info
            elif 'Dropped constant-reward' in line:
                m = re.search(r'samples (\d+) -> (\d+)', line)
                step_m = re.search(r'perf (\d+)|rollout (\d+)', line)
                if m:
                    dropped_info.append({
                        'total': int(m.group(1)),
                        'kept': int(m.group(2)),
                    })

    return rollout_metrics, train_metrics, perf_metrics, dropped_info

def save_csv(metrics_list, path, prefix=""):
    """Save list of dicts as CSV."""
    if not metrics_list:
        return
    keys = list(metrics_list[0].keys())
    with open(path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for m in metrics_list:
            f.write(','.join(str(m.get(k, '')) for k in keys) + '\n')
    print(f"  Saved {len(metrics_list)} rows to {path}")

def plot_curves(rollout_metrics, train_metrics, perf_metrics, output_dir):
    """Generate training curve plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ---- Figure 1: Reward & Loss (most important) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('OpenClaw-RL Toolcall Training Monitor', fontsize=14, fontweight='bold')

    # 1a. Raw Reward
    if rollout_metrics:
        steps = [m['rollout_id'] for m in rollout_metrics]
        rewards = [m.get('rollout/raw_reward', 0) for m in rollout_metrics]
        axes[0, 0].plot(steps, rewards, 'b-o', markersize=4, linewidth=1.5)
        axes[0, 0].set_title('Rollout Raw Reward')
        axes[0, 0].set_xlabel('Rollout Step')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].grid(True, alpha=0.3)

    # 1b. Training Loss
    if train_metrics:
        steps = [m['step'] for m in train_metrics]
        losses = [m.get('train/loss', 0) for m in train_metrics]
        pg_losses = [m.get('train/pg_loss', 0) for m in train_metrics]
        axes[0, 1].plot(steps, losses, 'r-o', markersize=3, linewidth=1.5, label='total_loss')
        axes[0, 1].plot(steps, pg_losses, 'orange', markersize=2, linewidth=1, alpha=0.7, label='pg_loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Train Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

    # 1c. KL Divergence & Clip Fraction
    if train_metrics:
        steps = [m['step'] for m in train_metrics]
        kl = [m.get('train/ppo_kl', 0) for m in train_metrics]
        clipfrac = [m.get('train/pg_clipfrac', 0) for m in train_metrics]
        ax_kl = axes[1, 0]
        ax_kl.plot(steps, kl, 'g-o', markersize=3, linewidth=1.5, label='PPO KL')
        ax_kl.set_title('KL Divergence & Clip Fraction')
        ax_kl.set_xlabel('Train Step')
        ax_kl.set_ylabel('KL', color='g')
        ax_kl.tick_params(axis='y', labelcolor='g')
        ax_kl.grid(True, alpha=0.3)
        ax2 = ax_kl.twinx()
        ax2.plot(steps, clipfrac, 'm-s', markersize=3, linewidth=1, alpha=0.7, label='Clip Frac')
        ax2.set_ylabel('Clip Fraction', color='m')
        ax2.tick_params(axis='y', labelcolor='m')
        lines1, labels1 = ax_kl.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_kl.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # 1d. Gradient Norm
    if train_metrics:
        steps = [m['step'] for m in train_metrics]
        grad_norm = [m.get('train/grad_norm', 0) for m in train_metrics]
        axes[1, 1].plot(steps, grad_norm, 'purple', marker='o', markersize=3, linewidth=1.5)
        axes[1, 1].set_title('Gradient Norm')
        axes[1, 1].set_xlabel('Train Step')
        axes[1, 1].set_ylabel('Grad Norm')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved main plot: {path}")

    # ---- Figure 2: Rollout Performance ----
    rollout_perfs = [m for m in perf_metrics if 'rollout/response_len/mean' in m]
    if rollout_perfs:
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        fig2.suptitle('Rollout Performance', fontsize=12, fontweight='bold')

        steps = [m['rollout_id'] for m in rollout_perfs]
        resp_len = [m.get('rollout/response_len/mean', 0) for m in rollout_perfs]
        trunc = [m.get('rollout/truncated_ratio', 0) for m in rollout_perfs]
        rep = [m.get('rollout/repetition_frac', 0) for m in rollout_perfs]

        axes2[0].plot(steps, resp_len, 'b-o', markersize=4)
        axes2[0].set_title('Mean Response Length')
        axes2[0].set_xlabel('Rollout Step')
        axes2[0].set_ylabel('Tokens')
        axes2[0].grid(True, alpha=0.3)

        axes2[1].plot(steps, trunc, 'r-o', markersize=4)
        axes2[1].set_title('Truncated Ratio')
        axes2[1].set_xlabel('Rollout Step')
        axes2[1].set_ylabel('Ratio')
        axes2[1].set_ylim(-0.05, 1.05)
        axes2[1].grid(True, alpha=0.3)

        axes2[2].plot(steps, rep, 'orange', marker='o', markersize=4)
        axes2[2].set_title('Repetition Fraction')
        axes2[2].set_xlabel('Rollout Step')
        axes2[2].set_ylabel('Fraction')
        axes2[2].set_ylim(-0.05, 1.05)
        axes2[2].grid(True, alpha=0.3)

        plt.tight_layout()
        path2 = os.path.join(output_dir, 'rollout_perf.png')
        fig2.savefig(path2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"  Saved rollout plot: {path2}")

    # ---- Figure 3: Timing ----
    train_perfs = [m for m in perf_metrics if 'perf/step_time' in m]
    if train_perfs:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        steps = [m['rollout_id'] for m in train_perfs]
        step_time = [m.get('perf/step_time', 0) for m in train_perfs]
        train_time = [m.get('perf/train_time', 0) for m in train_perfs]
        wait_time = [m.get('perf/train_wait_time', 0) for m in train_perfs]

        ax3.bar(steps, wait_time, label='Wait (rollout)', alpha=0.7, color='skyblue')
        ax3.bar(steps, train_time, bottom=wait_time, label='Train', alpha=0.7, color='salmon')
        ax3.set_title('Step Timing Breakdown')
        ax3.set_xlabel('Rollout Step')
        ax3.set_ylabel('Time (s)')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path3 = os.path.join(output_dir, 'timing.png')
        fig3.savefig(path3, dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"  Saved timing plot: {path3}")

def print_summary(rollout_metrics, train_metrics, perf_metrics, dropped_info):
    """Print a text summary of training progress."""
    print("\n" + "=" * 60)
    print("  TRAINING STATUS SUMMARY")
    print("=" * 60)

    if rollout_metrics:
        latest = rollout_metrics[-1]
        print(f"\n  Completed rollouts: {len(rollout_metrics)}")
        print(f"  Latest rollout ({latest['rollout_id']}):")
        print(f"    Raw reward:  {latest.get('rollout/raw_reward', 'N/A'):.4f}")
        print(f"    Advantages:  {latest.get('rollout/advantages', 'N/A'):.6f}")

        if len(rollout_metrics) > 1:
            first_r = rollout_metrics[0].get('rollout/raw_reward', 0)
            last_r = latest.get('rollout/raw_reward', 0)
            print(f"    Reward trend: {first_r:.4f} -> {last_r:.4f} (delta: {last_r - first_r:+.4f})")

    if train_metrics:
        latest = train_metrics[-1]
        print(f"\n  Completed train steps: {len(train_metrics)}")
        print(f"  Latest step ({latest['step']}):")
        print(f"    Loss:        {latest.get('train/loss', 'N/A'):.6f}")
        print(f"    PG Loss:     {latest.get('train/pg_loss', 'N/A'):.6f}")
        print(f"    PPO KL:      {latest.get('train/ppo_kl', 'N/A'):.6f}")
        print(f"    Clip Frac:   {latest.get('train/pg_clipfrac', 'N/A'):.6f}")
        print(f"    Grad Norm:   {latest.get('train/grad_norm', 'N/A'):.6f}")

    if dropped_info:
        total_dropped = sum(d['total'] - d['kept'] for d in dropped_info)
        total_total = sum(d['total'] for d in dropped_info)
        print(f"\n  Sample retention: {total_total - total_dropped}/{total_total} "
              f"({(total_total - total_dropped) / total_total * 100:.1f}%)")

    train_perfs = [m for m in perf_metrics if 'perf/step_time' in m]
    if train_perfs:
        avg_step = sum(m['perf/step_time'] for m in train_perfs) / len(train_perfs)
        print(f"\n  Avg step time: {avg_step:.1f}s")
        remaining = 3000 - len(rollout_metrics)
        if remaining > 0:
            eta_hours = (remaining * avg_step) / 3600
            print(f"  Estimated remaining: {remaining} steps (~{eta_hours:.1f} hours)")

    print("\n" + "=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Monitor OpenClaw-RL training')
    parser.add_argument('--log', default=os.path.expanduser('~/train_toolcall_rl.log'),
                        help='Path to training log file')
    parser.add_argument('--out', default=os.path.expanduser('~/training_monitor'),
                        help='Output directory for plots and CSVs')
    args = parser.parse_args()

    print(f"Parsing log: {args.log}")
    rollout_metrics, train_metrics, perf_metrics, dropped_info = parse_log(args.log)

    os.makedirs(args.out, exist_ok=True)

    # Save CSVs
    save_csv(rollout_metrics, os.path.join(args.out, 'rollout_metrics.csv'))
    save_csv(train_metrics, os.path.join(args.out, 'train_metrics.csv'))

    # Print summary
    print_summary(rollout_metrics, train_metrics, perf_metrics, dropped_info)

    # Plot
    plot_curves(rollout_metrics, train_metrics, perf_metrics, args.out)

    print(f"\nAll outputs saved to: {args.out}/")

if __name__ == '__main__':
    main()
