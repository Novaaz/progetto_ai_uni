import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from src.utils.constants import RESULTS_DIR
except Exception:
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def _gather_experiment_dirs(experiments_root):
    if not os.path.isdir(experiments_root):
        raise FileNotFoundError(f"experiments root not found: {experiments_root}")
    dirs = sorted([d for d in glob.glob(os.path.join(experiments_root, "exp_*")) if os.path.isdir(d)])
    return dirs


def _read_steps_csv(path):
    try:
        df = pd.read_csv(path)
        if 'reward' in df.columns:
            return np.array(df['reward'].values, dtype=float)
    except Exception:
        pass
    return np.array([], dtype=float)


def aggregate_per_model(experiments_root=None, models=("baseline","h20")):
    experiments_root = experiments_root or os.path.join(RESULTS_DIR, "experiments")
    exp_dirs = _gather_experiment_dirs(experiments_root)
    data = {m: [] for m in models}
    finals = {m: [] for m in models}

    for exp in exp_dirs:
        evals_dir = os.path.join(exp, "evals")
        if not os.path.isdir(evals_dir):
            continue
        for m in models:
            steps_path = os.path.join(evals_dir, f"{m}_steps.csv")
            summary_path = os.path.join(evals_dir, f"{m}_summary.csv")
            if os.path.exists(steps_path):
                arr = _read_steps_csv(steps_path)
                if arr.size > 0:
                    data[m].append(arr)
            final = None
            if os.path.exists(summary_path):
                try:
                    df = pd.read_csv(summary_path)
                    if not df.empty and 'final_reward' in df.columns:
                        final = float(df.iloc[0]['final_reward'])
                except Exception:
                    final = None
            else:
                for f in glob.glob(os.path.join(exp, "eval_summary_seed*.csv")):
                    try:
                        df = pd.read_csv(f)
                        row = df[df['model'] == m]
                        if not row.empty:
                            final = float(row.iloc[0].get('final_reward', np.nan))
                            break
                    except Exception:
                        continue
            if final is not None and not np.isnan(final):
                finals[m].append(final)
    return data, finals


def compute_mean_std_per_step(arrays):
    if not arrays:
        return np.array([]), np.array([])
    min_len = min([len(a) for a in arrays])
    stacked = np.vstack([a[:min_len] for a in arrays])
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    return mean, std


def compute_final_stats(finals: dict):
    """Ritorna dict di statistiche (mean, std, count) e improvement rispetto a baseline if present."""
    stats = {}
    baseline_mean = None
    if 'baseline' in finals and finals['baseline']:
        baseline_mean = float(np.mean(finals['baseline']))
    for m, vals in finals.items():
        if not vals:
            stats[m] = {'mean': np.nan, 'std': np.nan, 'count': 0, 'improvement_pct': np.nan}
            continue
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        count = len(vals)
        if baseline_mean is None:
            imp = np.nan
        else:
            denom = abs(baseline_mean) + 1e-8
            imp = ((mean_v - baseline_mean) / denom) * 100.0
        stats[m] = {'mean': mean_v, 'std': std_v, 'count': count, 'improvement_pct': imp}
    return stats


def plot_mean_curves(data, finals_stats, out_dir, smooth_window=5):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12,6))
    palette = sns.color_palette("tab10", n_colors=max(3, len(data)))
    for i, (model, arrays) in enumerate(data.items()):
        if not arrays:
            continue
        mean, std = compute_mean_std_per_step(arrays)
        # smoothing via rolling mean
        s = pd.Series(mean)
        ma = s.rolling(window=max(1, smooth_window), min_periods=1).mean().to_numpy()
        x = np.arange(len(ma))
        stats = finals_stats.get(model, {})
        mean_final = stats.get('mean', np.nan)
        imp = stats.get('improvement_pct', np.nan)
        if not np.isnan(mean_final):
            if model == 'baseline':
                label = f"{model} (final={mean_final:.1f})"
            else:
                label = f"{model} (final={mean_final:.1f}"
                if not np.isnan(imp):
                    label += f", Δ%={imp:+.1f}%"
                label += ")"
        else:
            label = model
        plt.plot(x, ma, color=palette[i], linewidth=2.6, label=label)
        plt.fill_between(x, ma - std[:len(ma)], ma + std[:len(ma)], color=palette[i], alpha=0.18)
        pd.DataFrame({"step": x, "mean": ma, "std": std[:len(ma)]}).to_csv(os.path.join(out_dir, f"{model}_mean_per_step.csv"), index=False)

    plt.xlabel("Step")
    plt.ylabel("Reward per step (mean across seeds)")
    plt.title("Mean reward per step (models aggregated across seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(out_dir, "mean_steps_models.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def plot_final_boxplot(finals, finals_stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for model, vals in finals.items():
        for v in vals:
            rows.append({"model": model, "final_reward": v})
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df, x='model', y='final_reward', hue='model', palette="Set2", dodge=False)
    sns.swarmplot(data=df, x='model', y='final_reward', hue='model', palette={m:'k' for m in df['model'].unique()},
                  dodge=False, alpha=0.6, size=4, linewidth=0)
    plt.legend([],[], frameon=False)
    plt.title("Final reward distribution per model (seeds)")
    plt.ylabel("Final reward")
    plt.xlabel("")
    plt.grid(alpha=0.2)
    models = sorted(df['model'].unique())
    for i, m in enumerate(models):
        stat = finals_stats.get(m, {})
        mean_v = stat.get('mean', None)
        if mean_v is not None and not np.isnan(mean_v):
            plt.text(i, mean_v, f"{mean_v:.1f}", ha='center', va='bottom', fontweight='bold', color='red')
    path = os.path.join(out_dir, "final_rewards_boxplot.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    # save csv of values
    csv_path = os.path.join(out_dir, "final_rewards_values.csv")
    df.to_csv(csv_path, index=False)
    summary_rows = []
    for m, s in finals_stats.items():
        summary_rows.append({"model": m, "mean_final": s['mean'], "std_final": s['std'], "count": s['count'], "improvement_pct": s['improvement_pct']})
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_dir, "final_rewards_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    return path, csv_path, summary_csv


def main(experiments_root=None, out_dir=None, models=("baseline","h20")):
    experiments_root = experiments_root or os.path.join(RESULTS_DIR, "experiments")
    out_dir = out_dir or os.path.join(experiments_root, "aggregated_plots")
    data, finals = aggregate_per_model(experiments_root, models=models)
    finals_stats = compute_final_stats(finals)
    mean_plot = plot_mean_curves(data, finals_stats, out_dir)
    box = plot_final_boxplot(finals, finals_stats, out_dir)
    print("Saved outputs in:", out_dir)
    print("Mean-steps plot:", mean_plot)
    print("Boxplot + csvs:", box)
    return {"mean_plot": mean_plot, "boxplot": box, "finals_stats": finals_stats}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate experiment evals and plot mean-step curves + final boxplot")
    parser.add_argument("--experiments_root", default=None, help="path to experiments root (default uses RESULTS_DIR/experiments)")
    parser.add_argument("--out_dir", default=None, help="output dir for aggregated plots/CSVs")
    parser.add_argument("--models", default="baseline,h20", help="comma separated model keys to aggregate (default 'baseline,h20')")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    main(experiments_root=args.experiments_root, out_dir=args.out_dir, models=tuple(models))