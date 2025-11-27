#!/usr/bin/env python3
import os
import argparse
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient


def load_metric_history(client: MlflowClient, run_id: str, key: str) -> Tuple[List[int], List[float]]:
    try:
        hist = client.get_metric_history(run_id, key)
        points = sorted([(m.step, m.value) for m in hist], key=lambda x: x[0])
        if not points:
            return [], []
        steps, vals = zip(*points)
        return list(steps), list(vals)
    except Exception:
        return [], []


def discover_metric_keys(mlruns_dir: str, run) -> List[str]:
    keys = list(run.data.metrics.keys()) if run and run.data and run.data.metrics else []
    if keys:
        return keys

    # fallback: try to find metrics files on disk
    run_dir = None
    if run and run.info and run.info.experiment_id:
        run_dir = os.path.join(mlruns_dir, str(run.info.experiment_id), run.info.run_id)
    else:
        for exp in os.listdir(mlruns_dir):
            candidate = os.path.join(mlruns_dir, exp, run.info.run_id if run and run.info else '')
            if os.path.isdir(candidate):
                run_dir = candidate
                break

    if run_dir:
        metrics_folder = os.path.join(run_dir, "metrics")
        if os.path.isdir(metrics_folder):
            return sorted(os.listdir(metrics_folder))

    return []


def select_reward_keys(keys: List[str]) -> List[str]:
    reward_keys = []
    for k in keys:
        kl = k.lower()
        if "reward" in kl or "return" in kl or "episode" in kl or "rew" in kl:
            reward_keys.append(k)
    return reward_keys


def plot_metrics(run_id: str, mlruns_dir: str, out: str, show: bool = False) -> None:
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{os.path.abspath(mlruns_dir)}"
    client = MlflowClient()
    run = client.get_run(run_id)

    keys = discover_metric_keys(mlruns_dir, run)
    if not keys:
        print("Nenhuma métrica encontrada para o run:", run_id)
        return

    metrics_hist: Dict[str, Tuple[List[int], List[float]]] = {}
    for k in keys:
        steps, vals = load_metric_history(client, run_id, k)
        if steps:
            metrics_hist[k] = (steps, vals)

    if not metrics_hist:
        print("Métricas sem histórico encontrado.")
        return

    reward_keys = select_reward_keys(list(metrics_hist.keys()))
    non_reward_keys = [k for k in metrics_hist.keys() if k not in reward_keys]

    plots = non_reward_keys[:]
    if reward_keys:
        plots.append("__rewards__")

    n_plots = len(plots)
    cols = 1
    rows = n_plots

    fig_height = max(3 * rows, 6)
    fig, axes = plt.subplots(rows, cols, figsize=(10, fig_height), squeeze=False)

    for idx, key in enumerate(plots):
        ax = axes[idx][0]
        if key == "__rewards__":
            for rk in reward_keys:
                steps, vals = metrics_hist[rk]
                ax.plot(steps, vals, marker="o", label=rk)
            ax.set_title("Recompensas / retornos")
            ax.set_xlabel("step / epoch")
            ax.set_ylabel("value")
            ax.legend(loc="best", fontsize="small")
            ax.grid(True)
        else:
            steps, vals = metrics_hist[key]
            ax.plot(steps, vals, marker="o", label=key)
            ax.set_title(key)
            ax.set_xlabel("step / epoch")
            ax.set_ylabel("value")
            ax.grid(True)

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()
    print("Saved plot to", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlruns-dir", default="./mlruns", help="Path to mlruns directory")
    parser.add_argument("--run-id", required=False, help="MLflow run id to plot")
    parser.add_argument("--out", default="mlflow_run_metrics.png", help="Output PNG file")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    args = parser.parse_args()

    run_id = args.run_id
    if not run_id:
        # default to the run in mlruns/847030684811318288/
        candidate_exp = os.path.join(os.path.abspath(args.mlruns_dir), "847030684811318288")
        if os.path.isdir(candidate_exp):
            # pick the first run folder inside
            runs = [d for d in os.listdir(candidate_exp) if os.path.isdir(os.path.join(candidate_exp, d))]
            if runs:
                run_id = runs[0]

    if not run_id:
        print("Nenhum run_id fornecido e não foi possível inferir um run padrão.")
        return

    plot_metrics(run_id, args.mlruns_dir, args.out, args.show)


if __name__ == "__main__":
    main()
