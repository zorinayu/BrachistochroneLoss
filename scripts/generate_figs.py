import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_both(fig: plt.Figure, filename: str, out_dirs: list[str]) -> None:
    for d in out_dirs:
        ensure_dir(d)
        target = os.path.join(d, filename)
        fig.savefig(target, bbox_inches="tight")


def parse_results(outputs_root: str) -> pd.DataFrame:
    """Parse Brachistochrone-only lines from known dataset result files."""
    dataset_to_file = {
        "adult": os.path.join(outputs_root, "adult", "adult_results.txt"),
        "cifar10": os.path.join(outputs_root, "cifar10", "cifar10_results.txt"),
        "imdb": os.path.join(outputs_root, "imdb", "imdb_results.txt"),
        "wine": os.path.join(outputs_root, "wine", "wine_results.txt"),
    }
    records = []
    for ds, fp in dataset_to_file.items():
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Expect line like: "Brachistochrone     0.6900  0.5634  0.5  1 epoch"
                if line.strip().startswith("Brachistochrone ") or line.strip() == "Brachistochrone":
                    parts = line.split()
                    # Be tolerant of spacing; try to extract floats in order
                    floats = [p for p in parts if _is_float(p)]
                    if len(floats) >= 3:
                        acc, f1, time_s = map(float, floats[:3])
                        records.append({
                            "dataset": ds,
                            "acc": acc,
                            "f1": f1,
                            "time_s": time_s,
                        })
                    break
    return pd.DataFrame.from_records(records)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def plot_results_summary(df: pd.DataFrame, out_dirs: list[str]) -> None:
    if df.empty:
        return
    # Sort datasets in a consistent order
    order = ["adult", "cifar10", "imdb", "wine"]
    df["dataset"] = pd.Categorical(df["dataset"], categories=order, ordered=True)
    df = df.sort_values("dataset")

    x = np.arange(len(df))
    width = 0.28

    fig, ax1 = plt.subplots(figsize=(7.0, 3.2))
    b1 = ax1.bar(x - width/2, df["acc"].values, width, label="Accuracy", color="#2c7fb8")
    b2 = ax1.bar(x + width/2, df["f1"].values, width, label="F1", color="#f03b20")
    ax1.set_ylabel("Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.upper() for s in df["dataset"].tolist()])
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(ncols=2, frameon=False)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(x, df["time_s"].values, marker="o", color="#636363", label="Time (s)")
    ax2.set_ylabel("Time (s)")
    ax2.legend(loc="upper right", frameon=False)

    plt.title("Brachistochrone Results Summary")
    save_both(fig, "results_summary.pdf", out_dirs)
    plt.close(fig)


def plot_energy_curve(out_dirs: list[str], T: int = 8, alpha: float = 1.0, eps: float = 1e-4) -> None:
    # Simulated monotone energy decay and corresponding speed
    E = np.exp(-np.linspace(0, 2.2, T)) + 0.02
    v = np.sqrt(2.0 * alpha * E + eps)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(np.arange(T), E, marker="o", color="#1f78b4", label="E(h)")
    ax.plot(np.arange(T), v, marker="s", color="#33a02c", label="v(h)")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Value")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(frameon=False)
    ax.set_title("Energy and Speed Across Stages")
    save_both(fig, "energy_curve.pdf", out_dirs)
    plt.close(fig)


def plot_loss_decomposition(out_dirs: list[str], B: int = 64, alpha: float = 1.0, eps: float = 1e-6) -> None:
    # Sample synthetic batch energies for two stages and compute L_path, L_mono
    rng = np.random.default_rng(1337)
    E0 = np.abs(rng.normal(1.0, 0.2, size=B)) + 0.1
    E1 = np.clip(E0 - np.abs(rng.normal(0.25, 0.15, size=B)), a_min=eps, a_max=None)
    v0 = np.sqrt(2.0 * alpha * E0 + eps)
    L_path = np.mean(np.abs(E1 - E0) / (v0 + eps))
    L_mono = np.mean(np.clip(E1 - E0, a_min=0.0, a_max=None))

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar(["L_path", "L_mono"], [L_path, L_mono], color=["#3182bd", "#feb24c"]) 
    ax.set_ylabel("Value")
    ax.set_title("Loss Decomposition (Synthetic Batch)")
    for i, v in enumerate([L_path, L_mono]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    save_both(fig, "loss_decomp.pdf", out_dirs)
    plt.close(fig)


def plot_trajectory_1d(out_dirs: list[str], steps: int = 6, dt: float = 0.2, alpha: float = 1.0, eps: float = 1e-4):
    # E(h) = 0.5 (h - h*)^2, h* = 0
    def acc(h, hdot):
        # ddot{h} = -(1+hdot^2) * alpha (h - h*) / ((h - h*)^2 + eps)
        return -(1.0 + hdot**2) * (alpha * (h)) / (h * h + eps)

    h, hdot = 1.0, 0.0
    traj = [h]
    for _ in range(steps):
        hdd = acc(h, hdot)
        hdot = hdot + dt * hdd
        h = h + dt * hdot
        traj.append(h)

    # Gradient flow baseline: h_{k+1} = h_k - dt * gradE = h_k - dt * h_k
    h_gd = 1.0
    traj_gd = [h_gd]
    for _ in range(steps):
        h_gd = h_gd - dt * h_gd
        traj_gd.append(h_gd)

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.plot(range(steps + 1), traj, marker="o", color="#e31a1c", label="Brachistochrone")
    ax.plot(range(steps + 1), traj_gd, marker="s", color="#1f78b4", label="Gradient Flow")
    ax.set_xlabel("Step")
    ax.set_ylabel("h")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(frameon=False)
    ax.set_title("1D Dynamics: Accelerate-then-Refine")
    save_both(fig, "trajectory_1d.pdf", out_dirs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_root", type=str, default=os.path.join(".", "outputs"))
    parser.add_argument("--paper_figs", type=str, required=True)
    args = parser.parse_args()

    # Where to save: project outputs (pdfs) and paper figs
    out_dirs = [args.outputs_root, args.paper_figs]

    # Results summary from text reports
    df = parse_results(args.outputs_root)
    plot_results_summary(df, out_dirs)

    # Synthetic illustrative figures
    plot_energy_curve(out_dirs)
    plot_loss_decomposition(out_dirs)
    plot_trajectory_1d(out_dirs)


if __name__ == "__main__":
    main()


