import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime


DATASETS = ["ETTh1", "ETTm1", "Weather"]
HORIZONS = [96, 720]
HEADS = ["linear", "kan"]
SEEDS = [42, 43, 44]


def format_pm(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.6f} +/- {std_value:.6f}"


def build_setting(data: str, horizon: int, head: str, seed: int) -> str:
    if head == "kan":
        return f"WPMixer_{data}_head-kan_dec-True_sl96_pl{horizon}_dm512_bt32_wvdb2_tf5_df5_ptl16_stl8_sd{seed}"
    return f"WPMixer_{data}_head-linear_dec-True_sl96_pl{horizon}_dm512_bt32_wvdb2_tf5_df5_ptl16_stl8_sd{seed}"


def metrics_path(repo_root: str, data: str, horizon: int, head: str, seed: int) -> str:
    return os.path.join(repo_root, "logs", build_setting(data, horizon, head, seed), "metrics.json")


def candidate_metrics_paths(repo_root: str, data: str, horizon: int, head: str, seed: int):
    if head == "kan":
        return [
            os.path.join(repo_root, "logs", f"WPMixerKAN_{data}_dec-True_sl96_pl{horizon}_dm512_bt32_wvdb2_tf5_df5_ptl16_stl8_sd{seed}", "metrics.json"),
            os.path.join(repo_root, "logs", f"WPMixer_{data}_head-kan_dec-True_sl96_pl{horizon}_dm512_bt32_wvdb2_tf5_df5_ptl16_stl8_sd{seed}", "metrics.json"),
        ]
    return [metrics_path(repo_root, data, horizon, head, seed)]


def load_existing(repo_root: str, data: str, horizon: int, head: str, seed: int):
    for path in candidate_metrics_paths(repo_root, data, horizon, head, seed):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            test_results = metrics.get("test_results", {})
            setting = os.path.basename(os.path.dirname(path))
            return {
                "dataset": data,
                "horizon": horizon,
                "head": head,
                "seed": seed,
                "mse": test_results.get("mse"),
                "mae": test_results.get("mae"),
                "setting": setting,
            }
    return None


def run_one(repo_root: str, data: str, horizon: int, head: str, seed: int):
    existing = load_existing(repo_root, data, horizon, head, seed)
    if existing is not None:
        return existing

    command = [
        sys.executable,
        "run_LTF.py",
        "--model",
        "WPMixer",
        "--head_type",
        head,
        "--data",
        data,
        "--seq_len",
        "96",
        "--pred_len",
        str(horizon),
        "--d_model",
        "512",
        "--batch_size",
        "32",
        "--learning_rate",
        "0.0001",
        "--dropout",
        "0.05",
        "--embedding_dropout",
        "0.05",
        "--weight_decay",
        "0.0",
        "--patience",
        "10",
        "--train_epochs",
        "30",
        "--label_len",
        "48",
        "--wavelet",
        "db2",
        "--level",
        "1",
        "--patch_len",
        "16",
        "--stride",
        "8",
        "--tfactor",
        "5",
        "--dfactor",
        "5",
        "--loss",
        "smoothL1",
        "--lradj",
        "type3",
        "--seed",
        str(seed),
    ]

    subprocess.run(command, cwd=repo_root, check=True)
    return load_existing(repo_root, data, horizon, head, seed)


def write_outputs(output_dir: str, rows: list[dict]):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "verification_runs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "horizon", "head", "seed", "mse", "mae", "setting"])
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for head in HEADS:
                subset = [r for r in rows if r["dataset"] == dataset and r["horizon"] == horizon and r["head"] == head]
                if not subset:
                    continue
                summary.append({
                    "dataset": dataset,
                    "horizon": horizon,
                    "head": head,
                    "num_runs": len(subset),
                    "mse_mean": statistics.mean(r["mse"] for r in subset),
                    "mse_std": statistics.stdev(r["mse"] for r in subset) if len(subset) > 1 else 0.0,
                    "mae_mean": statistics.mean(r["mae"] for r in subset),
                    "mae_std": statistics.stdev(r["mae"] for r in subset) if len(subset) > 1 else 0.0,
                })

    summary_csv = os.path.join(output_dir, "verification_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "horizon", "head", "num_runs", "mse_mean", "mse_std", "mae_mean", "mae_std"])
        writer.writeheader()
        writer.writerows(summary)

    md_path = os.path.join(output_dir, "small_seed_verification_summary.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Small Multi-Seed Verification Summary\n\n")
        handle.write(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`\n")
        handle.write("- Scope: `ETTh1 / ETTm1 / Weather x {96, 720} x {linear, kan} x seeds {42, 43, 44}`\n")
        handle.write("- Shared setting: `seq_len=96, d_model=512, batch_size=32, wavelet=db2, level=1, patch_len=16, stride=8, tfactor=5, dfactor=5`\n\n")
        handle.write("| Dataset | Horizon | Head | Runs | MSE (mean +/- std) | MAE (mean +/- std) |\n")
        handle.write("| --- | ---: | --- | ---: | ---: | ---: |\n")
        for row in summary:
            handle.write(
                f"| {row['dataset']} | {row['horizon']} | {row['head']} | {row['num_runs']} | {format_pm(row['mse_mean'], row['mse_std'])} | {format_pm(row['mae_mean'], row['mae_std'])} |\n"
            )


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, "results", "small_seed_verification")
    rows = []
    total = len(DATASETS) * len(HORIZONS) * len(HEADS) * len(SEEDS)
    index = 0

    for dataset in DATASETS:
        for horizon in HORIZONS:
            for head in HEADS:
                for seed in SEEDS:
                    index += 1
                    print(f"[{index}/{total}] dataset={dataset} horizon={horizon} head={head} seed={seed}", flush=True)
                    rows.append(run_one(repo_root, dataset, horizon, head, seed))
                    write_outputs(output_dir, rows)

    write_outputs(output_dir, rows)
    print(f"Saved outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
