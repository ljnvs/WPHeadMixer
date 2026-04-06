import csv
import json
import os
import subprocess
import sys
from datetime import datetime


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange"]
HORIZONS = [96, 192, 336, 720]


def build_setting(data: str, pred_len: int) -> str:
    return (
        f"WPMixer_{data}_head-linear_dec-True_sl96_pl{pred_len}_dm512_bt32"
        f"_wvdb2_tf5_df5_ptl16_stl8_sd42"
    )


def format_relative_log_dir(setting: str) -> str:
    return f"logs/{setting}"


def run_one(repo_root: str, data: str, pred_len: int) -> dict:
    setting = build_setting(data, pred_len)
    metrics_path = os.path.join(repo_root, "logs", setting, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        test_results = metrics.get("test_results", {})
        return {
            "dataset": data,
            "horizon": pred_len,
            "mse": test_results.get("mse"),
            "mae": test_results.get("mae"),
            "setting": setting,
        }

    command = [
        sys.executable,
        "run_LTF.py",
        "--model",
        "WPMixer",
        "--head_type",
        "linear",
        "--data",
        data,
        "--seq_len",
        "96",
        "--pred_len",
        str(pred_len),
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
        "42",
    ]
    subprocess.run(command, cwd=repo_root, check=True)

    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    test_results = metrics.get("test_results", {})
    return {
        "dataset": data,
        "horizon": pred_len,
        "mse": test_results.get("mse"),
        "mae": test_results.get("mae"),
        "setting": setting,
    }


def write_summary(output_dir: str, rows: list[dict]) -> None:
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "linear_main_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "horizon", "mse", "mae", "setting"])
        writer.writeheader()
        writer.writerows(rows)

    grouped = {dataset: {} for dataset in DATASETS}
    for row in rows:
        grouped[row["dataset"]][row["horizon"]] = row

    md_path = os.path.join(output_dir, "linear_main_results.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Linear Main Results Summary\n\n")
        handle.write(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`\n")
        handle.write("- Model: `WPMixer`\n")
        handle.write("- Head: `linear`\n")
        handle.write("- Shared setting: `seq_len=96, d_model=512, batch_size=32, wavelet=db2, level=1, patch_len=16, stride=8, tfactor=5, dfactor=5, seed=42`\n\n")
        handle.write("## Full Benchmark Table\n\n")
        handle.write("| Dataset | 96 | 192 | 336 | 720 |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        for dataset in DATASETS:
            values = []
            for horizon in HORIZONS:
                row = grouped[dataset].get(horizon)
                values.append("-" if row is None else f"{row['mse']:.6f} / {row['mae']:.6f}")
            handle.write(f"| {dataset} | {values[0]} | {values[1]} | {values[2]} | {values[3]} |\n")

        handle.write("\nValues are reported as `MSE / MAE`.\n\n")
        handle.write("## Per-Dataset Result Trace\n\n")
        for index, dataset in enumerate(DATASETS, start=1):
            handle.write(f"### 2.{index} {dataset}\n\n")
            handle.write("Associated log directories:\n")
            for horizon in HORIZONS:
                row = grouped[dataset].get(horizon)
                if row is not None:
                    handle.write(f"- `{format_relative_log_dir(row['setting'])}`\n")
            handle.write("\n")
            handle.write("| Dataset | Horizon | Linear MSE | Linear MAE |\n")
            handle.write("| --- | ---: | ---: | ---: |\n")
            for horizon in HORIZONS:
                row = grouped[dataset].get(horizon)
                if row is not None:
                    handle.write(f"| {dataset} | {horizon} | {row['mse']:.6f} | {row['mae']:.6f} |\n")
            handle.write("\n")


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, "results", "linear_main_body")
    rows = []
    total = len(DATASETS) * len(HORIZONS)
    index = 0

    for dataset in DATASETS:
        for horizon in HORIZONS:
            index += 1
            print(f"[{index}/{total}] Running {dataset} horizon={horizon}", flush=True)
            rows.append(run_one(repo_root, dataset, horizon))
            write_summary(output_dir, rows)

    write_summary(output_dir, rows)
    print(f"Saved summary to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
