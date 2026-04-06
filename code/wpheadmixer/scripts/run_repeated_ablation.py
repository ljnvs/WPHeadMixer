import argparse
import csv
import json
import os
import statistics
import subprocess
import sys


def build_setting(args, seed, head_type):
    return (
        f"{args.model}_{args.data}_head-{head_type}_dec-{not args.no_decomposition}"
        f"_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_bt{args.batch_size}"
        f"_wv{args.wavelet}_tf{args.tfactor}_df{args.dfactor}_ptl{args.patch_len}"
        f"_stl{args.stride}_sd{seed}"
    )


def run_single_experiment(repo_root, args, seed, head_type):
    command = [
        sys.executable,
        "run_LTF.py",
        "--model",
        args.model,
        "--data",
        args.data,
        "--head_type",
        head_type,
        "--seq_len",
        str(args.seq_len),
        "--pred_len",
        str(args.pred_len),
        "--d_model",
        str(args.d_model),
        "--tfactor",
        str(args.tfactor),
        "--dfactor",
        str(args.dfactor),
        "--wavelet",
        args.wavelet,
        "--level",
        str(args.level),
        "--patch_len",
        str(args.patch_len),
        "--stride",
        str(args.stride),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--dropout",
        str(args.dropout),
        "--embedding_dropout",
        str(args.embedding_dropout),
        "--weight_decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--train_epochs",
        str(args.train_epochs),
        "--seed",
        str(seed),
        "--lradj",
        args.lradj,
        "--loss",
        args.loss,
        "--kan_grid_size",
        str(args.kan_grid_size),
        "--kan_spline_order",
        str(args.kan_spline_order),
        "--hybrid_linear_ratio",
        str(args.hybrid_linear_ratio),
        "--kan_reg_weight",
        str(args.kan_reg_weight),
        "--kan_entropy_weight",
        str(args.kan_entropy_weight),
    ]

    if args.match_head_params:
        command.append("--match_head_params")
    if args.no_decomposition:
        command.append("--no_decomposition")
    if args.head_param_budget is not None:
        command.extend(["--head_param_budget", str(args.head_param_budget)])
    if args.mlp_hidden_dim is not None:
        command.extend(["--mlp_hidden_dim", str(args.mlp_hidden_dim)])
    if args.use_amp:
        command.append("--use_amp")
    if args.cpu:
        command.extend(["--use_gpu", "False"])
    if args.extra_args:
        command.extend(args.extra_args)

    subprocess.run(command, cwd=repo_root, check=True)

    setting = build_setting(args, seed, head_type)
    metrics_path = os.path.join(repo_root, "logs", setting, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    test_results = metrics.get("test_results", {})
    training_summary = metrics.get("performance_summary", {}).get("training", {})
    model_info = metrics.get("model_info", {})
    head_summary = model_info.get("head_summary", [])
    return {
        "setting": setting,
        "seed": seed,
        "head_type": head_type,
        "mse": test_results.get("mse"),
        "mae": test_results.get("mae"),
        "rmse": test_results.get("rmse"),
        "best_validation_loss": training_summary.get("best_validation_loss"),
        "best_test_loss_at_best_val": training_summary.get("best_test_loss_at_best_val"),
        "total_params": model_info.get("total_params"),
        "head_params_total": sum(branch.get("head_parameters", 0) for branch in head_summary),
    }


def mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_pm(mean_value, std_value):
    return f"{mean_value:.6f} +/- {std_value:.6f}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run repeated head ablations and summarize mean/std.")
    parser.add_argument("--repo_root", type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--model", type=str, default="WPMixer")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--heads", nargs="+", default=["linear", "mlp", "kan"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--tfactor", type=int, default=5)
    parser.add_argument("--dfactor", type=int, default=5)
    parser.add_argument("--wavelet", type=str, default="db2")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--embedding_dropout", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--lradj", type=str, default="type3")
    parser.add_argument("--loss", type=str, default="smoothL1")
    parser.add_argument("--kan_grid_size", type=int, default=5)
    parser.add_argument("--kan_spline_order", type=int, default=3)
    parser.add_argument("--hybrid_linear_ratio", type=float, default=0.5)
    parser.add_argument("--kan_reg_weight", type=float, default=0.0)
    parser.add_argument("--kan_entropy_weight", type=float, default=0.0)
    parser.add_argument("--match_head_params", action="store_true", default=False)
    parser.add_argument("--head_param_budget", type=int, default=None)
    parser.add_argument("--mlp_hidden_dim", type=int, default=None)
    parser.add_argument("--no_decomposition", action="store_true", default=False)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False, help="force CPU execution for each run")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.abspath(args.repo_root)
    output_dir = args.output_dir or os.path.join(repo_root, "results", f"ablation_{args.data}_pl{args.pred_len}")
    os.makedirs(output_dir, exist_ok=True)

    all_runs = []
    for head_type in args.heads:
        for seed in args.seeds:
            result = run_single_experiment(repo_root, args, seed, head_type)
            all_runs.append(result)
            print(f"[done] head={head_type} seed={seed} mse={result['mse']:.6f} mae={result['mae']:.6f}")

    summary_rows = []
    for head_type in args.heads:
        rows = [row for row in all_runs if row["head_type"] == head_type]
        mse_mean, mse_std = mean_std([row["mse"] for row in rows])
        mae_mean, mae_std = mean_std([row["mae"] for row in rows])
        summary_rows.append({
            "head_type": head_type,
            "num_runs": len(rows),
            "mse_mean": mse_mean,
            "mse_std": mse_std,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "total_params_mean": statistics.mean([row["total_params"] for row in rows]),
            "head_params_mean": statistics.mean([row["head_params_total"] for row in rows]),
        })

    write_csv(
        os.path.join(output_dir, "ablation_runs.csv"),
        all_runs,
        ["setting", "seed", "head_type", "mse", "mae", "rmse", "best_validation_loss", "best_test_loss_at_best_val", "total_params", "head_params_total"],
    )
    write_csv(
        os.path.join(output_dir, "ablation_summary.csv"),
        summary_rows,
        ["head_type", "num_runs", "mse_mean", "mse_std", "mae_mean", "mae_std", "total_params_mean", "head_params_mean"],
    )

    report_path = os.path.join(output_dir, "ablation_summary.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Head Ablation Summary\n\n")
        handle.write("| Head | Runs | MSE (mean +/- std) | MAE (mean +/- std) | Total Params | Head Params |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summary_rows:
            handle.write(
                f"| {row['head_type']} | {row['num_runs']} | {format_pm(row['mse_mean'], row['mse_std'])} | {format_pm(row['mae_mean'], row['mae_std'])} | {row['total_params_mean']:.0f} | {row['head_params_mean']:.0f} |\n"
            )

    print(f"Saved detailed runs to {os.path.join(output_dir, 'ablation_runs.csv')}")
    print(f"Saved summary to {os.path.join(output_dir, 'ablation_summary.csv')}")
    print(f"Saved markdown report to {report_path}")


if __name__ == "__main__":
    main()
