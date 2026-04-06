import argparse
import json
import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from exp.exp_main import Exp_Main
from utils.tools import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run branch-level and KAN-level interpretability analysis.")
    parser.add_argument("--setting_dir", type=str, required=True, help="Log directory containing config.json for a trained run")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint override")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save analysis outputs")
    parser.add_argument("--num_batches", type=int, default=8, help="Number of test batches to analyze")
    parser.add_argument("--max_points", type=int, default=2000, help="Maximum points used in scatter summaries")
    parser.add_argument("--use_gpu", type=str, default=None, help="Optional override: true/false")
    return parser.parse_args()


def str_to_bool(value):
    if value is None or isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def load_experiment_args(setting_dir, use_gpu_override=None):
    config_path = os.path.join(setting_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    if "head_type" not in config:
        config["head_type"] = "kan" if config.get("model") == "WPMixerKAN" else "linear"
    config.setdefault("match_head_params", False)
    config.setdefault("head_param_budget", None)
    config.setdefault("mlp_hidden_dim", None)
    config.setdefault("kan_grid_size", 5)
    config.setdefault("kan_spline_order", 3)
    config.setdefault("hybrid_linear_ratio", 0.5)
    config.setdefault("kan_reg_weight", 0.0)
    config.setdefault("kan_entropy_weight", 0.0)
    config.setdefault("use_multi_gpu", False)

    use_gpu = str_to_bool(use_gpu_override)
    if use_gpu is not None:
        config["use_gpu"] = use_gpu
    else:
        config["use_gpu"] = bool(config.get("use_gpu", True) and torch.cuda.is_available())

    return SimpleNamespace(**config)


def trim_prediction(prediction, args):
    return prediction[:, -args.pred_len :, -args.c_out :]


def reconstruct_prediction(wpmixer_core, approximation, details, pred_len, c_out):
    reconstructed = wpmixer_core.Decomposition_model.inv_transform(approximation, details)
    reconstructed = reconstructed.transpose(1, 2)
    reconstructed = reconstructed[:, -pred_len:, :]
    reconstructed = wpmixer_core.revin(reconstructed, "denorm")
    return reconstructed[:, :, -c_out:]


def compute_branch_features(branch, coeff_series):
    x = coeff_series.transpose(1, 2)
    x = branch.revin(x, "norm")
    x = x.transpose(1, 2)
    x_patch = branch.do_patching(x)
    x_patch = branch.patch_norm(x_patch)
    x_emb = branch.dropoutLayer(branch.patch_embedding_layer(x_patch))
    out = branch.mixer1(x_emb)
    out = out + branch.mixer2(out)
    out = branch.norm(out)
    batch_size, channel, patch_num, dim = out.shape
    return out.reshape(batch_size * channel, patch_num * dim)


def tensor_correlation(left, right):
    if left.size == 0 or right.size == 0:
        return 0.0
    if np.allclose(left.std(), 0.0) or np.allclose(right.std(), 0.0):
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def relpath_if_possible(path, start):
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return path


def save_bar_plot(path, labels, means, stds, title, ylabel):
    plt.figure(figsize=(8, 4.5))
    positions = np.arange(len(labels))
    plt.bar(positions, means, yerr=stds, capsize=4, color=["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"])
    plt.xticks(positions, labels, rotation=15)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_heatmap(path, matrix, row_labels, title):
    plt.figure(figsize=(10, 5.8))
    plt.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    colorbar = plt.colorbar(label="Mean basis activation")
    colorbar.ax.tick_params(labelsize=12)
    plt.yticks(np.arange(len(row_labels)), row_labels, fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Spline basis index", fontsize=13)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=420, bbox_inches="tight")
    plt.close()


def main():
    cli_args = parse_args()
    setting_dir = os.path.abspath(cli_args.setting_dir)
    setting_name = os.path.basename(setting_dir.rstrip(os.sep))
    repo_root = REPO_ROOT
    output_dir = cli_args.output_dir or os.path.join(repo_root, "results", f"interpretability_{setting_name}")
    os.makedirs(output_dir, exist_ok=True)

    args = load_experiment_args(setting_dir, cli_args.use_gpu)
    set_random_seed(getattr(args, "seed", 42))

    exp = Exp_Main(args)
    checkpoint_path = cli_args.checkpoint or os.path.join(repo_root, "checkpoints", setting_name, "checkpoint.pth")
    exp.model.load_state_dict(torch.load(checkpoint_path, map_location=exp.device))
    exp.model.eval()

    _, test_loader = exp._get_data(flag="test")
    model = exp._unwrap_model()
    wpmixer_core = model.wpmixerCore

    branch_labels = ["approx"] + [f"detail_{idx + 1}" for idx in range(len(wpmixer_core.resolutionBranch) - 1)]
    branch_energy_values = [[] for _ in branch_labels]
    branch_delta_values = [[] for _ in branch_labels]
    kan_basis_sums = [None for _ in branch_labels]
    kan_basis_counts = [0 for _ in branch_labels]
    kan_activation_strengths = [[] for _ in branch_labels]
    kan_error_values = [[] for _ in branch_labels]
    full_mse_values = []

    analyzed_batches = 0
    with torch.no_grad():
        for batch_x, batch_y, *rest in test_loader:
            batch_x = batch_x.to(dtype=torch.float, device=exp.device)
            target = batch_y.to(dtype=torch.float, device=exp.device)
            target = target[:, -args.pred_len :, -args.c_out :]

            full_pred, intermediates = model.forward_with_intermediates(batch_x)
            full_pred = trim_prediction(full_pred, args)
            sample_mse = ((full_pred - target) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
            full_mse_values.extend(sample_mse.tolist())

            coeff_outputs = [intermediates["approximation_output"]] + list(intermediates["detail_outputs"])
            coeff_inputs = [intermediates["approximation_input"]] + list(intermediates["detail_inputs"])
            coeff_energy = np.array([tensor.pow(2).mean().item() for tensor in coeff_outputs], dtype=np.float64)
            coeff_energy = coeff_energy / max(coeff_energy.sum(), 1e-12)
            for branch_idx, energy_share in enumerate(coeff_energy):
                branch_energy_values[branch_idx].append(float(energy_share))

            for branch_idx in range(len(branch_labels)):
                kept_approx = intermediates["approximation_output"]
                kept_details = list(intermediates["detail_outputs"])
                if branch_idx == 0:
                    kept_approx = torch.zeros_like(kept_approx)
                else:
                    kept_details[branch_idx - 1] = torch.zeros_like(kept_details[branch_idx - 1])

                loo_pred = reconstruct_prediction(wpmixer_core, kept_approx, kept_details, args.pred_len, args.c_out)
                loo_mse = ((loo_pred - target) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
                branch_delta_values[branch_idx].extend((loo_mse - sample_mse).tolist())

                branch = wpmixer_core.resolutionBranch[branch_idx]
                if branch.head_type != "kan":
                    continue

                flattened = compute_branch_features(branch, coeff_inputs[branch_idx])
                head_module = branch.head
                adapted = head_module.dropout(flattened)
                if head_module.input_adapter is not None:
                    adapted = head_module.input_adapter(adapted)

                kan_layer = head_module.head.layers[0]
                bases = kan_layer.b_splines(adapted)
                basis_mean = bases.mean(dim=(0, 1)).detach().cpu().numpy()
                if kan_basis_sums[branch_idx] is None:
                    kan_basis_sums[branch_idx] = basis_mean.copy()
                else:
                    kan_basis_sums[branch_idx] += basis_mean
                kan_basis_counts[branch_idx] += 1

                activation_strength = bases.abs().mean(dim=(1, 2)).detach().cpu().numpy()
                repeated_errors = np.repeat(sample_mse, branch.channel)
                if activation_strength.shape[0] == repeated_errors.shape[0]:
                    kan_activation_strengths[branch_idx].extend(activation_strength.tolist())
                    kan_error_values[branch_idx].extend(repeated_errors.tolist())

            analyzed_batches += 1
            if analyzed_batches >= cli_args.num_batches:
                break

    branch_energy_mean = [float(np.mean(values)) if values else 0.0 for values in branch_energy_values]
    branch_energy_std = [float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 for values in branch_energy_values]
    branch_delta_mean = [float(np.mean(values)) if values else 0.0 for values in branch_delta_values]
    branch_delta_std = [float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 for values in branch_delta_values]

    kan_basis_rows = []
    kan_basis_labels = []
    kan_activation_error_corr = {}
    for branch_idx, label in enumerate(branch_labels):
        if kan_basis_counts[branch_idx] > 0:
            basis_sum = kan_basis_sums[branch_idx]
            if basis_sum is None:
                continue
            kan_basis_rows.append(basis_sum / kan_basis_counts[branch_idx])
            kan_basis_labels.append(label)
            strengths = np.array(kan_activation_strengths[branch_idx][: cli_args.max_points], dtype=np.float64)
            errors = np.array(kan_error_values[branch_idx][: cli_args.max_points], dtype=np.float64)
            kan_activation_error_corr[label] = tensor_correlation(strengths, errors)

    summary = {
        "setting_name": setting_name,
        "checkpoint_path": relpath_if_possible(checkpoint_path, repo_root),
        "analyzed_batches": analyzed_batches,
        "full_prediction_mse_mean": float(np.mean(full_mse_values)) if full_mse_values else None,
        "full_prediction_mse_std": float(np.std(full_mse_values, ddof=1)) if len(full_mse_values) > 1 else 0.0,
        "branch_energy_share": {
            label: {"mean": branch_energy_mean[idx], "std": branch_energy_std[idx]}
            for idx, label in enumerate(branch_labels)
        },
        "leave_one_out_delta_mse": {
            label: {"mean": branch_delta_mean[idx], "std": branch_delta_std[idx]}
            for idx, label in enumerate(branch_labels)
        },
        "kan_activation_error_correlation": kan_activation_error_corr,
    }

    summary_path = os.path.join(output_dir, "interpretability_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    save_bar_plot(
        os.path.join(output_dir, "branch_energy_share.png"),
        branch_labels,
        branch_energy_mean,
        branch_energy_std,
        "Resolution-Branch Energy Share",
        "Energy share",
    )
    save_bar_plot(
        os.path.join(output_dir, "leave_one_out_delta_mse.png"),
        branch_labels,
        branch_delta_mean,
        branch_delta_std,
        "Leave-One-Branch-Out Error Increase",
        "Delta MSE",
    )

    if kan_basis_rows:
        save_heatmap(
            os.path.join(output_dir, "kan_basis_activation_heatmap.png"),
            np.vstack(kan_basis_rows),
            kan_basis_labels,
            "KAN Basis Activation by Resolution Branch",
        )

    markdown_path = os.path.join(output_dir, "interpretability_summary.md")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("# Interpretability Summary\n\n")
        handle.write(f"- Setting: `{setting_name}`\n")
        handle.write(f"- Checkpoint: `{relpath_if_possible(checkpoint_path, repo_root)}`\n")
        handle.write(f"- Analyzed batches: {analyzed_batches}\n")
        if summary["full_prediction_mse_mean"] is not None:
            handle.write(
                f"- Full-prediction sample MSE: {summary['full_prediction_mse_mean']:.6f} +/- {summary['full_prediction_mse_std']:.6f}\n\n"
            )

        handle.write("## Branch-Level Results\n\n")
        handle.write("| Branch | Energy share | Leave-one-out delta MSE |\n")
        handle.write("| --- | ---: | ---: |\n")
        for idx, label in enumerate(branch_labels):
            handle.write(
                f"| {label} | {branch_energy_mean[idx]:.6f} +/- {branch_energy_std[idx]:.6f} | {branch_delta_mean[idx]:.6f} +/- {branch_delta_std[idx]:.6f} |\n"
            )

        if kan_activation_error_corr:
            handle.write("\n## KAN-Level Results\n\n")
            handle.write("| Branch | Activation-error correlation |\n")
            handle.write("| --- | ---: |\n")
            for label, value in kan_activation_error_corr.items():
                handle.write(f"| {label} | {value:.6f} |\n")

    print(f"Saved interpretability summary to {summary_path}")
    print(f"Saved markdown report to {markdown_path}")
    print(f"Saved figures under {output_dir}")


if __name__ == "__main__":
    main()
