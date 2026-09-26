import argparse
import hashlib
import json

import numpy as np
from scipy.stats import pearsonr, spearmanr

from fogen.execution_graph import (
    additive_effect,
    compile_graph,
    fit_composition_scale,
    single_layer_effects,
)


def split_rows(rows, seed):
    calibration, test = [], []
    for row in rows:
        if sum(row["bits"]) < 2:
            continue
        key = f"{seed}:{''.join(map(str, row['bits']))}".encode()
        bucket = int(hashlib.sha256(key).hexdigest(), 16) % 2
        (calibration if bucket == 0 else test).append(row)
    return calibration, test


def raw_predictions(rows, effects):
    return np.asarray([additive_effect(row["bits"], effects) for row in rows])


def actual_effects(rows, field, baseline):
    return np.asarray([max(0.0, row[field] - baseline) for row in rows])


def prediction_metrics(predicted, actual):
    slope = fit_composition_scale(predicted, actual)
    calibrated = slope * predicted
    residual = actual - calibrated
    return slope, {
        "spearman": float(spearmanr(calibrated, actual).statistic),
        "pearson": float(pearsonr(calibrated, actual).statistic),
        "relative_rmse": float(
            np.sqrt(np.mean(residual**2)) /
            max(np.sqrt(np.mean(actual**2)), 1e-30)
        ),
    }


def latency_savings(rows):
    return np.ones(len(rows[0]["bits"]), dtype=float)


def compiler_trials(rows, effects, savings, scale):
    candidates = [row for row in rows if sum(row["bits"]) >= 1]
    for row in candidates:
        row["predicted_effect"] = additive_effect(row["bits"], effects, scale)
        row["predicted_latency"] = -float(np.dot(row["bits"], savings))
    predicted_values = np.asarray([row["predicted_effect"] for row in candidates])
    trials = []
    for quantile in (0.25, 0.5, 0.75):
        budget = float(np.quantile(predicted_values, quantile))
        feasible = [row for row in candidates if row["predicted_effect"] <= budget]
        selected = min(feasible, key=lambda row: row["predicted_latency"])
        actual_feasible = [
            row for row in candidates
            if row["actual_effect"] <= budget
        ]
        oracle = (max(
            actual_feasible,
            key=lambda row: (row["parallel_layers"], -row["latency_seconds"]))
            if actual_feasible else None)
        trials.append({
            "quantile": quantile,
            "budget": budget,
            "selected_bits": selected["bits"],
            "selected_parallel_layers": selected["parallel_layers"],
            "selected_predicted_effect": selected["predicted_effect"],
            "selected_actual_effect": selected["actual_effect"],
            "budget_satisfied": selected["actual_effect"] <= budget,
            "selected_latency": selected["latency_seconds"],
            "oracle_parallel_layers": oracle["parallel_layers"] if oracle else None,
            "parallelism_regret": (
                oracle["parallel_layers"] - selected["parallel_layers"]
                if oracle else None
            ),
        })
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_graph_results")
    parser.add_argument("--calibration_graph_results")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output")
    args = parser.parse_args()
    with open(args.test_graph_results) as handle:
        test_data = json.load(handle)
    calibration_path = args.calibration_graph_results or args.test_graph_results
    with open(calibration_path) as handle:
        calibration_data = json.load(handle)
    calibration_rows, _ = split_rows(calibration_data["rows"], args.seed)
    _, test_rows = split_rows(test_data["rows"], args.seed)
    calibration_effects = single_layer_effects(
        calibration_data["rows"], "symmetric_kl")
    test_effects = single_layer_effects(test_data["rows"], "symmetric_kl")
    calibration_baseline = next(
        row["symmetric_kl"] for row in calibration_data["rows"]
        if sum(row["bits"]) == 0)
    test_baseline = next(
        row["symmetric_kl"] for row in test_data["rows"]
        if sum(row["bits"]) == 0)
    calibration_raw = raw_predictions(calibration_rows, calibration_effects)
    calibration_actual = actual_effects(
        calibration_rows, "symmetric_kl", calibration_baseline)
    scale, _ = prediction_metrics(calibration_raw, calibration_actual)
    test_raw = raw_predictions(test_rows, test_effects)
    test_actual = actual_effects(test_rows, "symmetric_kl", test_baseline)
    calibrated = scale * test_raw
    metrics = {
        "spearman": float(spearmanr(calibrated, test_actual).statistic),
        "pearson": float(pearsonr(calibrated, test_actual).statistic),
        "relative_rmse": float(
            np.sqrt(np.mean((test_actual - calibrated) ** 2)) /
            max(np.sqrt(np.mean(test_actual**2)), 1e-30)
        ),
    }
    for row in test_rows:
        row["actual_effect"] = max(
            0.0, row["symmetric_kl"] - test_baseline)
    savings = latency_savings(test_data["rows"])
    budgets = [float(np.quantile(calibrated, q)) for q in (0.25, 0.5, 0.75)]
    result = {
        "calibration_path": calibration_path,
        "test_path": args.test_graph_results,
        "calibration_graphs": len(calibration_rows),
        "test_graphs": len(test_rows),
        "composition_scale": scale,
        "test_metrics": metrics,
        "compiler_trials": compiler_trials(test_rows, test_effects, savings, scale),
        "global_compiled_graphs": [
            compile_graph(test_effects, savings, budget, scale)
            for budget in budgets
        ],
    }
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
