"""Train a grouped-validation Random Forest and export auditable evaluation files."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                             precision_score, recall_score, roc_auc_score)

from maintenance.data import (FEATURES, add_training_target, last_test_observations,
                              read_engines, split_by_engine)

TARGET = "failure_within_horizon"
ROOT = Path(__file__).resolve().parents[1]


def choose_threshold(y_true, scores) -> float:
    """Maximize validation F2; ties favor the larger threshold (fewer alerts)."""
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Threshold selection requires both target classes.")
    candidates = np.round(np.arange(0.05, 0.96, 0.05), 2)
    return float(max(candidates, key=lambda t: (
        fbeta_score(y_true, np.asarray(scores) >= t, beta=2, zero_division=0), t)))


def metrics(y_true, scores, threshold: float) -> dict:
    prediction = np.asarray(scores) >= threshold
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": len(y_true), "positive_count": int(np.sum(y_true)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f2": float(fbeta_score(y_true, prediction, beta=2, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, scores)) if both_classes else None,
        "roc_auc": float(roc_auc_score(y_true, scores)) if both_classes else None,
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_true, prediction, labels=[0, 1]).ravel().tolist(),
    }


def make_model(seed: int, trees: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=trees, max_depth=12, min_samples_leaf=3,
                                  class_weight="balanced_subsample", random_state=seed, n_jobs=1)


def run(data_dir: Path, output: Path, seed: int = 42, horizon: int = 30,
        trees: int = 200) -> dict:
    full = add_training_target(read_engines(data_dir / "PM_train.txt"), horizon)
    fit, validation = split_by_engine(full, seed)
    for frame in [fit, validation]:
        if set(frame[TARGET].unique()) != {0, 1}:
            raise ValueError("Training and validation must each contain both target classes.")
    model = make_model(seed, trees).fit(fit[FEATURES], fit[TARGET])
    validation_scores = model.predict_proba(validation[FEATURES])[:, 1]
    threshold = choose_threshold(validation[TARGET], validation_scores)

    # Freeze the threshold before reading the external test set. Do not tune on test.
    model.fit(full[FEATURES], full[TARGET])
    test = last_test_observations(read_engines(data_dir / "PM_test.txt"),
                                  data_dir / "PM_truth.txt", horizon)
    scores = model.predict_proba(test[FEATURES])[:, 1]
    dummy = DummyClassifier(strategy="prior").fit(full[FEATURES], full[TARGET])
    dummy_scores = dummy.predict_proba(test[FEATURES])[:, 1]
    report = {
        "seed": seed, "horizon_cycles": horizon, "threshold": threshold,
        "threshold_selection": "Validation F2, 0.05..0.95 in steps of 0.05; largest threshold wins ties",
        "validation_unit": "All cycles of 25% held-out training engines; descriptive selection score",
        "test_unit": "One last observation per external test engine",
        "features": FEATURES,
        "fit_engine_ids": sorted(fit.engine_id.unique().tolist()),
        "validation_engine_ids": sorted(validation.engine_id.unique().tolist()),
        "training_rows": len(full),
        "model_parameters": model.get_params(),
        "validation_at_selected_threshold": metrics(validation[TARGET], validation_scores, threshold),
        "test_random_forest": metrics(test[TARGET], scores, threshold),
        "test_dummy_prior": metrics(test[TARGET], dummy_scores, threshold),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(data_dir.glob("PM_*.txt"))},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    predictions = test[["engine_id", "cycle", "rul", TARGET]].copy()
    predictions["risk_score"] = scores
    predictions["alert"] = (scores >= threshold).astype(int)
    predictions.to_csv(output / "test_predictions.csv", index=False, float_format="%.10f")
    pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--trees", type=int, default=200)
    args = parser.parse_args()
    report = run(args.data_dir, args.output, args.seed, args.horizon, args.trees)
    print(json.dumps({"threshold": report["threshold"],
                      "test_random_forest": report["test_random_forest"]}, indent=2))


if __name__ == "__main__":
    main()
