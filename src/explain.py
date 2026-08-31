"""Fraud model training, persistence, and SHAP-based transaction explanations.

Mirrors the temporal split, XGBoost training, and cost-based threshold selection
from notebooks/03_threshold_optimization.ipynb. Intended for use by the agent
pipeline — not for interactive notebook work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

# --- constants (must match notebook 03) ---
RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION_OF_TOTAL = 0.15
FALSE_POSITIVE_COST_INR = 500

CONFIG_ATTR = "_riskshield_config"


def _config_path(model_path: Path) -> Path:
    """JSON config lives alongside the joblib model file."""
    return model_path.with_name(f"{model_path.stem}_config.json")


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """55% train / 15% validation / 30% test — chronological, no shuffle."""
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    split_test = int(n * TRAIN_FRACTION)
    split_val = int(n * (TRAIN_FRACTION - VAL_FRACTION_OF_TOTAL))

    train_df = df.iloc[:split_val].copy()
    val_df = df.iloc[split_val:split_test].copy()
    test_df = df.iloc[split_test:].copy()
    return train_df, val_df, test_df


def _compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    fp_cost: float,
) -> dict[str, float]:
    """Precision/recall/F1 and expected cost at a single decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_cost = float(amounts[fn_mask].sum())
    fp_cost_total = float(fp_mask.sum() * fp_cost)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_count": fp,
        "fn_count": fn,
        "fn_cost": fn_cost,
        "fp_cost": fp_cost_total,
        "total_cost": fn_cost + fp_cost_total,
    }


def _find_cost_minimizing_threshold(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    amount_val: np.ndarray,
    fp_cost: float = FALSE_POSITIVE_COST_INR,
) -> float:
    """Sweep thresholds 0.05–0.95 and return the cost-minimizing value (validation only)."""
    thresholds = np.arange(0.05, 1.0, 0.05)
    best_threshold = 0.5
    best_cost = float("inf")

    for threshold in thresholds:
        metrics = _compute_threshold_metrics(y_val, val_probs, amount_val, threshold, fp_cost)
        if metrics["total_cost"] < best_cost:
            best_cost = metrics["total_cost"]
            best_threshold = float(threshold)

    return best_threshold


def train_and_save_model(data_path: str | Path, model_path: str | Path) -> dict[str, Any]:
    """Train XGBoost on the 55% temporal train split and persist model + config.

    Threshold is selected on the 15% validation slice via cost minimization
    (same logic as notebook 03). The 30% test split is not used here.

    Returns the saved config dict.
    """
    data_path = Path(data_path)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    train_df, val_df, _test_df = _temporal_split(df)

    feature_cols = [c for c in df.columns if c != "Class"]
    X_train, y_train = train_df[feature_cols], train_df["Class"]
    X_val, y_val = val_df[feature_cols], val_df["Class"]
    amount_val = val_df["Amount"].values

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    threshold = _find_cost_minimizing_threshold(y_val.values, val_probs, amount_val)

    config = {
        "threshold": round(threshold, 2),
        "feature_cols": feature_cols,
        "scale_pos_weight": float(scale_pos_weight),
        "false_positive_cost_inr": FALSE_POSITIVE_COST_INR,
        "random_state": RANDOM_STATE,
        "train_fraction": TRAIN_FRACTION,
        "val_fraction_of_total": VAL_FRACTION_OF_TOTAL,
    }

    joblib.dump(model, model_path)
    with _config_path(model_path).open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    setattr(model, CONFIG_ATTR, config)
    return config


def load_model(model_path: str | Path) -> tuple[XGBClassifier, dict[str, Any]]:
    """Load a persisted model and its JSON config from disk."""
    model_path = Path(model_path)
    model = joblib.load(model_path)

    with _config_path(model_path).open(encoding="utf-8") as f:
        config = json.load(f)

    setattr(model, CONFIG_ATTR, config)
    return model, config


def build_explainer(model: XGBClassifier) -> shap.TreeExplainer:
    """Fit a SHAP TreeExplainer once and reuse across explanation requests."""
    return shap.TreeExplainer(model)


def _features_dict_to_row(transaction_features: dict[str, Any], feature_cols: list[str]) -> pd.DataFrame:
    """Convert a feature dict to a single-row DataFrame in training column order."""
    missing = set(feature_cols) - set(transaction_features)
    if missing:
        raise ValueError(f"Transaction missing required features: {sorted(missing)}")

    return pd.DataFrame([[transaction_features[col] for col in feature_cols]], columns=feature_cols)


def _shap_values_for_row(shap_explainer: shap.TreeExplainer, row: pd.DataFrame) -> np.ndarray:
    """Return 1-D SHAP contributions for the positive (fraud) class."""
    shap_values = shap_explainer.shap_values(row)

    # XGBoost binary classifiers may return a list [neg, pos] or a single array.
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1])
    else:
        values = np.asarray(shap_values)

    return values.reshape(-1)


def explain_transaction(
    model: XGBClassifier,
    shap_explainer: shap.TreeExplainer,
    transaction_features: dict[str, Any],
) -> dict[str, Any]:
    """Score a transaction and return risk score, flag, and top SHAP factors.

    Expects ``model`` to carry config from ``load_model`` / ``train_and_save_model``
    (threshold and feature column order). ``shap_explainer`` should be built once via
    ``build_explainer`` and reused.
    """
    config = getattr(model, CONFIG_ATTR, None)
    if config is None:
        raise ValueError(
            "Model has no attached config. Load via load_model() or train_and_save_model()."
        )

    feature_cols: list[str] = config["feature_cols"]
    threshold: float = config["threshold"]

    row = _features_dict_to_row(transaction_features, feature_cols)
    risk_score = float(model.predict_proba(row)[0, 1])
    flagged = risk_score >= threshold

    shap_contribs = _shap_values_for_row(shap_explainer, row)
    ranked_idx = np.argsort(np.abs(shap_contribs))[::-1][:3]

    top_factors = []
    for idx in ranked_idx:
        value = float(shap_contribs[idx])
        top_factors.append(
            {
                "feature": feature_cols[idx],
                "shap_value": round(value, 6),
                "direction": "increases_risk" if value > 0 else "decreases_risk",
            }
        )

    return {
        "risk_score": round(risk_score, 6),
        "flagged": flagged,
        "top_factors": top_factors,
    }


def _row_to_feature_dict(row: pd.Series, feature_cols: list[str]) -> dict[str, Any]:
    return {col: row[col] for col in feature_cols}


def _find_confusion_examples(
    model: XGBClassifier,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Pick one true positive, false positive, and false negative from the test set."""
    X_test = test_df[feature_cols]
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    y_true = test_df["Class"].values

    categories = {
        "true_positive": (y_true == 1) & (preds == 1),
        "false_positive": (y_true == 0) & (preds == 1),
        "false_negative": (y_true == 1) & (preds == 0),
    }

    examples: dict[str, dict[str, Any]] = {}
    for name, mask in categories.items():
        indices = np.where(mask)[0]
        if len(indices) == 0:
            raise RuntimeError(f"No {name} examples found in test set at threshold {threshold}")
        row = test_df.iloc[indices[0]]
        examples[name] = _row_to_feature_dict(row, feature_cols)

    return examples


if __name__ == "__main__":
    import pprint

    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "creditcard.csv"
    model_path = project_root / "models" / "fraud_xgb.joblib"

    if not model_path.exists():
        print(f"Model not found at {model_path} — training now...\n")
        config = train_and_save_model(data_path, model_path)
        print(f"Saved model and config (threshold={config['threshold']:.2f})\n")
    else:
        print(f"Loading existing model from {model_path}\n")

    model, config = load_model(model_path)
    explainer = build_explainer(model)

    # Recreate test split only to source sanity-check examples (not used for training).
    df = pd.read_csv(data_path)
    _train_df, _val_df, test_df = _temporal_split(df)
    examples = _find_confusion_examples(
        model, test_df, config["feature_cols"], config["threshold"]
    )

    for label, features in examples.items():
        result = explain_transaction(model, explainer, features)
        print(f"=== {label.replace('_', ' ').title()} ===")
        print(f"  Actual class: {1 if label != 'false_positive' else 0}")
        pprint.pprint(result, sort_dicts=False)
        print()
