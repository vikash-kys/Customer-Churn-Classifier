"""Reproducible Telco churn analysis.

Run from the repository root:
    python src/run_analysis.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

If --data is omitted, the script searches common local paths and then tries
KaggleHub if installed and configured. Reports and charts are written to reports/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
MIN_RECALL_TARGET = 0.75
BOOTSTRAP_REPLICATES = 1000

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = ROOT / "reports"
DATASET_FILENAMES = (
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "Telco-Customer-Churn.csv",
    "telco-customer-churn.csv",
)
REQUIRED_COLUMNS = {"Churn", "tenure", "TotalCharges", "MonthlyCharges"}

sns.set_theme(style="whitegrid", context="notebook")


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas objects to JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _find_local_dataset(explicit_path: str | Path | None = None) -> Path | None:
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"The CSV specified by --data does not exist: {p}")

    search_dirs = [ROOT, ROOT / "data", ROOT / "src", ROOT.parent, Path.cwd(), Path.cwd() / "data"]
    seen: set[Path] = set()
    for directory in search_dirs:
        for filename in DATASET_FILENAMES:
            candidate = (directory / filename).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file():
                return candidate
    return None


def load_dataset(data_path: str | Path | None = None, allow_download: bool = True) -> tuple[pd.DataFrame, Path | None]:
    """Load the source CSV from disk or, optionally, the KaggleHub cache."""
    source = _find_local_dataset(data_path)
    if source is None and allow_download:
        try:
            import kagglehub  # type: ignore
            downloaded = Path(kagglehub.dataset_download("blastchar/telco-customer-churn"))
            matches = sorted(downloaded.rglob("*.csv"))
            if matches:
                source = matches[0]
        except Exception as exc:  # fail with actionable guidance below
            warnings.warn(f"KaggleHub download was unavailable: {exc}", RuntimeWarning)

    if source is None:
        raise FileNotFoundError(
            "Telco churn CSV not found. Download WA_Fn-UseC_-Telco-Customer-Churn.csv "
            "from https://www.kaggle.com/datasets/blastchar/telco-customer-churn and put it in "
            "the repository's data/ folder, or pass --data /path/to/file.csv. "
            "The raw dataset is not bundled in this project ZIP."
        )

    frame = pd.read_csv(source)
    missing_cols = REQUIRED_COLUMNS - set(frame.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_cols)}")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    return frame, source


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean known data-quality issues without fitting statistics on all rows."""
    df = raw.copy()
    original_rows = len(df)
    if "customerID" in df.columns:
        duplicate_ids = int(df["customerID"].duplicated().sum())
        df = df.drop(columns=["customerID"])
    else:
        duplicate_ids = None

    # TotalCharges is text in the source CSV. Blank values are known to occur for
    # zero-tenure accounts; only those semantically justified blanks are set to zero.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    missing_total_before = int(df["TotalCharges"].isna().sum())
    zero_tenure_missing = df["TotalCharges"].isna() & df["tenure"].eq(0)
    n_zero_tenure_fixed = int(zero_tenure_missing.sum())
    df.loc[zero_tenure_missing, "TotalCharges"] = 0.0
    residual_total_missing = int(df["TotalCharges"].isna().sum())

    target_map = {"Yes": 1, "No": 0, 1: 1, 0: 0}
    df["Churn"] = df["Churn"].map(target_map)
    if df["Churn"].isna().any():
        bad = raw.loc[df["Churn"].isna(), "Churn"].dropna().unique().tolist()
        raise ValueError(f"Unexpected Churn labels found: {bad}")
    df["Churn"] = df["Churn"].astype(int)

    # Distinct customers can have identical predictor profiles, so do not drop
    # duplicate rows after removing the customer identifier. Record the count only.
    duplicate_feature_profiles = int(df.drop(columns=["Churn"]).duplicated().sum())

    audit = {
        "rows_in_source": original_rows,
        "rows_after_duplicate_check": len(df),
        "columns_in_source": len(raw.columns),
        "duplicate_customer_ids": duplicate_ids,
        "duplicate_feature_profiles_retained": duplicate_feature_profiles,
        "totalcharges_missing_before_treatment": missing_total_before,
        "zero_tenure_totalcharges_set_to_zero": n_zero_tenure_fixed,
        "totalcharges_missing_after_semantic_treatment": residual_total_missing,
        "remaining_missing_values": {str(k): int(v) for k, v in df.isna().sum().items() if int(v) > 0},
        "churn_count": int(df["Churn"].sum()),
        "churn_rate": float(df["Churn"].mean()),
    }
    return df, audit


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])


def make_candidates(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """A baseline, an interpretable linear model, and a nonlinear tree ensemble."""
    return {
        "Dummy baseline": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", DummyClassifier(strategy="prior")),
        ]),
        "Logistic Regression": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", RandomForestClassifier(
                n_estimators=150,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


def cross_validate_candidates(candidates: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Pipeline]]:
    """Compare candidates and tune small parameter grids on training data only.

    The best CV score is used for selection and is therefore not an unbiased
    performance estimate. The held-out test set remains the final evaluation.
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "neg_brier_score": "neg_brier_score",
    }
    grids = {
        "Logistic Regression": {"classifier__C": [0.1, 1.0, 10.0]},
        "Random Forest": {
            "classifier__max_depth": [None, 12],
            "classifier__min_samples_leaf": [3, 8],
        },
    }
    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    best_estimators: dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        if name == "Dummy baseline":
            scores = cross_validate(
                estimator, X_train, y_train, cv=cv, scoring=scoring,
                n_jobs=1, return_train_score=False, error_score="raise"
            )
            raw[name] = scores
            metric_values = {metric: np.asarray(scores[f"test_{metric}"], dtype=float) for metric in scoring}
            params_label = "Fixed prior-only baseline"
        else:
            search = GridSearchCV(
                estimator=estimator, param_grid=grids[name], scoring=scoring,
                refit="average_precision", cv=cv, n_jobs=1,
                return_train_score=False, error_score="raise",
            )
            search.fit(X_train, y_train)
            raw[name] = search
            best_estimators[name] = search.best_estimator_
            best_index = int(search.best_index_)
            # GridSearchCV stores per-fold results in split{i}_test_<scorer>.
            metric_values = {
                metric: np.asarray([search.cv_results_[f"split{i}_test_{metric}"][best_index] for i in range(N_SPLITS)], dtype=float)
                for metric in scoring
            }
            params_label = json.dumps(search.best_params_, sort_keys=True)
        if name == "Dummy baseline":
            best_estimators[name] = clone(estimator)
        row: dict[str, Any] = {"Model": name, "Selected parameters": params_label}
        for metric in scoring:
            values = metric_values[metric]
            if metric == "neg_brier_score":
                values = -values
                label = "Brier score"
            else:
                label = metric.replace("_", " ").title()
            row[f"{label} mean"] = float(np.mean(values))
            row[f"{label} std"] = float(np.std(values, ddof=1))
        rows.append(row)
    table = pd.DataFrame(rows)
    return table, raw, best_estimators


def select_model(cv_table: pd.DataFrame) -> str:
    eligible = cv_table[cv_table["Model"].isin(["Logistic Regression", "Random Forest"])]
    if eligible.empty:
        raise RuntimeError("No non-baseline candidate models were evaluated.")
    return str(eligible.sort_values("Average Precision mean", ascending=False).iloc[0]["Model"])


def _select_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, min_recall: float = MIN_RECALL_TARGET) -> dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return {"threshold": 0.5, "precision_oof": None, "recall_oof": None, "target_recall": min_recall, "target_met": False}
    valid = np.flatnonzero(recall[:-1] >= min_recall)
    if valid.size:
        # Maximize precision among thresholds satisfying the recall constraint;
        # use the larger threshold to reduce outreach when precision ties.
        best_precision = float(np.max(precision[valid]))
        tied = valid[np.isclose(precision[valid], best_precision)]
        best_idx = int(tied[np.argmax(thresholds[tied])])
        target_met = True
    else:
        # Fallback is explicit: choose the threshold with the greatest recall.
        best_idx = int(np.argmax(recall[:-1]))
        target_met = False
    return {
        "threshold": float(thresholds[best_idx]),
        "precision_oof": float(precision[best_idx]),
        "recall_oof": float(recall[best_idx]),
        "target_recall": float(min_recall),
        "target_met": bool(target_met),
    }


def _binary_metrics(y_true: pd.Series | np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (np.asarray(prob) >= threshold).astype(int)
    y = np.asarray(y_true).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "brier_score": float(brier_score_loss(y, prob)),
        "confusion_matrix": cm.astype(int).tolist(),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "flagged_count": int(pred.sum()),
        "support": int(len(y)),
    }


def _bootstrap_intervals(y_true: pd.Series, prob: np.ndarray, threshold: float, n_boot: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    rng = np.random.default_rng(RANDOM_STATE)
    metrics: dict[str, list[float]] = {k: [] for k in ["roc_auc", "average_precision", "brier_score", "precision", "recall", "f1"]}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys, ps = y[idx], p[idx]
        if np.unique(ys).size < 2:
            continue
        pred = (ps >= threshold).astype(int)
        metrics["roc_auc"].append(float(roc_auc_score(ys, ps)))
        metrics["average_precision"].append(float(average_precision_score(ys, ps)))
        metrics["brier_score"].append(float(brier_score_loss(ys, ps)))
        metrics["precision"].append(float(precision_score(ys, pred, zero_division=0)))
        metrics["recall"].append(float(recall_score(ys, pred, zero_division=0)))
        metrics["f1"].append(float(f1_score(ys, pred, zero_division=0)))
    return {
        name: {"lower_95": float(np.percentile(vals, 2.5)), "upper_95": float(np.percentile(vals, 97.5)), "valid_replicates": len(vals)}
        for name, vals in metrics.items() if vals
    }


def _risk_capture(y_true: pd.Series, probabilities: np.ndarray, fractions=(0.10, 0.20)) -> list[dict[str, Any]]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities)
    order = np.argsort(-p)
    rows = []
    for fraction in fractions:
        n_select = max(1, int(math.ceil(len(y) * fraction)))
        chosen = order[:n_select]
        n_captured = int(y[chosen].sum())
        total_positive = int(y.sum())
        rows.append({
            "risk_fraction": float(fraction),
            "customers_selected": n_select,
            "churners_captured": n_captured,
            "precision_in_selected": float(n_captured / n_select),
            "fraction_of_all_churners_captured": float(n_captured / total_positive) if total_positive else None,
            "total_churners_in_test": total_positive,
            "random_selection_expected_recall": float(n_select / len(y)),
        })
    return rows


def _subgroup_metrics(X_test: pd.DataFrame, y_test: pd.Series, prob: np.ndarray, threshold: float) -> pd.DataFrame:
    base = X_test.reset_index(drop=True).copy()
    base["actual_churn"] = np.asarray(y_test).astype(int)
    base["predicted_churn"] = (prob >= threshold).astype(int)
    base["probability"] = prob
    if "tenure" in base:
        base["Tenure band"] = pd.cut(
            base["tenure"], bins=[-1, 6, 12, 24, 48, 72],
            labels=["0–6 months", "7–12 months", "13–24 months", "25–48 months", "49–72 months"],
            include_lowest=True,
        ).astype("object").fillna("Unknown")
    columns = [c for c in ["Contract", "InternetService", "PaymentMethod", "Tenure band"] if c in base.columns]
    rows: list[dict[str, Any]] = []
    for col in columns:
        for value, group in base.groupby(col, dropna=False, observed=False):
            n = len(group)
            positives = int(group["actual_churn"].sum())
            pred_pos = int(group["predicted_churn"].sum())
            tp = int(((group["actual_churn"] == 1) & (group["predicted_churn"] == 1)).sum())
            rows.append({
                "dimension": col, "segment": str(value), "n": int(n), "actual_churners": positives,
                "observed_churn_rate": float(group["actual_churn"].mean()) if n else None,
                "precision": float(tp / pred_pos) if pred_pos else 0.0,
                "recall": float(tp / positives) if positives else None,
                "flagged_fraction": float(pred_pos / n) if n else None,
                "minimum_n_met": bool(n >= 50),
            })
    return pd.DataFrame(rows)


def _feature_importance(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    # Used only for post-hoc interpretation after the model has been frozen;
    # it is not used to select features or tune hyperparameters.
    importance = permutation_importance(
        model, X_test, y_test, scoring="average_precision", n_repeats=5,
        random_state=RANDOM_STATE, n_jobs=1,
    )
    return pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _make_plots(out_dir: Path, cv_table: pd.DataFrame, threshold_info: dict[str, Any], y_train: pd.Series,
                oof_prob: np.ndarray, y_test: pd.Series, raw_test_prob: np.ndarray,
                calibrated_test_prob: np.ndarray, chosen_prob: np.ndarray, threshold: float,
                metrics: dict[str, Any], subgroup: pd.DataFrame, importance: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Cross-validation model comparison: standard deviations are fold-to-fold variability, not CIs.
    plot_df = cv_table[["Model", "Average Precision mean", "Average Precision std"]].copy()
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    ax.bar(plot_df["Model"], plot_df["Average Precision mean"], yerr=plot_df["Average Precision std"], capsize=5)
    ax.set_ylabel("Average precision (higher is better)")
    ax.set_title("Training-only stratified cross-validation")
    upper = min(1.0, max(0.5, float((plot_df["Average Precision mean"] + plot_df["Average Precision std"]).max() + 0.08)))
    ax.set_ylim(0, upper)
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout(); fig.savefig(out_dir / "cv_average_precision.png", dpi=170); plt.close(fig)

    prec, rec, thresholds = precision_recall_curve(y_train, oof_prob)
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    ax.plot(thresholds, prec[:-1], label="Precision")
    ax.plot(thresholds, rec[:-1], label="Recall")
    ax.axhline(threshold_info["target_recall"], linestyle="--", linewidth=1.2, label=f"Recall target ({threshold_info['target_recall']:.0%})")
    ax.axvline(threshold_info["threshold"], linestyle=":", linewidth=1.7, label=f"Selected threshold ({threshold_info['threshold']:.3f})")
    ax.set_xlabel("Probability threshold selected from training OOF predictions")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02); ax.set_title("Operating-threshold selection (training only)")
    ax.legend(loc="best", frameon=True); fig.tight_layout()
    fig.savefig(out_dir / "threshold_tradeoff.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    for label, probs in [("Uncalibrated", raw_test_prob), ("Sigmoid calibrated", calibrated_test_prob)]:
        fraction_positive, mean_predicted = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_predicted, fraction_positive, marker="o", label=label)
    ax.set_xlabel("Mean predicted churn probability")
    ax.set_ylabel("Observed churn fraction")
    ax.set_title("Probability calibration on the held-out test set")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "calibration_curve.png", dpi=170); plt.close(fig)

    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax,
                xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
    ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label")
    ax.set_title(f"Test-set confusion matrix (threshold={threshold:.3f})")
    fig.tight_layout(); fig.savefig(out_dir / "test_confusion_matrix.png", dpi=170); plt.close(fig)

    if not importance.empty:
        imp = importance.head(12).sort_values("importance_mean", ascending=True)
        fig, ax = plt.subplots(figsize=(8.8, 5.8))
        ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"], capsize=2)
        ax.axvline(0, color="gray", linewidth=1)
        ax.set_xlabel("Decrease in average precision after permutation")
        ax.set_title("Permutation importance on held-out data (post-hoc only)")
        fig.tight_layout(); fig.savefig(out_dir / "permutation_importance.png", dpi=170); plt.close(fig)

    if not subgroup.empty:
        sg = subgroup[(subgroup["dimension"] == "Contract") & subgroup["minimum_n_met"] & subgroup["recall"].notna()]
        if not sg.empty:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.bar(sg["segment"], sg["recall"])
            ax.set_ylim(0, 1); ax.set_ylabel("Recall at selected threshold")
            ax.set_title("Churner detection by contract type (test set)")
            ax.tick_params(axis="x", rotation=12)
            fig.tight_layout(); fig.savefig(out_dir / "contract_subgroup_recall.png", dpi=170); plt.close(fig)


def _logistic_coefficients(fitted_logistic_pipeline: Pipeline) -> pd.DataFrame:
    """Return interpretable coefficients from the fitted logistic baseline.

    Numeric inputs have been standardized; one-hot encoded levels are contrasts
    against the dropped reference category. Odds ratios are conditional model
    associations, not causal effects.
    """
    preprocessor = fitted_logistic_pipeline.named_steps["preprocessor"]
    classifier = fitted_logistic_pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = np.asarray(classifier.coef_).reshape(-1)
    if len(feature_names) != len(coefficients):
        raise RuntimeError("Preprocessor feature names do not align with logistic coefficients.")
    return pd.DataFrame({
        "feature": feature_names,
        "coefficient_log_odds": coefficients,
        "odds_ratio": np.exp(np.clip(coefficients, -20, 20)),
        "direction": np.where(coefficients >= 0, "Higher modeled churn odds", "Lower modeled churn odds"),
    }).assign(abs_coefficient=lambda d: d["coefficient_log_odds"].abs()).sort_values("abs_coefficient", ascending=False).drop(columns=["abs_coefficient"]).reset_index(drop=True)


def _make_eda_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Create descriptive EDA charts from the cleaned dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = df["Churn"].map({0: "No churn", 1: "Churn"}).value_counts().reindex(["No churn", "Churn"])
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    counts.plot(kind="bar", ax=ax, color=["#7569cf", "#d9913a"])
    ax.set_title("Target distribution")
    ax.set_ylabel("Customer count")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(out_dir / "target_distribution.png", dpi=170); plt.close(fig)

    numeric_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df]
    if numeric_cols:
        fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5.0 * len(numeric_cols), 4.0))
        if len(numeric_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, numeric_cols):
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            ax.hist(vals, bins=30, color="#7569cf", edgecolor="white")
            ax.set_title(col)
            ax.set_ylabel("Customer count")
        fig.suptitle("Distributions of key numeric features", y=1.03)
        fig.tight_layout(); fig.savefig(out_dir / "numeric_feature_distributions.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    for col, filename, title in [
        ("Contract", "churn_by_contract.png", "Observed churn rate by contract"),
        ("InternetService", "churn_by_internet_service.png", "Observed churn rate by internet service"),
    ]:
        if col not in df:
            continue
        summary = df.groupby(col, observed=False)["Churn"].agg(["mean", "size"]).sort_values("mean", ascending=False)
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        bars = ax.bar(summary.index.astype(str), summary["mean"], color="#7569cf")
        ax.set_title(title)
        ax.set_ylabel("Observed churn fraction")
        ax.set_ylim(0, min(1.0, float(summary["mean"].max() + 0.12)))
        ax.tick_params(axis="x", rotation=12)
        for bar, n in zip(bars, summary["size"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f"n={int(n):,}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout(); fig.savefig(out_dir / filename, dpi=170); plt.close(fig)

    if "tenure" in df:
        work = df[["tenure", "Churn"]].copy()
        work["Tenure band"] = pd.cut(work["tenure"], bins=[-1, 6, 12, 24, 48, 72],
                                      labels=["0–6", "7–12", "13–24", "25–48", "49–72"], include_lowest=True)
        summary = work.groupby("Tenure band", observed=False)["Churn"].agg(["mean", "size"])
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.bar(summary.index.astype(str), summary["mean"], color="#7569cf")
        ax.set_title("Observed churn rate by tenure band")
        ax.set_ylabel("Observed churn fraction")
        ax.set_xlabel("Tenure in months")
        ax.set_ylim(0, min(1.0, float(summary["mean"].max() + 0.12)))
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout(); fig.savefig(out_dir / "churn_by_tenure_band.png", dpi=170); plt.close(fig)


def run_analysis(data_path: str | Path | None = None, report_dir: str | Path | None = None, allow_download: bool = True,
                 bootstrap_replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    report_path = Path(report_dir).resolve() if report_dir else DEFAULT_REPORT_DIR
    report_path.mkdir(parents=True, exist_ok=True)

    raw, source_path = load_dataset(data_path, allow_download=allow_download)
    df, audit = clean_dataset(raw)
    X_train, X_test, y_train, y_test, numeric_features, categorical_features = split_data(df)
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    candidates = make_candidates(preprocessor)
    cv_table, raw_cv, tuned_candidates = cross_validate_candidates(candidates, X_train, y_train)
    cv_table.to_csv(report_path / "cv_results.csv", index=False)

    chosen_name = select_model(cv_table)
    base_model = clone(tuned_candidates[chosen_name])
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Calibration selection uses training-only cross-validation Brier scores.
    raw_brier = float(cv_table.loc[cv_table["Model"] == chosen_name, "Brier score mean"].iloc[0])
    calibrated_template = CalibratedClassifierCV(estimator=clone(base_model), method="sigmoid", cv=3)
    calibrated_scores = cross_validate(
        calibrated_template, X_train, y_train, cv=cv,
        scoring={"brier": "neg_brier_score", "average_precision": "average_precision"},
        n_jobs=1, return_train_score=False, error_score="raise",
    )
    calibrated_brier = float(-np.mean(calibrated_scores["test_brier"]))
    use_calibration = calibrated_brier < raw_brier
    final_template = clone(calibrated_template if use_calibration else base_model)

    # Out-of-fold probabilities keep threshold choice separate from the final test set.
    oof_prob = cross_val_predict(final_template, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    threshold_info = _select_threshold(y_train, oof_prob, MIN_RECALL_TARGET)
    threshold = threshold_info["threshold"]

    # Fit once on all available training records; the held-out test set is not used for selection.
    final_model = clone(final_template).fit(X_train, y_train)
    raw_fitted = clone(base_model).fit(X_train, y_train)
    calibrated_fitted = clone(calibrated_template).fit(X_train, y_train)
    raw_test_prob = raw_fitted.predict_proba(X_test)[:, 1]
    calibrated_test_prob = calibrated_fitted.predict_proba(X_test)[:, 1]
    test_prob = final_model.predict_proba(X_test)[:, 1]

    metrics_default = _binary_metrics(y_test, test_prob, 0.5)
    metrics_selected = _binary_metrics(y_test, test_prob, threshold)
    intervals = _bootstrap_intervals(y_test, test_prob, threshold, n_boot=bootstrap_replicates)
    risk_capture = _risk_capture(y_test, test_prob)
    subgroup = _subgroup_metrics(X_test, y_test, test_prob, threshold)
    subgroup.to_csv(report_path / "subgroup_metrics.csv", index=False)
    importance = _feature_importance(final_model, X_test, y_test)
    importance.to_csv(report_path / "permutation_importance.csv", index=False)
    logistic_reference = clone(tuned_candidates["Logistic Regression"]).fit(X_train, y_train)
    coefficients = _logistic_coefficients(logistic_reference)
    coefficients.to_csv(report_path / "logistic_coefficients.csv", index=False)

    # Compact EDA summaries are descriptive, not causal.
    eda: dict[str, Any] = {
        "target_counts": {str(k): int(v) for k, v in df["Churn"].value_counts().sort_index().items()},
        "target_labels": {"0": "No churn", "1": "Churn"},
        "churn_rate_by_contract": {},
        "churn_rate_by_internet_service": {},
        "numeric_summary": df[[c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df]].describe().round(3).to_dict(),
    }
    for col, target in [("Contract", "churn_rate_by_contract"), ("InternetService", "churn_rate_by_internet_service")]:
        if col in df:
            eda[target] = {str(k): {"customers": int(v["size"]), "churn_rate": float(v["mean"])}
                           for k, v in df.groupby(col, dropna=False)["Churn"].agg(["mean", "size"]).iterrows()}

    cv_rows = cv_table.to_dict(orient="records")
    cv_calibration = {
        "raw_brier_cv_mean": raw_brier,
        "sigmoid_calibrated_brier_cv_mean": calibrated_brier,
        "brier_improved_in_cv": bool(use_calibration),
        "selected_probability_method": "Sigmoid calibrated" if use_calibration else "Uncalibrated",
        "note": "Calibration choice was made using training-only cross-validation Brier scores; test-set curves are descriptive final evaluation.",
    }
    _make_eda_plots(df, report_path)
    _make_plots(report_path, cv_table, threshold_info, y_train, oof_prob, y_test,
                raw_test_prob, calibrated_test_prob, test_prob, threshold,
                metrics_selected, subgroup, importance)

    result = {
        "status": "ready",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset": {
            "source_filename": source_path.name if source_path else None,
            "rows": int(len(df)),
            "predictor_count_after_identifier_drop": int(df.shape[1] - 1),
            "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
            "positive_class": "Churn = Yes (1)",
            "prevalence_overall": float(df["Churn"].mean()),
            "prevalence_test": float(y_test.mean()),
        },
        "data_audit": audit,
        "cross_validation": {
            "n_splits": N_SPLITS,
            "method": "StratifiedKFold(shuffle=True, random_state=42), training set only",
            "selection_metric": "Mean cross-validated average precision; hyperparameters are searched using training-only CV. These selected CV scores are for selection and can be optimistic; the held-out test set is the final evaluation. DummyClassifier is a reference baseline and is not eligible for selection.",
            "models": cv_rows,
        },
        "selected_model": chosen_name,
        "calibration": cv_calibration,
        "threshold_selection": {
            **threshold_info,
            "method": "Among thresholds meeting the 75% recall target on out-of-fold training probabilities, select the threshold with the highest precision; ties prefer the larger threshold.",
            "final_test_recall_target_met": bool(metrics_selected["recall"] >= MIN_RECALL_TARGET),
        },
        "test_metrics_default_threshold": metrics_default,
        "test_metrics_selected_threshold": metrics_selected,
        "bootstrap_95_percent_intervals": intervals,
        "risk_capture": risk_capture,
        "subgroup_metrics": subgroup.to_dict(orient="records"),
        "feature_importance": importance.head(20).to_dict(orient="records"),
        "logistic_coefficients": coefficients.head(30).to_dict(orient="records"),
        "eda": eda,
        "figures": {
            "cv_average_precision": "reports/cv_average_precision.png",
            "threshold_tradeoff": "reports/threshold_tradeoff.png",
            "calibration_curve": "reports/calibration_curve.png",
            "test_confusion_matrix": "reports/test_confusion_matrix.png",
            "permutation_importance": "reports/permutation_importance.png",
            "contract_subgroup_recall": "reports/contract_subgroup_recall.png",
            "target_distribution": "reports/target_distribution.png",
            "churn_by_contract": "reports/churn_by_contract.png",
            "churn_by_internet_service": "reports/churn_by_internet_service.png",
            "churn_by_tenure_band": "reports/churn_by_tenure_band.png",
            "numeric_feature_distributions": "reports/numeric_feature_distributions.png",
        },
        "limitations": [
            "This is a single snapshot of one company's customers, not a temporal or external validation study.",
            "Observed churn risk is not the same as responsiveness to a retention intervention; causal uplift requires an experiment or suitable causal design.",
            "Segment metrics with fewer than 50 test customers are descriptive and should not be over-interpreted.",
            "The selected recall target is an illustrative operating constraint, not a business requirement supplied by a telecom operator.",
            "Bootstrap intervals quantify sampling uncertainty conditional on this held-out split; they do not capture dataset shift or all sources of model uncertainty.",
        ],
    }
    with (report_path / "results.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, indent=2, ensure_ascii=False)
    with (report_path / "eda_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(eda), f, indent=2, ensure_ascii=False)

    print(f"Dataset loaded: {source_path}")
    print(f"Rows: {len(df):,} | train: {len(X_train):,} | held-out test: {len(X_test):,} | churn prevalence: {df['Churn'].mean():.1%}")
    print("\nTraining-only cross-validation comparison:")
    display_cols = ["Model", "Average Precision mean", "Average Precision std", "Roc Auc mean", "Brier score mean"]
    print(cv_table[[c for c in display_cols if c in cv_table]].round(4).to_string(index=False))
    print(f"\nSelected model by CV average precision: {chosen_name}")
    print(f"Probability calibration selected by CV Brier score: {cv_calibration['selected_probability_method']}")
    print(f"Threshold selected from training out-of-fold probabilities: {threshold:.4f} (OOF recall {threshold_info['recall_oof']:.3f})")
    print("\nFinal held-out test metrics at selected threshold:")
    print(pd.Series({k: v for k, v in metrics_selected.items() if k not in {"confusion_matrix"}}).round(4).to_string())
    print(f"\nReports and figures written to: {report_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible Telco customer churn analysis.")
    parser.add_argument("--data", type=str, default=None, help="Path to the Telco churn CSV (optional if KaggleHub is configured).")
    parser.add_argument("--report-dir", type=str, default=None, help="Output directory (defaults to repository reports/).")
    parser.add_argument("--no-download", action="store_true", help="Do not attempt a KaggleHub download if no local CSV is found.")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES, help=f"Bootstrap replicates for 95% intervals (default: {BOOTSTRAP_REPLICATES}).")
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100 for useful interval estimates.")
    run_analysis(data_path=args.data, report_dir=args.report_dir, allow_download=not args.no_download,
                 bootstrap_replicates=args.bootstrap)


if __name__ == "__main__":
    main()
