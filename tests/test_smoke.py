"""Synthetic-data smoke test. This does NOT validate real Telco model metrics.

Run from the repository root with:
    python tests/test_smoke.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_analysis import run_analysis  # noqa: E402


def make_synthetic_dataset(path: Path, n: int = 720) -> None:
    rng = np.random.default_rng(2026)
    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.22, 0.23])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.48, 0.17])
    monthly = np.clip(rng.normal(72, 22, n), 18, 120)
    total = tenure * monthly + rng.normal(0, 20, n)
    logit = -1.2 + 1.0 * (contract == "Month-to-month") - 0.7 * (contract == "Two year") - 0.02 * tenure
    probability = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < probability, "Yes", "No")
    df = pd.DataFrame({
        "customerID": [f"SYN{i:05d}" for i in range(n)],
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": tenure,
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": internet,
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": contract,
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges": monthly.round(2),
        "TotalCharges": total.round(2).astype(str),
        "Churn": churn,
    })
    df.loc[tenure == 0, "TotalCharges"] = " "
    df.to_csv(path, index=False)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="telco-churn-smoke-") as tmp:
        work = Path(tmp)
        data_path = work / "synthetic_test_data.csv"
        make_synthetic_dataset(data_path)
        result = run_analysis(data_path=data_path, report_dir=work / "reports", allow_download=False, bootstrap_replicates=100)
        assert result["status"] == "ready"
        assert result["dataset"]["rows"] > 0
        assert result["selected_model"] in {"Logistic Regression", "Random Forest"}
        assert 0 <= result["threshold_selection"]["threshold"] <= 1
        assert 0 <= result["test_metrics_selected_threshold"]["roc_auc"] <= 1
        assert (work / "reports" / "results.json").is_file()
        assert (work / "reports" / "cv_average_precision.png").is_file()
        print("PASS: analysis pipeline and report generation work on a synthetic smoke fixture.")
        print("NOTE: this does not validate real Telco dataset metrics.")


if __name__ == "__main__":
    main()
