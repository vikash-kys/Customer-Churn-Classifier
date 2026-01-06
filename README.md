# Telco Customer Churn: Applied Model Evaluation

A reproducible applied data-science project evaluating telecom customer churn risk. It focuses on sound validation, train-only preprocessing, model comparison, probability calibration, threshold selection, uncertainty, risk ranking, subgroup errors, and the limits of causal interpretation.

## Project structure

- `index.html`: portfolio page; reads generated report data from `reports/results.json`.
- `src/run_analysis.py`: source of truth for loading, preprocessing, model selection, calibration, thresholding, metrics, confidence intervals, interpretation, and report generation.
- `src/Churn_prediction.ipynb`: guided notebook for running and inspecting the same implementation.
- `data/README.md`: instructions for obtaining the raw dataset.
- `reports/`: generated JSON, CSV tables, and charts. Results are not considered final until the analysis has run against the real CSV.

## Setup

Use Python 3.10 or later. Download the Telco Customer Churn CSV from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Install dependencies and run the analysis from the repository root:

```bash
python -m pip install -r requirements.txt
python src/run_analysis.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The script writes `reports/results.json`, `reports/eda_summary.json`, model-comparison/subgroup/permutation-importance/Logistic-coefficient CSVs, and EDA/evaluation PNG figures. Refresh `index.html` (or the published static site) after running the script to display the generated results. The raw CSV is ignored by Git; do not accidentally commit it.

You can also open `src/Churn_prediction.ipynb` in Jupyter and run the cells in order. The notebook calls the same implementation as the command-line runner, so the notebook, report JSON, and website do not maintain separate model implementations.

## Evaluation protocol

1. Audit `TotalCharges`, `tenure`, target labels, and customer identifiers.
2. Create a stratified 80/20 train/test split with a fixed random seed.
3. Put imputation, scaling, and categorical encoding inside scikit-learn pipelines.
4. Compare a `DummyClassifier`, `LogisticRegression`, and `RandomForestClassifier` using stratified five-fold CV on training data only. Small grids tune Logistic Regression's `C` and Random Forest's `max_depth` and `min_samples_leaf`; select the non-baseline candidate using mean average precision. The selected CV score is a selection statistic and can be optimistic.
5. Compare raw and sigmoid-calibrated probabilities by training-only cross-validated Brier score.
6. Select a threshold using out-of-fold training predictions, targeting at least 75% recall and preferring the highest precision among thresholds meeting that target.
7. Evaluate the frozen workflow on the held-out test set. Report ROC-AUC, average precision, threshold-specific metrics, bootstrap intervals, calibration, top-risk capture, subgroup performance, and permutation importance.

The 75% recall target and top-risk fractions are illustrative. A business deployment requires a documented objective and evidence about offer costs, intervention effectiveness, customer value, and campaign capacity.

## Limitations

This is a single snapshot from one fictional telecom provider. Churn prediction does not establish which customers can be persuaded to stay. Causal retention effectiveness requires a suitable experiment or causal design. Subgroup metrics are descriptive, particularly for small groups, and feature importance is not causal attribution.

## Dataset reference

Telco Customer Churn, originally distributed as an IBM sample dataset and mirrored on Kaggle by `blastchar`: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
