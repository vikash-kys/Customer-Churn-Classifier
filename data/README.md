# Dataset setup

The raw Telco Customer Churn CSV is intentionally not included in this repository ZIP. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), then place the CSV in this folder.

Expected path:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Alternatively, pass the file's location explicitly to the runner:

```bash
python src/run_analysis.py --data /absolute/path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The script also tries KaggleHub if installed and configured, but a local CSV is the most reproducible option. The CSV is ignored by Git so the data file is not accidentally committed with the source code.
