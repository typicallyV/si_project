# Social Media Analyzer — Full Local Dashboard

## Included functionality
- Kaggle dataset download using KaggleHub
- Pandas CSV import, duplicate removal, whitespace cleanup, missing-value report, metric normalization
- Support for a CSV upload or a local CSV path
- Interactive platform, sentiment, and topic filters
- KPI metrics, engagement distributions, top posts, platform comparison
- Hashtag frequency analysis
- Sentiment summary and average engagement by sentiment
- Daily/weekly/monthly posting and engagement trends
- Viral/non-viral insights (uses a provided viral column or labels the top 10% of engagement as viral)
- Correlation matrix, descriptive statistics, author/topic analysis, frequent words
- Optional PySpark summary and word count in local mode
- Optional Cassandra table creation and record insertion
- CSV exports

## Install (Windows)
Use Python 3.10 or 3.11 for a smoother PySpark setup.

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configure Kaggle API
1. Sign in to Kaggle and create/download an API token from Account Settings.
2. Place `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json`.
3. Keep it private and do not commit it to GitHub.
4. Run the app:

```bat
streamlit run app.py
```

Then open `http://localhost:8501`.

## Load data
In the left sidebar, choose Kaggle dataset and enter the slug copied from a URL such as:
`https://www.kaggle.com/datasets/OWNER/DATASET-SLUG`
Enter only `OWNER/DATASET-SLUG`, then click Download from Kaggle. If multiple CSV files exist, choose one from the selector.

Alternatively, upload a CSV or enter its local path.

## Cassandra (optional)
The Cassandra section assumes a reachable Cassandra server and the Python package `cassandra-driver` (included in requirements). The app creates the keyspace/table if needed and inserts rows. Use only a local/trusted Cassandra instance. For large datasets, use prepared batches or a bulk-loading approach rather than one insert per row.

## PySpark (optional)
The app starts Spark in `local[*]` mode; no Hadoop cluster is required for this local Spark feature. Ensure Java and PySpark versions are compatible if Spark startup fails.

## Important
Column aliases are defined at the top of `app.py`. Edit `ALIASES` if your Kaggle dataset uses different field names. Sentiment analysis uses existing sentiment labels; this code does not run a trained sentiment classifier.
