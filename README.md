
# ML-Based Portfolio Optimization

An end-to-end portfolio optimization project combining machine learning, historical risk estimation, and constrained mean-variance optimization. The system predicts stock returns, constructs portfolios, and evaluates their performance against the NIFTY 50 benchmark.

## 1. Project Pipeline

1. **Data preprocessing:** Clean and consolidate historical stock data.
2. **Price adjustment:** Apply backward adjustments for configured corporate actions.
3. **Feature engineering:** Generate return, momentum, volatility, price, and volume features.
4. **ML prediction:** Train an XGBoost-based cross-sectional model using walk-forward validation.
5. **Model evaluation:** Evaluate predictions using Information Coefficient (IC), IC Information Ratio, and hit rate.
6. **Portfolio optimization:** Generate portfolio weights using mean-variance optimization.
7. **Backtesting:** Simulate periodic rebalancing, transaction costs, and portfolio drift.
8. **Performance analysis:** Compare portfolio performance with NIFTY 50.

## 2. Repository Structure

```text
Optimization-Project/
│
├── data/
│   ├── raw/                 # Input datasets (not tracked)
│   └── processed/           # Generated datasets (not tracked)
│
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   │   ├── adjust_prices.py
│   │   └── extract_benchmark.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── model2_ranker.py
│   │   └── evaluate_ranker.py
│   │
│   ├── optimization/
│   │   ├── data_interface.py
│   │   ├── risk_model.py
│   │   ├── constraints.py
│   │   ├── mean_variance.py
│   │   ├── constrained_portfolio.py
│   │   └── optimizer.py
│   │
│   └── backtesting/
│       ├── backtest.py
│       ├── metrics.py
│       └── plot_results.py
│
├── results/
│   └── figures/
│
├── requirements.txt
└── README.md
```

## 3. Requirements

- Python 3.10+
- Git
- Historical stock datasets
- SCIP solver installation for mixed-integer optimization (PySCIPOpt)

## 4. Clone the Repository

```bash
git clone https://github.com/abhipullela/Optimization-Project.git
cd Optimization-Project
```

## 5. Environment Setup

### Windows (PowerShell)

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** PySCIPOpt requires a compatible SCIP installation. If its installation fails, install SCIP and configure the required libraries before installing PySCIPOpt.

## 6. Dataset Setup

Place the input datasets in `data/raw/`:

```text
data/
└── raw/
    ├── OPT_DATA.zip
    └── nifty500_stocks.csv
```

- `OPT_DATA.zip`: Primary historical stock data and NIFTY 50 benchmark data.
- `nifty500_stocks.csv`: Secondary stock dataset used for matching and missing-value enrichment.

Do not rename the files unless you also update the corresponding command-line paths.

## 7. Running the Project

Execute the following commands **in order from the repository root**, with the virtual environment activated.

### Step 1: Data Preprocessing

```bash
python -m src.data.preprocess --primary "data/raw/OPT_DATA.zip" --secondary "data/raw/nifty500_stocks.csv"
```

Generates:

- `cleaned_primary_master.csv`
- `data_quality_report.csv`
- `supplemental_dummy_live_replay.csv`

Location: `data/processed/`

### Step 2: Corporate Action Price Adjustment

```bash
python src/data/adjust_prices.py
```

Generates:

- `adjusted_primary_master.csv`

This dataset is used for feature engineering and historical risk estimation.

### Step 3: Extract NIFTY 50 Benchmark

```bash
python src/data/extract_benchmark.py
```

Generates:

- `nifty_benchmark.csv`

### Step 4: Feature Engineering

```bash
python src/features/build_features.py
```

Generates:

- `ml_features_primary.csv`

Contains engineered features and the next-observation return target.

### Step 5: Generate ML Predictions

```bash
python src/models/model2_ranker.py
```

Generates:

- `predictions_ranker.csv`

The model uses walk-forward training and cross-sectional target neutralization.

Optional arguments:

```bash
python src/models/model2_ranker.py --min-train-rows 1000 --retrain-freq 21
```

### Step 6: Evaluate ML Predictions

```bash
python src/models/evaluate_ranker.py
```

Reports:

- Mean Information Coefficient (IC)
- IC Information Ratio
- Hit Rate

### Step 7: Run Portfolio Backtesting

**Hard optimization (default):**

```bash
python src/backtesting/backtest.py --predictions=data/processed/predictions_ranker.csv --mode=hard --rebalance-freq=21
```

**Easy optimization:**

```bash
python src/backtesting/backtest.py --predictions=data/processed/predictions_ranker.csv --mode=easy --rebalance-freq=21
```

Available arguments:

| Argument | Description | Default |
|---|---|---|
| `--mode` | `easy` or `hard` optimization | `hard` |
| `--predictions` | Prediction CSV path | `predictions.csv` |
| `--historical-data` | Historical price data path | `adjusted_primary_master.csv` |
| `--benchmark` | Benchmark CSV path | `nifty_benchmark.csv` |
| `--rebalance-freq` | Trading days between rebalances | `21` |
| `--transaction-cost` | Transaction cost rate | `0.001` |
| `--max-turnover` | Maximum one-way turnover | `0.50` |

Generated outputs:

- `backtest_returns.csv`
- `backtest_metrics.json`

Location: `data/processed/`

### Step 8: Generate Performance Visualizations

```bash
python src/backtesting/plot_results.py
```

Generates:

- `cumulative_performance.png`

Location: `results/figures/`

## 8. Complete Execution Sequence

For convenience, execute the following commands sequentially from the repository root.

```bash
python -m src.data.preprocess --primary "data/raw/OPT_DATA.zip" --secondary "data/raw/nifty500_stocks.csv"

python src/data/adjust_prices.py

python src/data/extract_benchmark.py

python src/features/build_features.py

python src/models/model2_ranker.py

python src/models/evaluate_ranker.py

python src/backtesting/backtest.py --predictions=data/processed/predictions_ranker.csv --mode=hard --rebalance-freq=21

python src/backtesting/plot_results.py
```

## 9. Optimization Modes

| Feature | Easy | Hard |
|---|---|---|
| Mean-variance optimization | Yes | Yes |
| Long-only constraint | Yes | Yes |
| Maximum stock weight | Yes | Yes |
| Cardinality constraint | No | Yes |
| Turnover constraint | No | Yes |
| Transaction costs in objective | No | Yes |
| Optimization approach | Continuous | Mixed-integer |

The Hard formulation uses binary selection variables to limit the number of portfolio holdings.

## 10. Performance Metrics

The backtesting module evaluates:

- Cumulative Return
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Beta against NIFTY 50
- Annualized Alpha

Visualizations compare the cumulative growth of the optimized portfolio against the NIFTY 50 benchmark.

## 11. Important Notes

- Execute commands from the repository root.
- Run preprocessing before price adjustment and feature engineering.
- Feature engineering depends on `adjusted_primary_master.csv`.
- Backtesting uses the generated predictions and adjusted historical data.
- Raw and processed datasets are excluded from Git tracking.
- Corporate action adjustments currently depend on the events configured in `adjust_prices.py`; they are not automatically sourced from a complete corporate-action database.
- The XGBoost ranker's scaled predictions are model signals, not independently calibrated expected returns. Their suitability for mean-variance optimization must be evaluated.
- Backtest results are historical simulations and do not guarantee future performance.

## 12. License

This project is developed for academic and research purposes.
