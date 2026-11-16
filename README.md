Run commands - 
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.preprocess --primary "data/raw/OPT_DATA.zip" --secondary "data/raw/nifty500_stocks.csv"
python src/data/adjust_prices.py
python src/data/extract_benchmark.py

python src/features/build_features.py
python src/models/model2_ranker.py
python src/models/evaluate_ranker.py

python src/backtesting/backtest.py --predictions=data/processed/predictions_ranker.csv --mode=hard --rebalance-freq=21
python src/backtesting/plot_results.py