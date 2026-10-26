Optimization-Project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_return_analysis.ipynb
│   ├── 03_ml_model.ipynb
│   ├── 04_baseline_optimization.ipynb
│   ├── 05_constrained_optimization.ipynb
│   └── 06_results_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   └── return_prediction.py
│   │
│   ├── optimization/
│   │   ├── mean_variance.py
│   │   ├── constrained_portfolio.py
│   │   └── constraints.py
│   │
│   ├── backtesting/
│   │   ├── backtest.py
│   │   └── metrics.py
│   │
│   └── utils/
│       └── config.py
│
├── tests/
│   ├── test_data.py
│   ├── test_mean_variance.py
│   └── test_constraints.py
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── README.md
│
├── reports/
│   └── project_report.pdf
│
└── presentation/
    └── final_presentation.pptx
