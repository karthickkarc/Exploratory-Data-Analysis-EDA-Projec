Exploratory Data Analysis (EDA) Project

A structured, reproducible EDA pipeline that uncovers patterns and trends in any tabular dataset through statistical summaries, rich visualizations, correlation analysis, and automated insights.

---

Project Structure

```
eda-project/
├── data/                   ← Place your CSV / Excel / JSON files here
├── notebooks/
│   └── eda_notebook.ipynb  ← Interactive Jupyter walkthrough
├── src/
│   └── eda_analysis.py     ← Core analysis module
├── reports/
│   ├── eda_report.md       ← Auto-generated Markdown report (after run)
│   └── figures/            ← All charts saved here (after run)
├── tests/
│   └── test_eda.py         ← pytest unit tests
├── requirements.txt
└── README.md
```

---

Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run with built-in sample data
```bash
cd src
python eda_analysis.py
```

### 3. Run with your own dataset
```bash
python src/eda_analysis.py data/your_file.csv
```
Supported formats: `.csv`, `.xlsx`, `.xls`, `.json`

### 4. Interactive Jupyter notebook
```bash
jupyter lab notebooks/eda_notebook.ipynb
```

### 5. Run tests
```bash
pytest tests/ -v
```

---

What Gets Generated

| Output | Location | Description |
|--------|----------|-------------|
| Distributions | `reports/figures/01_distributions.png` | Histograms + KDE for all numeric columns |
| Correlation Heatmap | `reports/figures/02_correlation_heatmap.png` | Pearson r matrix |
| Categorical Counts | `reports/figures/03_categorical_counts.png` | Bar charts per category |
| Scatter Matrix | `reports/figures/04_scatter_matrix.png` | Pair-plot for top 4 numeric cols |
| Box-plots | `reports/figures/05_boxplots_by_category.png` | Distribution split by category |
| EDA Report | `reports/eda_report.md` | Full Markdown report with insights |

---
Key Features

- **Statistical summaries** — descriptive stats, missing value audit, skewness
- **Correlation analysis** — Pearson heatmap, top correlated pairs
- **Outlier detection** — IQR-based flagging
- **Category insights** — imbalance detection, value distributions
- **Auto-generated report** — Markdown report ready to share or convert to PDF
- **Modular design** — import any function in your own notebooks or scripts
- **Plug-and-play** — works with any CSV/Excel/JSON, no config needed

---
Extending the Project

| Task | How |
|------|-----|
| Add a new chart | Add a `plot_*` function in `src/eda_analysis.py` and call it in `run_eda()` |
| Use your own data | Drop a file in `data/` and pass the path to `run_eda()` |
| Export report as PDF | `pandoc reports/eda_report.md -o reports/eda_report.pdf` |
| Add feature engineering | Create `src/feature_engineering.py` and import it in the notebook |

---

Learning Goals

After completing this project you will be able to:

1. Load and audit a real-world dataset for quality issues
2. Compute and interpret descriptive statistics
3. Build publication-quality charts with Matplotlib & Seaborn
4. Identify correlated features and potential drivers
5. Communicate findings in a structured, written report

---

🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-analysis`
3. Commit changes: `git commit -m "Add time-series analysis"`
4. Push and open a Pull Request

---
📄 License

MIT — free to use, modify, and share.
