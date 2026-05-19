"""
Exploratory Data Analysis (EDA) - Core Analysis Module
=======================================================
Performs statistical summaries, visualizations, correlation analysis,
and generates structured insights from a dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
import os

warnings.filterwarnings("ignore")

# ── Styling ──────────────────────────────────────────────────────────────────
PALETTE   = ["#2D6A4F", "#40916C", "#52B788", "#74C69D", "#95D5B2", "#B7E4C7"]
ACCENT    = "#FF6B35"
BG_COLOR  = "#F8F9FA"
TEXT_COLOR = "#1A1A2E"

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    BG_COLOR,
    "axes.edgecolor":    "#CCCCCC",
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV, Excel, or JSON dataset."""
    ext = Path(filepath).suffix.lower()
    loaders = {".csv": pd.read_csv, ".xlsx": pd.read_excel,
               ".xls": pd.read_excel, ".json": pd.read_json}
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")
    df = loaders[ext](filepath)
    print(f"✔ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns from '{filepath}'")
    return df


def load_sample_data() -> pd.DataFrame:
    """Generate a rich sample dataset when no file is provided."""
    np.random.seed(42)
    n = 500

    age        = np.random.normal(35, 10, n).clip(18, 70).astype(int)
    experience = (age - 18 + np.random.normal(0, 2, n)).clip(0, 45).astype(int)
    education  = np.random.choice(
        ["High School", "Bachelor's", "Master's", "PhD"],
        n, p=[0.20, 0.45, 0.25, 0.10]
    )
    edu_bonus  = {"High School": 0, "Bachelor's": 8000,
                  "Master's": 15000, "PhD": 25000}
    department = np.random.choice(
        ["Engineering", "Marketing", "Sales", "HR", "Finance"],
        n, p=[0.30, 0.20, 0.25, 0.10, 0.15]
    )
    dept_bonus = {"Engineering": 20000, "Marketing": 5000,
                  "Sales": 8000, "HR": 2000, "Finance": 12000}

    salary = (
        30000
        + age * 800
        + experience * 600
        + np.array([edu_bonus[e] for e in education])
        + np.array([dept_bonus[d] for d in department])
        + np.random.normal(0, 5000, n)
    ).clip(25000, 150000).astype(int)

    performance = (
        50
        + experience * 0.8
        + np.random.normal(0, 10, n)
    ).clip(1, 100).round(1)

    satisfaction = (
        3
        + (salary - salary.mean()) / salary.std() * 0.5
        + np.random.normal(0, 0.8, n)
    ).clip(1, 5).round(1)

    df = pd.DataFrame({
        "Age":          age,
        "Experience":   experience,
        "Education":    education,
        "Department":   department,
        "Salary":       salary,
        "Performance":  performance,
        "Satisfaction": satisfaction,
        "Remote":       np.random.choice(["Yes", "No"], n, p=[0.4, 0.6]),
    })

    # Inject 2% missing values
    for col in ["Salary", "Performance", "Satisfaction"]:
        mask = np.random.random(n) < 0.02
        df.loc[mask, col] = np.nan

    print(f"✔ Sample dataset created ({n} rows, {df.shape[1]} columns)")
    return df


# ── Statistical Summary ───────────────────────────────────────────────────────
def statistical_summary(df: pd.DataFrame) -> dict:
    """Compute and display comprehensive statistics."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    print("\n" + "═" * 60)
    print("  STATISTICAL SUMMARY")
    print("═" * 60)
    print(f"  Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
    print(f"  Numeric: {len(num_cols)}  |  Categorical: {len(cat_cols)}")

    missing = df.isnull().sum()
    if missing.any():
        print("\n  Missing Values:")
        for col, cnt in missing[missing > 0].items():
            print(f"    {col}: {cnt} ({cnt/len(df)*100:.1f}%)")

    print("\n  Numeric Columns:")
    desc = df[num_cols].describe().round(2)
    print(desc.to_string())

    print("\n  Categorical Columns:")
    for col in cat_cols:
        vc = df[col].value_counts()
        print(f"\n  {col}:")
        for val, cnt in vc.items():
            print(f"    {val}: {cnt} ({cnt/len(df)*100:.1f}%)")

    return {"numeric": num_cols, "categorical": cat_cols, "missing": missing}


# ── Visualizations ────────────────────────────────────────────────────────────
def plot_distributions(df: pd.DataFrame, num_cols: list):
    """Histograms + KDE for numeric columns."""
    n = len(num_cols)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    fig.suptitle("Distribution of Numeric Variables", fontsize=16,
                 fontweight="bold", y=1.01)
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=30, color=PALETTE[i % len(PALETTE)],
                     edgecolor="white", linewidth=0.5, alpha=0.85, density=True)
        kde_x = np.linspace(data.min(), data.max(), 200)
        kde   = stats.gaussian_kde(data)
        axes[i].plot(kde_x, kde(kde_x), color=ACCENT, linewidth=2)
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "01_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved {path}")


def plot_correlation_heatmap(df: pd.DataFrame, num_cols: list):
    """Annotated correlation heatmap."""
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", linewidths=0.5,
        cmap=sns.diverging_palette(150, 10, as_cmap=True),
        vmin=-1, vmax=1, center=0, ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Correlation Heatmap", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    path = OUTPUT_DIR / "02_correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved {path}")
    return corr


def plot_categorical_counts(df: pd.DataFrame, cat_cols: list):
    """Bar charts for categorical variables."""
    n = len(cat_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Categorical Variable Distributions", fontsize=16,
                 fontweight="bold")

    for ax, col in zip(axes, cat_cols):
        vc = df[col].value_counts()
        bars = ax.bar(vc.index, vc.values, color=PALETTE[:len(vc)],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vc.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + vc.max() * 0.01,
                    str(val), ha="center", va="bottom", fontsize=9)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = OUTPUT_DIR / "03_categorical_counts.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved {path}")


def plot_scatter_matrix(df: pd.DataFrame, num_cols: list):
    """Pair-plot scatter matrix for the top 4 numeric columns."""
    cols = num_cols[:4]
    cat_col = df.select_dtypes(exclude=np.number).columns
    hue = cat_col[0] if len(cat_col) > 0 else None

    g = sns.pairplot(
        df[cols + ([hue] if hue else [])].dropna(),
        hue=hue, diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 20},
        palette=PALETTE[:df[hue].nunique()] if hue else None
    )
    g.figure.suptitle("Scatter Matrix (Top 4 Numeric Variables)",
                       y=1.01, fontsize=14, fontweight="bold")
    path = OUTPUT_DIR / "04_scatter_matrix.png"
    g.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved {path}")


def plot_boxplots_by_category(df: pd.DataFrame, num_cols: list, cat_cols: list):
    """Box-plots of numeric variables grouped by first categorical column."""
    if not cat_cols:
        return
    cat = cat_cols[0]
    cols_to_plot = num_cols[:3]

    fig, axes = plt.subplots(1, len(cols_to_plot),
                             figsize=(6 * len(cols_to_plot), 5))
    if len(cols_to_plot) == 1:
        axes = [axes]
    fig.suptitle(f"Distribution by {cat}", fontsize=16, fontweight="bold")

    for ax, col in zip(axes, cols_to_plot):
        groups = [df[df[cat] == g][col].dropna()
                  for g in df[cat].unique()]
        labels = df[cat].unique()
        bp = ax.boxplot(groups, labels=labels, patch_artist=True,
                        medianprops={"color": ACCENT, "linewidth": 2})
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel(cat)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = OUTPUT_DIR / "05_boxplots_by_category.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved {path}")


# ── Insights ──────────────────────────────────────────────────────────────────
def extract_insights(df: pd.DataFrame, corr: pd.DataFrame,
                     num_cols: list, cat_cols: list) -> list[str]:
    """Derive actionable insights from statistics and correlations."""
    insights = []

    # Missing data
    missing_pct = df.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 5]
    if not high_missing.empty:
        insights.append(
            f"⚠ High missing data in: {', '.join(high_missing.index)} "
            f"({high_missing.max():.1f}% max) — consider imputation."
        )

    # Strong correlations
    if len(num_cols) > 1:
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                r = corr.iloc[i, j]
                if abs(r) >= 0.5:
                    pairs.append((num_cols[i], num_cols[j], r))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for a, b, r in pairs[:3]:
            direction = "positive" if r > 0 else "negative"
            insights.append(
                f"📈 Strong {direction} correlation between '{a}' and '{b}' "
                f"(r = {r:.2f})."
            )

    # Skewness
    for col in num_cols:
        sk = df[col].dropna().skew()
        if abs(sk) > 1:
            direction = "right (positive)" if sk > 0 else "left (negative)"
            insights.append(
                f"📊 '{col}' is heavily skewed {direction} (skew = {sk:.2f}) "
                f"— consider log transformation."
            )

    # Outliers via IQR
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr   = q3 - q1
        n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        if n_out > 0:
            pct = n_out / len(df) * 100
            insights.append(
                f"🔍 '{col}' has {n_out} outliers ({pct:.1f}%) beyond 1.5× IQR."
            )

    # Category imbalance
    for col in cat_cols:
        vc     = df[col].value_counts(normalize=True)
        top_pct = vc.iloc[0] * 100
        if top_pct > 60:
            insights.append(
                f"⚖ '{col}' is imbalanced: '{vc.index[0]}' represents "
                f"{top_pct:.1f}% of records."
            )

    return insights


# ── Report ────────────────────────────────────────────────────────────────────
def generate_report(df: pd.DataFrame, insights: list[str],
                    num_cols: list, cat_cols: list):
    """Write a Markdown EDA report."""
    desc   = df[num_cols].describe().round(2)
    report = f"""# Exploratory Data Analysis Report

## 1. Dataset Overview
| Attribute | Value |
|-----------|-------|
| Rows | {df.shape[0]:,} |
| Columns | {df.shape[1]} |
| Numeric columns | {len(num_cols)} |
| Categorical columns | {len(cat_cols)} |
| Missing values | {df.isnull().sum().sum()} |

## 2. Statistical Summary

### Numeric Variables
{desc.to_markdown()}

### Categorical Variables
"""
    for col in cat_cols:
        vc = df[col].value_counts()
        report += f"\n**{col}**\n"
        for val, cnt in vc.items():
            report += f"- {val}: {cnt} ({cnt/len(df)*100:.1f}%)\n"

    report += "\n## 3. Key Insights\n"
    for ins in insights:
        report += f"- {ins}\n"

    report += """
## 4. Visualizations

| # | Chart | Description |
|---|-------|-------------|
| 1 | `reports/figures/01_distributions.png` | Histograms + KDE for all numeric columns |
| 2 | `reports/figures/02_correlation_heatmap.png` | Pearson correlation matrix |
| 3 | `reports/figures/03_categorical_counts.png` | Value counts for categorical columns |
| 4 | `reports/figures/04_scatter_matrix.png` | Pair-plot scatter matrix |
| 5 | `reports/figures/05_boxplots_by_category.png` | Distribution by category |

## 5. Recommendations

1. **Missing Data** – Impute or drop rows depending on volume and importance.
2. **Skewed Features** – Apply log/Box-Cox transforms before modeling.
3. **Outlier Handling** – Investigate outliers; cap or remove if non-representative.
4. **Feature Engineering** – Leverage strong correlations to build composite features.
5. **Encoding** – One-hot or ordinal encode categorical variables before ML pipelines.

---
*Report generated by `src/eda_analysis.py`*
"""
    path = Path("reports/eda_report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    print(f"  ✔ Report saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_eda(filepath: str = None):
    print("\n🔬 Starting Exploratory Data Analysis…\n")

    df = load_data(filepath) if filepath else load_sample_data()

    info = statistical_summary(df)
    num_cols = info["numeric"]
    cat_cols = info["categorical"]

    print("\n📊 Generating visualizations…")
    plot_distributions(df, num_cols)
    corr = plot_correlation_heatmap(df, num_cols)
    if cat_cols:
        plot_categorical_counts(df, cat_cols)
    plot_scatter_matrix(df, num_cols)
    plot_boxplots_by_category(df, num_cols, cat_cols)

    print("\n💡 Extracting insights…")
    insights = extract_insights(df, corr, num_cols, cat_cols)
    for ins in insights:
        print(f"  {ins}")

    print("\n📝 Writing report…")
    generate_report(df, insights, num_cols, cat_cols)

    print("\n✅ EDA complete! Check the 'reports/' directory.\n")
    return df, insights


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    run_eda(filepath)
