# 🏠 Real Estate Scout Optimizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Investment analysis tool for real estate opportunities using data-driven scoring algorithms.**

🌐 **[Try Live Demo](https://real-estate-scout-optimizer.streamlit.app/)**
---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Academic Context](#academic-context)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

The **Real Estate Scout Optimizer** is a comprehensive web application that analyzes housing market data to identify optimal investment opportunities. It implements a proprietary **Smart Scout Score** algorithm that combines financial metrics, structural quality, and vintage value to rank properties by investment potential.

**Key Innovation:** All detection thresholds are data-driven and automatically calibrated to the specific market, making the algorithm portable to any geographic region without manual recalibration.

---

## ✨ Key Features

### ☁️ **Dynamic Intelligence Dashboard** (New)
- **Live Cloud Link**: Connect directly to Google Drive datasets via URL.
- **Magic Links**: The app updates the browser URL with your data source, creating **bookmarkable dashboards** that auto-update when the source CSV changes.
- **No Database Required**: State persistence handled via URL query parameters.

### 📊 **Smart Analysis**
- Market distribution analysis with statistical validation
- Chi-Square independence tests
- ANOVA group comparisons
- Outlier detection using skewness/kurtosis metrics
- Correlation analysis (Pearson vs Spearman)

### 🎯 **Data-Driven Strategy**
- **Auto-detection** of optimal property conditions
- **Auto-detection** of optimal building grades
- **Auto-detection** of optimal construction decades
- Efficiency trap filtering (removes overpriced properties)
- Zipcode-aware imputation (zero data loss)

### 📈 **Interactive Visualizations**
- Market distribution plots
- Size trap analysis with marginal distributions
- Correlation heatmaps
- Strategic zones scatter plot (interactive Plotly version)
- Interactive geographic map with clustered markers

### 📥 **Actionable Outputs**
- Prioritized action list (CSV export)
- Full dataset with Smart Scores (Power BI ready)
- Data quality audit report
- Top 10 investment opportunities
- Strategic zipcode rankings

---

## 🧠 Methodology

The **Smart Scout Score** algorithm follows a 6-step process:

1. **Data Cleaning**: Zipcode-aware median imputation (no rows dropped)
2. **Market Context**: Calculate price gaps, efficiency metrics, density types
3. **Statistical Validation**: Chi-Square and ANOVA tests confirm strategy validity
4. **Parameter Auto-Detection**: Identify optimal conditions, grades, and decades from data
5. **Efficiency Gate**: Filter properties with negative $/sqft gaps (size traps)
6. **Weighted Scoring**: 
   - 80% Financial (price gap potential)
   - 10% Structural (optimal grade bonus)
   - 10% Vintage (optimal decade bonus)

**Result:** Properties ranked 0-100, where higher scores indicate better investment opportunities.

---

## 🛠️ Technology Stack

### **Backend & Analysis**
- **Python 3.10+** - Core language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **SciPy** - Statistical tests (Chi-Square, ANOVA, f_oneway)

### **Visualization**
- **Matplotlib** - Static plots
- **Seaborn** - Statistical visualizations
- **Plotly** - Interactive charts
- **Folium** - Interactive maps with clustering

### **Web Framework**
- **Streamlit** - Web application framework
- **Streamlit Cloud** - Deployment platform

### **Development Tools**
- **UV** - Fast Python package manager (10-100x faster than pip)
- **PyCharm** - IDE
- **Git/GitHub** - Version control

---

## 🚀 Installation

### **Prerequisites**
- Python 3.10 or higher
- UV package manager (recommended) or pip

### **Clone Repository**
```bash
git clone https://github.com/CarloCogni/real-estate-scout-optimizer.git
cd real-estate-scout-optimizer
```

### **Install Dependencies**

**Option A: Using UV (Recommended)**
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# or
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Install dependencies
uv sync
```

**Option B: Using pip**
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### **Run Locally**
```bash
# Using UV
uv run streamlit run app.py

# Using pip
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### **Using the App**

1. **Choose Data Source**:
   - **Demo Dataset**: Use the pre-loaded KC Housing data to explore features.
   - **Upload CSV**: Upload a local file for ad-hoc analysis.
   - **☁️ Connect to Cloud**: Paste a Google Drive "Share" link (set to *Anyone with the link*).

2. **Bookmark Your Dashboard**:
   - When using the Cloud Link, the app URL updates automatically.
   - **Save this URL**. Opening it later will reload the live data from Google Drive automatically.

3. **Set Parameters**:
   - Adjust maximum acquisition budget ($100K-$500K).

4. **Run Analysis**:
   - Click "🚀 Run Complete Analysis".
   - Navigate through the analysis tabs and download reports.

---

## 📁 Project Structure
```
real-estate-scout-optimizer/
├── .streamlit/
│   └── config.toml              # Streamlit theme configuration
├── data/
│   └── [demo_dataset].csv       # Demo housing data
├── modules/
│   ├── __init__.py
│   ├── utils.py                 # Formatting utilities
│   ├── data_prep.py             # Data cleaning & context
│   ├── data_validation.py       # Dataset validation
│   ├── analysis.py              # Statistical analysis
│   ├── scoring.py               # Smart Score algorithm
│   ├── visualization.py         # Matplotlib plots
│   └── visualization_plotly.py  # Interactive Plotly charts
├── app.py                       # Main Streamlit application
├── pyproject.toml               # Project dependencies (UV)
├── requirements.txt             # Project dependencies (pip)
├── uv.lock                      # Dependency lock file
├── .gitignore
├── LICENSE
└── README.md
```
---

## 🎓 Academic Context

**Developed for:**
- **Institution**: ZIGURAT Global Institute of Technology
- **Program**: Master's in Artificial Intelligence for Architecture & Construction
- **Module**: M3U1&U2 - Machine Learning & Data Analysis
- **Assignment**: Real Estate Investment Optimization

**Learning Objectives Demonstrated:**
- Data preprocessing and cleaning strategies
- Statistical hypothesis testing (Chi-Square, ANOVA)
- Algorithm design and validation
- Web application development
- Data visualization best practices
- Geospatial analysis

**Group**: FMP Group 4

---

## 📜 License & Commercial Use

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

**What this means:**
- ✅ You can use, modify, and distribute this software
- ✅ You must share your modifications under the same license
- ✅ **Web deployment requires source code disclosure** (key difference from GPL)
- ✅ Perfect for academic and open-source projects

**For commercial use or alternative licensing**, please contact: carlo.cogni@protonmail.com

This license ensures the project remains open-source while protecting academic work.

See [LICENSE](LICENSE) file for full terms.

---

## 👤 Contact

**Carlo Cogni**
- 📧 Email: carlo.cogni@protonmail.com
- 💼 LinkedIn: Carlo Cogni
- 🐙 GitHub: [@CarloCogni](https://github.com/CarloCogni)

**Questions?** Open an issue or reach out directly!

---

## 🙏 Acknowledgments

- **ZIGURAT Institute of Technology** - For providing the academic framework and dataset
- **Streamlit** - For the excellent web framework
- **Open Source Community** - For the amazing Python libraries

---
*Built with ❤️ using Python, Streamlit, and data science best practices. Cheers!*
```

                    GNU AFFERO GENERAL PUBLIC LICENSE
                       Version 3, 19 November 2007

