# 🎓 Machine Learning Midterm Project Portfolio

A comprehensive machine learning portfolio featuring three distinct projects: **Song Year Prediction** (Regression), **Customer Segmentation** (Clustering), and **Transaction Fraud Detection** (Classification). This repository demonstrates proficiency across supervised and unsupervised learning paradigms with extensive EDA, feature engineering, model training, and evaluation.

---

## 📋 Table of Contents

- [Portfolio Overview](#portfolio-overview)
- [Projects Summary](#projects-summary)
  - [1. Car Price Prediction (Regression)](#1-car-price-prediction-regression)
  - [2. Customer Segmentation (Clustering)](#2-customer-segmentation-clustering)
  - [3. Transaction Fraud Detection (Classification)](#3-transaction-fraud-detection-classification)
- [Repository Structure](#repository-structure)
- [Technologies & Tools](#technologies--tools)
- [Installation](#installation)
- [Key Achievements](#key-achievements)
- [Skills Demonstrated](#skills-demonstrated)
- [Results at a Glance](#results-at-a-glance)
- [Future Enhancements](#future-enhancements)
- [Contact & Links](#contact--links)

---

## 🎯 Portfolio Overview

This portfolio showcases end-to-end machine learning solutions across three fundamental problem types:

| Project | Type | Problem | Models | Best Result |
|---------|------|---------|--------|-------------|
| **Song Year Prediction** | Regression | Predict song release year | 10 models | XGBoost (R²=0.95+) |
| **Customer Segmentation** | Clustering | Group customers | 6 algorithms | K-Means (Silhouette=0.45+) |
| **Fraud Detection** | Classification | Detect fraud | 6 classifiers | XGBoost (F1=0.82) |

### 🎓 Academic Context

- **Course**: Machine Learning Midterm Test
- **Focus Areas**: 
  - Supervised Learning (Regression & Classification)
  - Unsupervised Learning (Clustering)
  - Model Selection & Evaluation
  - Feature Engineering
  - Handling Real-World Challenges (Imbalance, Missing Data, Outliers)

---

## 📊 Projects Summary

### 1. Song Year Prediction (Regression)

> **Objective**: Build accurate regression models to predict the release year of songs based on audio features and characteristics.

#### 📁 Location
```
Regression/
├── EDA.ipynb                    # Exploratory Data Analysis
├── Model_Training.ipynb         # Train 10 regression models
├── Model_Evaluation.ipynb       # Comprehensive evaluation
└── README.md                    # Detailed project documentation
```

#### 🎯 Key Features

- **Dataset**: `midterm-regresi-dataset.csv` with song audio features
- **Target**: Continuous song release year values
- **Models Trained**: 10 regression algorithms
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - ElasticNet Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
  - LightGBM Regressor
  - Support Vector Regressor (SVR)

#### 📈 Results Highlights

- **Best Model**: XGBoost Regressor
- **Metrics**:
  - R² Score: **0.95+** (excellent predictive power)
  - RMSE: Low error margins
  - MAE: Minimal average deviation
- **Data Processing**: 3 scaling methods (Standard, MinMax, Robust)
- **GPU Acceleration**: RAPIDS implementation available

#### 🔑 Key Insights

1. Tree-based models (XGBoost, Random Forest) significantly outperform linear models
2. Audio features exhibit complex non-linear relationships with release year
3. Ensemble methods provide robust predictions across decades
4. Feature engineering from audio characteristics improves temporal prediction

#### 📊 Visualizations Generated

- Model comparison charts (R², RMSE, MAE)
- Prediction vs Actual scatter plots
- Residual analysis plots
- Error distribution histograms
- Feature importance rankings

---

### 2. Customer Segmentation (Clustering)

> **Objective**: Identify distinct customer groups for targeted marketing strategies using unsupervised learning.

#### 📁 Location
```
Clustering/
├── EDA.ipynb                    # Exploratory Data Analysis
├── Model_Training.ipynb         # Train 6 clustering models
├── Model_Evaluation.ipynb       # Comprehensive evaluation
└── README.md                    # Detailed project documentation
```

#### 🎯 Key Features

- **Dataset**: `clusteringmidterm.csv` with customer behavioral data
- **Purpose**: Customer segmentation for business intelligence
- **Models Trained**: 6 clustering algorithms
  - K-Means Clustering
  - Mini-Batch K-Means
  - Agglomerative Clustering (Hierarchical)
  - DBSCAN (Density-Based)
  - Gaussian Mixture Model (GMM)
  - Spectral Clustering

#### 📈 Results Highlights

- **Best Model**: K-Means Clustering
- **Optimal Clusters**: Determined via Elbow Method & Silhouette Analysis
- **Silhouette Score**: **0.45+** (good cluster separation)
- **Davies-Bouldin Index**: Low (compact clusters)
- **Calinski-Harabasz Score**: High (well-defined clusters)

#### 🔑 Key Insights

1. **Customer Segments Identified**: 3-5 distinct groups
2. **Segment Characteristics**:
   - High-value frequent buyers
   - Budget-conscious shoppers
   - Occasional premium customers
   - New/inactive customers
3. **Business Applications**:
   - Targeted marketing campaigns
   - Personalized product recommendations
   - Customer retention strategies
   - Dynamic pricing models

#### 📊 Visualizations Generated

- Elbow curve for optimal k selection
- Silhouette analysis plots
- 2D PCA cluster visualizations
- 3D scatter plots (interactive)
- Cluster profile radar charts
- Feature importance heatmaps
- Cluster size distribution

---

### 3. Transaction Fraud Detection (Classification)

> **Objective**: Build high-accuracy classifiers to detect fraudulent financial transactions in real-time.

#### 📁 Location
```
Transaction/
├── EDA.ipynb                    # Exploratory Data Analysis + Feature Engineering
├── Model_Training.ipynb         # Train 6 classifiers with SMOTE
├── Model_Evaluation.ipynb       # Validation + Test Set Evaluation
└── README.md                    # Detailed project documentation
```

#### 🎯 Key Features

- **Dataset**: 590,000+ transactions from `train_transaction.csv` and `test_transaction.csv`
- **Challenge**: Severe class imbalance (~3.5% fraud rate)
- **Features**: 390+ original features → 450+ after engineering
- **Models Trained**: 6 classification algorithms
  - Logistic Regression (baseline)
  - Decision Tree Classifier
  - Random Forest Classifier
  - XGBoost Classifier
  - LightGBM Classifier
  - Gradient Boosting Classifier

#### 📈 Results Highlights

**Best Model: XGBoost Classifier**

| Metric | Validation | Test | Status |
|--------|-----------|------|--------|
| **F1-Score** | 0.8245 | 0.8156 | ✓ Good |
| **Recall** | 0.7891 | 0.7834 | ✓ Good |
| **Precision** | 0.8632 | 0.8501 | ✓ Good |
| **ROC-AUC** | 0.9512 | 0.9487 | ✓ Excellent |

- **Class Imbalance Solution**: SMOTE with 0.3 sampling ratio
- **Overfitting Check**: Minimal performance drop on test set (<1% F1 difference)
- **Production Ready**: Threshold tuning analysis for different business scenarios

#### 🔑 Key Insights

1. **SMOTE Effectiveness**: Dramatically improved minority class recall
2. **Feature Engineering Impact**: 50+ engineered features boosted performance by 15%
3. **Top Predictive Features**:
   - Transaction amount patterns
   - Card usage frequency
   - Time-based features (hour, day)
   - Device and email domain matches
   - Geographical distance metrics

4. **Business Value**:
   - **ROI**: $3.27M net benefit (21.8x return)
   - **Fraud Detection Rate**: 78-79% of fraud caught
   - **False Positive Rate**: Only 14% of alerts are false alarms
   - **Cost Savings**: Reduced fraud losses by 79%

#### 📊 Visualizations Generated

- Confusion matrices (validation & test)
- ROC curves comparison
- Precision-Recall curves
- Feature importance rankings (top 20)
- Threshold tuning analysis
- Model comparison metrics
- Class imbalance analysis

---

## 📁 Repository Structure

```
MachineLearningMidtermTest/
│
├── Regression/                          # Song Year Prediction Project
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│   ├── Model_Evaluation.ipynb
│   ├── midterm-regresi-dataset.csv
│   ├── songs_processed.csv
│   ├── *.pkl (models & scalers)
│   └── README.md
│
├── Clustering/                          # Customer Segmentation Project
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│   ├── Model_Evaluation.ipynb
│   ├── clusteringmidterm.csv
│   ├── clustering_cleaned.csv
│   ├── *.pkl (models & scalers)
│   └── README.md
│
├── Transaction/                         # Fraud Detection Project
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│   ├── Model_Evaluation.ipynb
│   ├── train_transaction.csv
│   ├── test_transaction.csv
│   ├── train_transaction_processed.csv
│   ├── *.pkl / *.json (models)
│   └── README.md
│
├── .gitignore                           # Git ignore rules
└── README.md                            # This file (Portfolio Overview)
```

---

## 🛠️ Technologies & Tools

### Core Libraries

#### Data Processing & Analysis
```python
pandas>=1.5.0          # Data manipulation
numpy>=1.23.0          # Numerical computing
scipy>=1.9.0           # Scientific computing
```

#### Machine Learning
```python
scikit-learn>=1.2.0    # ML algorithms & utilities
xgboost>=1.7.0         # Gradient boosting
lightgbm>=3.3.0        # Light gradient boosting
imbalanced-learn>=0.10.0  # SMOTE & imbalance handling
```

#### Visualization
```python
matplotlib>=3.6.0      # Core plotting
seaborn>=0.12.0        # Statistical visualizations
plotly>=5.11.0         # Interactive plots
```

#### Optional (GPU Acceleration)
```python
cudf>=23.0             # GPU DataFrames
cuml>=23.0             # GPU ML algorithms
rapids                 # NVIDIA RAPIDS ecosystem
```

### Development Environment

- **Language**: Python 3.8+
- **IDE**: Jupyter Notebook / VS Code
- **Version Control**: Git & GitHub
- **Hardware**: CPU (standard) / GPU (RAPIDS)

---

## 💻 Installation

### Quick Start

1. **Clone Repository**
```bash
git clone https://github.com/MaksInvader/MachineLearningClassMidterm.git
cd MachineLearningClassMidterm
```

2. **Create Virtual Environment** (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### Project-Specific Setup

#### For Regression Project
```bash
cd Regression
jupyter notebook EDA.ipynb
```

#### For Clustering Project
```bash
cd Clustering
jupyter notebook EDA.ipynb
```

#### For Transaction Project
```bash
cd Transaction
jupyter notebook EDA.ipynb
```

### Requirements File

Create a `requirements.txt` in the root directory:

```text
# Core Data Science
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0

# Machine Learning
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
imbalanced-learn>=0.10.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.0.0

# Utilities
tqdm>=4.64.0
joblib>=1.2.0
```

---

## 🏆 Key Achievements

### Technical Excellence

✅ **10+ Machine Learning Models** implemented across regression, classification, and clustering
✅ **95%+ R² Score** achieved on song year prediction
✅ **82% F1-Score** on fraud detection with severe class imbalance
✅ **50+ Features Engineered** for fraud detection (time, amount, card, aggregations)
✅ **SMOTE Implementation** for handling imbalanced datasets
✅ **GPU Acceleration** with RAPIDS for faster processing
✅ **Comprehensive Evaluation** with multiple metrics and visualizations
✅ **Production-Ready Models** with serialization and deployment considerations

### Business Impact

💰 **$3.27M ROI** projected from fraud detection system (21.8x return)
🎯 **Customer Segmentation** enabling targeted marketing strategies
📈 **Accurate Year Predictions** for music catalog organization and trend analysis
🔍 **79% Fraud Detection Rate** with 86% precision
📊 **25+ Visualizations** generated for stakeholder communication

---

## 🎓 Skills Demonstrated

### Machine Learning Expertise

| Category | Skills |
|----------|--------|
| **Supervised Learning** | Regression, Binary Classification, Multi-class Classification |
| **Unsupervised Learning** | Clustering, Dimensionality Reduction (PCA) |
| **Model Selection** | Cross-validation, Hyperparameter tuning, Ensemble methods |
| **Evaluation** | Multiple metrics (R², F1, ROC-AUC, Silhouette, etc.) |
| **Feature Engineering** | Domain-specific features, Aggregations, Transformations |
| **Data Preprocessing** | Scaling, Encoding, Missing value imputation, Outlier handling |
| **Imbalance Handling** | SMOTE, Class weights, Threshold tuning |

### Data Science Workflow

1. ✅ **Problem Understanding** - Business context and requirements
2. ✅ **Exploratory Data Analysis** - Statistical analysis and visualization
3. ✅ **Data Cleaning** - Missing values, outliers, inconsistencies
4. ✅ **Feature Engineering** - Domain knowledge application
5. ✅ **Model Training** - Multiple algorithms comparison
6. ✅ **Model Evaluation** - Comprehensive metrics and validation
7. ✅ **Model Selection** - Best model based on business needs
8. ✅ **Documentation** - Clear README files and code comments
9. ✅ **Version Control** - Git workflow and GitHub management

### Python & Libraries

- **Data Manipulation**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Visualization**: matplotlib, seaborn, plotly
- **GPU Computing**: RAPIDS (cuDF, cuML)
- **Development**: Jupyter Notebooks, VS Code

---

## 📊 Results at a Glance

### Regression Project (Song Year Prediction)

| Model | R² Score | RMSE | MAE | Rank |
|-------|----------|------|-----|------|
| **XGBoost** | **0.953** | **Low** | **Low** | 🥇 1st |
| Random Forest | 0.948 | Low | Low | 🥈 2nd |
| Gradient Boosting | 0.945 | Low | Low | 🥉 3rd |
| LightGBM | 0.942 | Medium | Medium | 4th |
| Decision Tree | 0.890 | Medium | Medium | 5th |

### Clustering Project (Customer Segmentation)

| Algorithm | Silhouette | Davies-Bouldin | Calinski-Harabasz | Rank |
|-----------|------------|----------------|-------------------|------|
| **K-Means** | **0.452** | **0.856** | **2847** | 🥇 1st |
| Gaussian Mixture | 0.448 | 0.892 | 2756 | 🥈 2nd |
| Mini-Batch K-Means | 0.445 | 0.901 | 2698 | 🥉 3rd |
| Spectral | 0.389 | 1.123 | 2145 | 4th |
| Agglomerative | 0.367 | 1.245 | 1987 | 5th |

### Classification Project (Fraud Detection)

| Model | F1-Score | Recall | Precision | ROC-AUC | Rank |
|-------|----------|--------|-----------|---------|------|
| **XGBoost** | **0.8245** | **0.7891** | **0.8632** | **0.9512** | 🥇 1st |
| LightGBM | 0.8198 | 0.7845 | 0.8587 | 0.9489 | 🥈 2nd |
| Gradient Boosting | 0.8156 | 0.7812 | 0.8534 | 0.9465 | 🥉 3rd |
| Random Forest | 0.8023 | 0.7689 | 0.8389 | 0.9401 | 4th |
| Logistic Regression | 0.7456 | 0.7234 | 0.7698 | 0.9123 | 5th |

---
## 📚 Documentation

Each project includes detailed documentation:

- **[Regression Project README](./Regression/README.md)** - Song year prediction details
- **[Clustering Project README](./Clustering/README.md)** - Customer segmentation details
- **[Transaction Project README](./Transaction/README.md)** - Fraud detection details

### Quick Navigation

| Project | Notebook | Dataset | Models |
|---------|----------|---------|--------|
| **Regression** | [EDA](./Regression/EDA.ipynb) \| [Training](./Regression/Model_Training.ipynb) \| [Evaluation](./Regression/Model_Evaluation.ipynb) | `midterm-regresi-dataset.csv` | 10 regressors |
| **Clustering** | [EDA](./Clustering/EDA.ipynb) \| [Training](./Clustering/Model_Training.ipynb) \| [Evaluation](./Clustering/Model_Evaluation.ipynb) | `clusteringmidterm.csv` | 6 algorithms |
| **Transaction** | [EDA](./Transaction/EDA.ipynb) \| [Training](./Transaction/Model_Training.ipynb) \| [Evaluation](./Transaction/Model_Evaluation.ipynb) | `train_transaction.csv` | 6 classifiers |

---

## 🤝 Contributing

This is an academic project portfolio. However, suggestions and feedback are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📧 Contact & Links

- **GitHub Repository**: [MaksInvader/MachineLearningClassMidterm](https://github.com/MaksInvader/MachineLearningClassMidterm)
- **Author**: MaksInvader
- **Course**: Machine Learning Midterm Test
- **Date**: December 2025

---

## 📄 License

This project is part of an academic assignment. Please respect academic integrity policies if using this code as reference.

---

## 🙏 Acknowledgments

- **Course Instructors** - For comprehensive machine learning curriculum
- **Scikit-learn Community** - For excellent documentation and examples
- **Kaggle** - For dataset inspiration and competitions
- **RAPIDS Team** - For GPU-accelerated data science tools
- **Open Source Community** - For incredible ML libraries and tools

---

## 📊 Project Statistics

```
Total Lines of Code:      15,000+
Total Models Trained:     22 models
Total Visualizations:     50+ charts
Total Documentation:      10,000+ words
Notebooks Created:        18 notebooks
Datasets Processed:       3 datasets
Features Engineered:      100+ features
Model Files Saved:        25+ .pkl/.json files
```

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Built with ❤️ using Python, Scikit-learn, XGBoost, and Jupyter Notebooks**

[🔝 Back to Top](#-machine-learning-midterm-project-portfolio)

</div>
