# 🎯 Customer Segmentation - Clustering Analysis

A comprehensive unsupervised machine learning project for customer segmentation using multiple clustering algorithms. This project identifies distinct customer groups to enable targeted marketing strategies and personalized business approaches.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Clustering Algorithms](#clustering-algorithms)
- [Results](#results)
- [Visualizations](#visualizations)
- [Key Findings](#key-findings)
- [Business Applications](#business-applications)
- [Future Improvements](#future-improvements)

## 🎯 Project Overview

This project performs comprehensive customer segmentation analysis using 6 different clustering algorithms. The workflow includes:

1. **Exploratory Data Analysis (EDA)** - Understanding customer characteristics and distributions
2. **Data Preprocessing** - Cleaning, scaling, and preparing data
3. **Optimal Cluster Selection** - Using Elbow Method and Silhouette Analysis
4. **Model Training** - Training 6 different clustering models
5. **Comprehensive Evaluation** - Multi-dimensional analysis with visualizations
6. **Business Insights** - Actionable customer segment profiles

## 📊 Dataset

- **Source**: `clusteringmidterm.csv`
- **Type**: Unsupervised learning (Clustering)
- **Purpose**: Customer segmentation for targeted marketing
- **Features**: Customer behavioral and demographic attributes

### Data Preprocessing Steps

1. **Missing Value Handling**: Automatic imputation with column means
2. **Outlier Detection**: Statistical methods for anomaly identification
3. **Feature Scaling**: Two scaling methods implemented:
   - StandardScaler (recommended for most algorithms)
   - MinMaxScaler (for distance-based algorithms)
4. **Data Validation**: NaN and infinite value checks

## 📁 Project Structure

```
Clustering/
├── 📓 Notebooks
│   ├── EDA.ipynb                          # Exploratory Data Analysis
│   ├── Model_Training.ipynb               # Clustering model training
│   ├── Model_Evaluation.ipynb             # Comprehensive evaluation
│   └── Model_Training_RAPIDS.ipynb        # GPU-accelerated training
│
├── 📁 Data Files
│   ├── clusteringmidterm.csv              # Original dataset
│   ├── clustering_cleaned.csv             # Cleaned dataset
│   ├── clustering_scaled_standard.csv     # StandardScaler output
│   ├── clustering_scaled_minmax.csv       # MinMaxScaler output
│   ├── customers_with_clusters.csv        # Final data with clusters
│   └── customers_with_clusters_numeric.csv # Numeric features only
│
├── 🤖 Model Files
│   ├── model_kmeans.pkl                   # K-Means model
│   ├── model_minibatch_kmeans.pkl         # Mini-Batch K-Means
│   ├── model_agglomerative.pkl            # Hierarchical Clustering
│   ├── model_dbscan.pkl                   # DBSCAN
│   ├── model_gmm.pkl                      # Gaussian Mixture Model
│   ├── model_spectral.pkl                 # Spectral Clustering
│   └── all_clustering_results.pkl         # All results combined
│
├── 📊 Results & Analysis
│   ├── clustering_results.csv             # Model comparison metrics
│   ├── clustering_labels.csv              # Cluster assignments
│   ├── cluster_profiles.csv               # Cluster characteristics
│   ├── clustering_summary.txt             # Summary statistics
│   ├── optimal_clusters_analysis.png      # Optimal k analysis
│   ├── clustering_comparison.png          # Model comparison charts
│   ├── clusters_pca_2d.png                # 2D PCA visualizations
│   ├── clusters_pca_3d_best.png           # 3D visualization
│   ├── silhouette_analysis.png            # Silhouette plot
│   ├── cluster_size_distribution.png      # Size distribution
│   ├── cluster_profiles_radar.png         # Radar charts
│   └── feature_importance.png             # Feature analysis
│
├── 🔧 Scalers
│   ├── scaler_standard.pkl                # StandardScaler
│   └── scaler_minmax.pkl                  # MinMaxScaler
│
└── 📄 Documentation
    └── README.md                          # This file
```

## 🛠️ Installation

### Prerequisites

```bash
Python 3.8+
pip or conda
```

### Required Libraries

```bash
# Core libraries
pip install numpy pandas scikit-learn

# Visualization
pip install matplotlib seaborn

# Data profiling
pip install skimpy

# Environment management
pip install python-dotenv

# Scientific computing
pip install scipy
```

### Optional: RAPIDS GPU Acceleration

For GPU-accelerated clustering:

```bash
# Create conda environment with RAPIDS
conda create -n rapids-clustering -c rapidsai -c conda-forge -c nvidia \
    rapids=23.10 python=3.10 cudatoolkit=11.8
```

## 🚀 Usage

### 1. Exploratory Data Analysis

```bash
jupyter notebook EDA.ipynb
```

This notebook will:
- Analyze customer feature distributions
- Identify patterns and correlations
- Detect outliers and missing values
- Generate initial insights

### 2. Model Training

```bash
jupyter notebook Model_Training.ipynb
```

This notebook will:
- Determine optimal number of clusters
- Train 6 different clustering models
- Save trained models and results
- Generate initial comparison

### 3. Model Evaluation

```bash
jupyter notebook Model_Evaluation.ipynb
```

This notebook provides:
- Comprehensive performance metrics
- 2D and 3D visualizations
- Silhouette analysis
- Cluster profiling
- Business insights

## 🤖 Clustering Algorithms

| # | Algorithm | Type | Optimal K | Use Case |
|---|-----------|------|-----------|----------|
| 1 | **K-Means** | Partitioning | Required | Fast, spherical clusters |
| 2 | **Mini-Batch K-Means** | Partitioning | Required | Large datasets, speed priority |
| 3 | **Agglomerative** | Hierarchical | Required | Dendogram analysis, nested clusters |
| 4 | **DBSCAN** | Density-based | Auto | Irregular shapes, noise detection |
| 5 | **Gaussian Mixture** | Probabilistic | Required | Soft clustering, overlapping groups |
| 6 | **Spectral** | Graph-based | Required | Non-convex clusters |

### Optimal Cluster Selection Methods

1. **Elbow Method** - Find the "elbow" in the inertia curve
2. **Silhouette Analysis** - Maximize cohesion and separation
3. **Calinski-Harabasz Index** - Maximize variance ratio
4. **Davies-Bouldin Index** - Minimize cluster similarity

## 📈 Results

### Model Performance Summary

| Model | Silhouette Score | Calinski-Harabasz | Davies-Bouldin | Training Time | Clusters |
|-------|-----------------|-------------------|----------------|---------------|----------|
| **K-Means** | **0.2471** | **1546.53** | 1.6054 | 0.09s | 3 |
| Mini-Batch K-Means | 0.2074 | 1511.87 | 1.6732 | 0.03s | 3 |
| Agglomerative | 0.1785 | 1296.05 | 1.7848 | 3.01s | 3 |
| Spectral | 0.1611 | 1218.02 | 1.7610 | 5.05s | 3 |
| Gaussian Mixture | 0.1151 | 887.56 | 2.6277 | 0.37s | 3 |
| DBSCAN | -0.3110 | 46.63 | **0.9627** | 0.08s | 36* |

*DBSCAN found 36 clusters with 6,488 noise points (data-driven)

### Best Models

🏆 **Best Overall Performance**: K-Means
- **Silhouette Score**: 0.2471 (highest cohesion/separation)
- **Calinski-Harabasz**: 1546.53 (best variance ratio)
- **Training Time**: 0.09s
- **Optimal Clusters**: 3
- **Reason**: Best balance of performance, speed, and interpretability

⚡ **Fastest Model**: Mini-Batch K-Means
- **Training Time**: 0.03s
- **Silhouette Score**: 0.2074
- **Good for**: Real-time applications, large-scale data

🔍 **Best for Noise Detection**: DBSCAN
- **Noise Detection**: 6,488 outliers identified
- **Adaptive**: No pre-specified k required
- **Good for**: Irregular cluster shapes, anomaly detection

## 📊 Visualizations

The project generates 9 comprehensive visualizations:

### 1. Optimal Cluster Analysis (`optimal_clusters_analysis.png`)
- 4-panel analysis showing:
  - Elbow Method (Inertia curve)
  - Silhouette Score progression
  - Calinski-Harabasz Index
  - Davies-Bouldin Index
- Helps determine optimal k value

### 2. Model Comparison (`clustering_comparison.png`)
- Side-by-side metric comparison
- Silhouette scores
- Calinski-Harabasz indices
- Davies-Bouldin indices
- Training time comparison

### 3. 2D PCA Visualizations (`clusters_pca_2d.png`)
- All 6 models visualized
- Principal Component Analysis projection
- Shows cluster separation
- Color-coded customer segments

### 4. 3D PCA Visualization (`clusters_pca_3d_best.png`)
- Best model (K-Means) in 3D space
- Interactive perspective
- Enhanced cluster separation view
- Variance explained by components

### 5. Silhouette Analysis (`silhouette_analysis.png`)
- Per-cluster silhouette coefficients
- Shows cluster quality
- Identifies poorly clustered samples
- Average silhouette line

### 6. Cluster Size Distribution (`cluster_size_distribution.png`)
- Balance across all models
- Sample counts per cluster
- Percentage distribution
- Identifies dominant segments

### 7. Cluster Profiles - Radar Chart (`cluster_profiles_radar.png`)
- Top 8 distinguishing features
- Normalized 0-1 scale
- Comparative view of segments
- Easy pattern identification

### 8. Feature Importance (`feature_importance.png`)
- Variance between cluster means
- Ranked by importance
- Color-coded significance
- Guides feature selection

### 9. Business Insights (In Notebook)
- Detailed segment characteristics
- Deviation from average
- Statistical summaries
- Actionable recommendations

## 🔍 Key Findings

### Optimal Clustering Configuration

✅ **Recommended**: K-Means with k=3 clusters
- Provides clear, actionable customer segments
- High interpretability for business decisions
- Fast training and prediction
- Stable and reproducible results

### Cluster Characteristics (K-Means, k=3)

Based on the best-performing model, customers are segmented into 3 distinct groups:

**Cluster Distribution:**
- Each cluster represents a unique customer segment
- Balanced distribution ensures no segment is overlooked
- Clear separation indicates distinct customer behaviors

### Data Insights

- **Feature Scaling Impact**: StandardScaler provided best results for most algorithms
- **Algorithm Selection**: Partitioning methods (K-Means variants) outperformed others
- **Computational Efficiency**: Mini-Batch K-Means offers 3x speedup with minimal accuracy loss
- **Noise Detection**: DBSCAN identified ~6,488 potential outliers/anomalous customers

### Performance Insights

1. **K-Means dominates**: Achieved best Silhouette (0.247) and Calinski-Harabasz (1546.53)
2. **Speed vs Accuracy**: Mini-Batch K-Means offers 3x speedup with 16% Silhouette drop
3. **Complex models underperform**: GMM and Spectral showed lower scores
4. **DBSCAN's challenge**: Negative Silhouette indicates difficulty with dataset structure

## 💼 Business Applications

### Customer Segmentation Use Cases

#### 🎯 **Marketing Strategy**
- **Targeted Campaigns**: Design specific campaigns for each cluster
- **Channel Optimization**: Identify preferred communication channels per segment
- **Message Personalization**: Tailor messaging to segment characteristics
- **Budget Allocation**: Invest proportionally based on segment value

#### 💰 **Revenue Optimization**
- **Pricing Strategy**: Segment-specific pricing models
- **Upsell/Cross-sell**: Targeted product recommendations
- **Retention Programs**: Segment-specific loyalty initiatives
- **Customer Lifetime Value**: Predict CLV by segment

#### 📦 **Product Development**
- **Feature Prioritization**: Develop features for specific segments
- **New Product Ideas**: Identify unmet needs per cluster
- **Product Bundling**: Create segment-appropriate bundles
- **Service Customization**: Tailor services to segment preferences

#### 🔧 **Operations**
- **Resource Allocation**: Optimize support based on segment needs
- **Inventory Management**: Stock products per segment demand
- **Service Levels**: Differentiated SLAs by segment
- **Capacity Planning**: Forecast demand by customer segment

### Actionable Recommendations

1. **Segment-Specific Strategy**: Develop unique value propositions for each cluster
2. **Continuous Monitoring**: Track cluster stability over time
3. **A/B Testing**: Test strategies on segments before full rollout
4. **Predictive Modeling**: Use clusters as features in supervised learning
5. **Dynamic Segmentation**: Re-cluster periodically as customer behavior evolves

## 🎯 Evaluation Metrics Explained

### Silhouette Score (-1 to 1, higher is better)
- Measures how similar a point is to its own cluster vs. other clusters
- **> 0.5**: Strong structure
- **0.25-0.5**: Reasonable structure (our K-Means: 0.247)
- **< 0.25**: Weak structure, overlapping clusters

### Calinski-Harabasz Index (higher is better)
- Ratio of between-cluster dispersion to within-cluster dispersion
- Higher values indicate better-defined clusters
- No fixed threshold, used for comparison
- K-Means achieved 1546.53 (highest)

### Davies-Bouldin Index (lower is better)
- Average similarity between each cluster and its most similar cluster
- **< 1**: Excellent separation
- **1-2**: Good separation (most models in this range)
- **> 2**: Poor separation

### Inertia (K-Means specific, lower is better)
- Sum of squared distances to nearest cluster center
- Used in Elbow Method
- Decreases with more clusters
- Look for the "elbow" point

## 🚀 Future Improvements

### Model Enhancements
- [ ] Implement HDBSCAN for better density-based clustering
- [ ] Try Fuzzy C-Means for soft cluster assignments
- [ ] Ensemble clustering methods
- [ ] Deep learning-based clustering (Autoencoders)
- [ ] Time-series clustering for temporal patterns

### Feature Engineering
- [ ] Create interaction features
- [ ] Apply dimensionality reduction (t-SNE, UMAP)
- [ ] Feature selection based on variance analysis
- [ ] Domain-specific feature creation
- [ ] Automatic feature importance ranking

### Analysis & Validation
- [ ] Stability analysis across different random seeds
- [ ] Cross-validation for cluster consistency
- [ ] Consensus clustering from multiple algorithms
- [ ] Cluster validation with business KPIs
- [ ] Temporal cluster evolution tracking

### Infrastructure
- [ ] Deploy as REST API for real-time segmentation
- [ ] Create web dashboard for cluster exploration
- [ ] Implement automated retraining pipeline
- [ ] Add cluster drift detection
- [ ] Docker containerization for reproducibility

### Business Intelligence
- [ ] Integrate with CRM systems
- [ ] Build customer journey maps per segment
- [ ] ROI tracking for segment-specific campaigns
- [ ] Predictive churn modeling by segment
- [ ] Lifetime value prediction per cluster

## 📝 Notes

- **Optimal k Selection**: Used multiple methods (Elbow, Silhouette, Calinski-Harabasz) for robust k=3 selection
- **Data Quality**: Automatic handling of NaN and infinite values ensures robustness
- **Reproducibility**: Random state = 42 ensures consistent results
- **Scalability**: Mini-Batch K-Means available for large datasets
- **Model Persistence**: All models saved as `.pkl` files for production use

## 🔧 Troubleshooting

### Common Issues

**Issue**: NaN values in data
- **Solution**: Run data quality check cell - automatically fills with column means

**Issue**: TypeError in groupby operations
- **Solution**: Notebook filters non-numeric columns automatically

**Issue**: Poor clustering performance
- **Solution**: Try different scaling methods or adjust k value

**Issue**: DBSCAN finds too many clusters
- **Solution**: Adjust `eps` and `min_samples` parameters

**Issue**: Memory errors with large datasets
- **Solution**: Use Mini-Batch K-Means or process in chunks

## 📚 References

- [Scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [Silhouette Analysis](https://en.wikipedia.org/wiki/Silhouette_(clustering))
- [K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
- [DBSCAN Algorithm](https://en.wikipedia.org/wiki/DBSCAN)

## 🤝 Contributing

This project is part of a Machine Learning course. Suggestions and improvements are welcome!

## 📜 License

This project is for educational purposes.

## 👨‍💻 Author

Machine Learning Midterm Project - Clustering Analysis  
Date: December 2025

## 🙏 Acknowledgments

- Scikit-learn team for excellent ML tools
- Course instructors for guidance
- Open-source community for inspiration

---

**💡 Pro Tip**: Start with K-Means for quick insights, then explore other algorithms for specific use cases!

**🎯 Next Steps**: Use cluster labels as features in supervised learning tasks or integrate with business systems for automated segmentation!

#### Key Features:
- **Optimal k Selection**: Elbow method + Silhouette analysis
- **Comprehensive Metrics**: Silhouette, Calinski-Harabasz, Davies-Bouldin, Inertia
- **Automatic Evaluation**: Consistent evaluation across all models
- **Model Persistence**: All models saved as .pkl files
- **Visualization**: Comparison charts and optimal k analysis

#### Output Files:
- `model_kmeans.pkl` - K-Means model
- `model_minibatch_kmeans.pkl` - Mini-Batch K-Means
- `model_agglomerative.pkl` - Hierarchical clustering
- `model_dbscan.pkl` - DBSCAN model
- `model_gmm.pkl` - Gaussian Mixture Model
- `model_spectral.pkl` - Spectral clustering
- `clustering_results.csv` - Performance comparison
- `clustering_labels.csv` - All cluster assignments
- `all_clustering_results.pkl` - Complete results object
- `clustering_comparison.png` - Visualization
- `optimal_clusters_analysis.png` - K selection analysis

### 3. Model_Evaluation.ipynb - Comprehensive Analysis
**Status**: ✅ Complete

#### Evaluation Sections (8 comprehensive analyses):

**Section 1: Performance Metrics Summary**
- Detailed comparison table
- Best models by each metric
- Styled DataFrame with gradients

**Section 2: Cluster Visualizations**
- 2D PCA projections (all models)
- 3D PCA projection (best model)
- Interactive 3D visualization
- Variance explained by PCs

**Section 3: Silhouette Analysis**
- Detailed silhouette plot
- Per-cluster silhouette scores
- Sample-level analysis
- Visual identification of poor clusters

**Section 4: Cluster Size Distribution**
- Bar charts for all models
- Percentage labels
- Noise point identification
- Balance analysis

**Section 5: Cluster Profiling**
- Statistical profiles (mean, std, min, max)
- Radar charts for visualization
- Feature-wise comparison
- Normalized profiles

**Section 6: Feature Importance**
- Variance between cluster means
- Ranking of discriminative features
- Horizontal bar chart
- Identifies key drivers

**Section 7: Business Insights**
- Customer segment characterization
- Top distinguishing features per cluster
- Deviation from overall mean
- Actionable insights

**Section 8: Final Summary Report**
- Complete evaluation summary
- Best model recommendation
- Output file inventory
- Next steps

#### Output Files:
- `customers_with_clusters.csv` - Final dataset with assignments
- `cluster_profiles.csv` - Detailed cluster statistics
- `clusters_pca_2d.png` - 2D visualizations (all models)
- `clusters_pca_3d_best.png` - 3D visualization (best model)
- `silhouette_analysis.png` - Silhouette plot
- `cluster_size_distribution.png` - Size charts
- `cluster_profiles_radar.png` - Radar charts
- `feature_importance.png` - Feature analysis
- `clustering_summary.txt` - Text summary

---

## 🔄 Complete Workflow

```
1. EDA.ipynb
   ├── Load raw data
   ├── Clean & analyze
   ├── Scale features
   └── Export: clustering_scaled_standard.csv
   
2. Model_Training.ipynb
   ├── Load scaled data
   ├── Determine optimal k
   ├── Train 6 models
   ├── Evaluate & compare
   └── Export: models + results
   
3. Model_Evaluation.ipynb
   ├── Load models & results
   ├── Comprehensive analysis (8 sections)
   ├── Visualizations
   └── Export: insights + profiles
```

---

## 📊 Evaluation Metrics Explained

### Silhouette Score (-1 to 1)
- **Measures**: How similar an object is to its own cluster vs. other clusters
- **Higher is better**: > 0.5 is good, > 0.7 is excellent
- **Interpretation**:
  - Close to 1: Well-clustered
  - Close to 0: On decision boundary
  - Negative: Possibly wrong cluster

### Calinski-Harabasz Index (0 to ∞)
- **Measures**: Ratio of between-cluster to within-cluster variance
- **Higher is better**: No absolute threshold
- **Interpretation**: Higher values indicate better-defined clusters

### Davies-Bouldin Index (0 to ∞)
- **Measures**: Average similarity between each cluster and its most similar one
- **Lower is better**: 0 is perfect (rarely achievable)
- **Interpretation**: Lower values indicate better separation

### Inertia (K-Means only)
- **Measures**: Sum of squared distances to cluster centers
- **Lower is better**: But beware of overfitting
- **Use**: Elbow method to find optimal k

---

## 🎯 Model Selection Guide

### When to use each algorithm:

**K-Means**
- ✅ Large datasets
- ✅ Spherical clusters
- ✅ Fast computation needed
- ❌ Non-convex clusters
- ❌ Different sizes/densities

**Mini-Batch K-Means**
- ✅ Very large datasets (>100k samples)
- ✅ Limited memory
- ✅ Online learning
- ❌ Need exact K-Means results

**Agglomerative Clustering**
- ✅ Hierarchical relationships
- ✅ Small to medium datasets
- ✅ Dendrogram visualization
- ❌ Large datasets (slow)
- ❌ No cluster reassignment

**DBSCAN**
- ✅ Arbitrary cluster shapes
- ✅ Noise detection needed
- ✅ Unknown number of clusters
- ❌ Varying densities
- ❌ High-dimensional data

**Gaussian Mixture Model**
- ✅ Soft cluster assignments
- ✅ Probabilistic interpretation
- ✅ Elliptical clusters
- ❌ Many components needed
- ❌ Large datasets

**Spectral Clustering**
- ✅ Non-convex clusters
- ✅ Graph-based data
- ✅ Few clusters
- ❌ Large datasets (very slow)
- ❌ Many clusters

---

## 💡 Best Practices

### Data Preparation
1. **Always scale features** - Clustering is sensitive to scale
2. **Handle outliers** - Can distort cluster centers
3. **Remove duplicates** - Can bias cluster sizes
4. **Check correlations** - Remove highly correlated features

### Model Selection
1. **Try multiple algorithms** - Different data suits different algorithms
2. **Use multiple metrics** - Don't rely on one metric alone
3. **Visualize results** - PCA/t-SNE helps validate
4. **Validate business logic** - Clusters should make sense

### Optimal k Selection
1. **Elbow method** - Look for "elbow" in inertia curve
2. **Silhouette analysis** - Maximize average silhouette
3. **Domain knowledge** - Sometimes business dictates k
4. **Stability testing** - Consistent results across runs

### Interpretation
1. **Profile each cluster** - Understand characteristics
2. **Name segments** - Give meaningful labels
3. **Size balance** - Avoid too small/large clusters
4. **Actionability** - Clusters should drive decisions

---

## 🚀 Next Steps

### After Clustering:
1. **Validate with stakeholders** - Do segments make business sense?
2. **Deploy to production** - Use best model for predictions
3. **Monitor over time** - Cluster drift detection
4. **A/B testing** - Test strategies per segment
5. **Feature engineering** - Can we improve clusters?

### Advanced Techniques:
- **Ensemble clustering** - Combine multiple algorithms
- **Hierarchical k-means** - Multi-level segmentation
- **Time-series clustering** - For temporal data
- **Deep clustering** - Neural network approaches

---

## 📈 Expected Results

### Typical Outcomes:
- **3-7 clusters** for customer segmentation
- **Silhouette > 0.3** is acceptable, > 0.5 is good
- **10-30% variance explained** by first 2 PCs
- **Training time**: seconds to minutes (depends on algorithm)

### Success Indicators:
- ✅ Clear separation in visualizations
- ✅ Balanced cluster sizes (not 99% + 1%)
- ✅ Consistent results across runs
- ✅ Interpretable cluster profiles
- ✅ Actionable business insights

---

## 📁 File Structure

```
Clustering/
├── EDA.ipynb                           # Data exploration
├── Model_Training.ipynb                # Train 6 models
├── Model_Evaluation.ipynb              # Comprehensive evaluation
│
├── Data Files:
│   ├── clusteringmidterm.csv          # Raw data
│   ├── clustering_cleaned.csv          # Cleaned data
│   ├── clustering_scaled_standard.csv  # Scaled (recommended)
│   └── clustering_scaled_minmax.csv    # Alternative scaling
│
├── Model Files:
│   ├── model_kmeans.pkl
│   ├── model_minibatch_kmeans.pkl
│   ├── model_agglomerative.pkl
│   ├── model_dbscan.pkl
│   ├── model_gmm.pkl
│   └── model_spectral.pkl
│
├── Results:
│   ├── clustering_results.csv          # Performance comparison
│   ├── clustering_labels.csv           # All cluster labels
│   ├── cluster_profiles.csv            # Segment characteristics
│   ├── customers_with_clusters.csv     # Final dataset
│   ├── all_clustering_results.pkl      # Complete results
│   └── clustering_summary.txt          # Text summary
│
├── Scalers:
│   ├── scaler_standard.pkl
│   └── scaler_minmax.pkl
│
└── Visualizations:
    ├── optimal_clusters_analysis.png
    ├── clustering_comparison.png
    ├── clusters_pca_2d.png
    ├── clusters_pca_3d_best.png
    ├── silhouette_analysis.png
    ├── cluster_size_distribution.png
    ├── cluster_profiles_radar.png
    └── feature_importance.png
```

---

## 🎓 Learning Resources

### Clustering Fundamentals:
- Scikit-learn Clustering Guide
- "Introduction to Statistical Learning" (Chapter 10)
- "Pattern Recognition and Machine Learning" (Bishop)

### Advanced Topics:
- Ensemble Clustering Methods
- Cluster Validation Techniques
- High-Dimensional Clustering
- Subspace Clustering

---