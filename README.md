# Hierarchical and DBSCAN Clustering Lab - README

## Overview
This lab explores two powerful unsupervised learning techniques—Hierarchical Clustering and DBSCAN—applied to the Wine dataset from scikit-learn. The lab demonstrates how different clustering algorithms perform on the same dataset and provides insights into their strengths, weaknesses, and optimal use cases.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Dataset Information](#dataset-information)
4. [Lab Structure](#lab-structure)
5. [Running the Lab](#running-the-lab)
6. [Expected Outputs](#expected-outputs)
7. [Key Findings](#key-findings)
8. [Troubleshooting](#troubleshooting)
9. [Additional Resources](#additional-resources)

---

## Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of NumPy and Pandas
- Familiarity with machine learning concepts
- Basic knowledge of clustering algorithms

### Software Requirements
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Required Python libraries (see Installation section)

---

## Installation

### Step 1: Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

Or if you're using conda:

```bash
conda install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Step 2: Verify Installation

Open a Python terminal and run:

```python
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
print("All libraries installed successfully!")
```

---

## Dataset Information

### Wine Dataset
- **Source**: UCI Machine Learning Repository (available in sklearn.datasets)
- **Samples**: 178 wine samples
- **Features**: 13 chemical properties
- **Classes**: 3 wine cultivars (varieties)

### Features Include:
1. Alcohol
2. Malic acid
3. Ash
4. Alkalinity of ash
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline

### Why This Dataset?
The Wine dataset is ideal for clustering because:
- Well-separated natural groups (3 wine types)
- Clean data with no missing values
- Sufficient samples for meaningful analysis
- Multiple features for pattern recognition

---

## Lab Structure

### Step 1: Data Preparation and Exploration (15-20 minutes)
**Objective**: Understand the dataset and prepare it for clustering

**Activities**:
- Load Wine dataset
- Explore data structure and statistics
- Standardize features using StandardScaler
- Visualize feature distributions

**Key Outputs**:
- Dataset summary statistics
- Box plots showing feature distributions before/after scaling

---

### Step 2: Hierarchical Clustering (20-25 minutes)
**Objective**: Apply and evaluate Agglomerative Hierarchical Clustering

**Activities**:
- Test multiple n_clusters values (2, 3, 4, 5)
- Calculate evaluation metrics
- Visualize clusters in 2D space
- Generate and interpret dendrogram

**Key Outputs**:
- Silhouette, Homogeneity, and Completeness scores
- 4 scatter plots showing different cluster configurations
- Dendrogram with suggested cut points

**Parameters Tested**:
| n_clusters | Purpose |
|------------|---------|
| 2 | Minimal clustering |
| 3 | Matches true number of classes |
| 4 | Over-clustering test |
| 5 | Maximum over-clustering test |

---

### Step 3: DBSCAN Clustering (25-30 minutes)
**Objective**: Apply DBSCAN and explore parameter sensitivity

**Activities**:
- Test 12 parameter combinations (4 eps × 3 min_samples)
- Identify noise points
- Calculate evaluation metrics
- Compare different configurations

**Key Outputs**:
- Performance metrics for all 12 configurations
- 4 visualizations showing different parameter effects
- Best configuration identification

**Parameters Tested**:
| eps | min_samples | Expected Behavior |
|-----|-------------|-------------------|
| 0.5 | 3, 5, 10 | Many small clusters, more noise |
| 1.0 | 3, 5, 10 | Moderate clustering |
| 1.5 | 3, 5, 10 | Fewer, larger clusters |
| 2.0 | 3, 5, 10 | Very few clusters, minimal noise |

---

### Step 4: Analysis and Insights (15-20 minutes)
**Objective**: Compare methods and draw conclusions

**Activities**:
- Compare best configurations from both methods
- Analyze parameter influence
- Discuss strengths and weaknesses
- Provide practical recommendations

**Key Outputs**:
- Side-by-side metric comparison
- Bar chart visualization
- Comprehensive analysis report

---

## Running the Lab

### Option 1: Jupyter Notebook

1. **Download the notebook**:
   - Save the provided Python code as `clustering_lab.ipynb`

2. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

3. **Open and run**:
   - Navigate to `clustering_lab.ipynb`
   - Update your name and course information in the first cell
   - Run all cells: `Cell → Run All`

### Option 2: JupyterLab

1. **Start JupyterLab**:
   ```bash
   jupyter lab
   ```

2. **Create new notebook**:
   - File → New → Notebook
   - Copy and paste the code
   - Run all cells

### Option 3: Google Colab

1. **Upload to Google Colab**:
   - Go to https://colab.research.google.com/
   - File → Upload notebook
   - Upload your `.ipynb` file

2. **Run**:
   - All required libraries are pre-installed in Colab
   - Run all cells

---

## Expected Outputs

### Visualizations (7 total)
1. **Feature Distribution Comparison**: Box plots before/after standardization
2. **Hierarchical Clusters (4 plots)**: Different n_clusters configurations
3. **Dendrogram**: Hierarchical structure with suggested cuts
4. **DBSCAN Results (4 plots)**: Different parameter combinations
5. **Performance Comparison Bar Chart**: Metric comparison

### Metrics Summary

#### Typical Hierarchical Clustering Results:
- **Best n_clusters**: 3
- **Silhouette Score**: ~0.55-0.60
- **Homogeneity Score**: ~0.40-0.45
- **Completeness Score**: ~0.42-0.47

#### Typical DBSCAN Results:
- **Best eps**: 1.0-1.5
- **Best min_samples**: 3-5
- **Silhouette Score**: ~0.35-0.50
- **Noise points**: 5-20 samples
- **Number of clusters**: 2-4

---

## Key Findings

### When Hierarchical Clustering Excels:
✓ Dataset has clear hierarchical structure  
✓ Need to explore multiple granularities  
✓ Data forms spherical/convex clusters  
✓ Dataset size is manageable (<10,000 samples)  
✓ Want deterministic, reproducible results

### When DBSCAN Excels:
✓ Clusters have irregular, non-convex shapes  
✓ Dataset contains significant outliers/noise  
✓ Don't know number of clusters in advance  
✓ Dealing with spatial or geographic data  
✓ Cluster densities vary significantly

### For the Wine Dataset:
- **Winner**: Hierarchical Clustering
- **Reason**: Well-separated, spherical clusters match hierarchical assumptions
- **DBSCAN limitation**: Dataset is too clean; noise detection not needed

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Import Errors
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: 
```bash
pip install scikit-learn
```

#### Issue 2: Plotting Issues
```
UserWarning: Matplotlib is currently using agg
```
**Solution**: Add to the top of your notebook:
```python
%matplotlib inline
```

#### Issue 3: Seaborn Style Warning
```
FutureWarning: seaborn style
```
**Solution**: Already handled in the code with:
```python
plt.style.use('seaborn-v0_8-darkgrid')
```

#### Issue 4: Memory Issues
If running on limited memory systems:
- Reduce the number of parameter combinations tested
- Comment out some visualization cells
- Use smaller portions of the dataset

#### Issue 5: Dendrogram Too Crowded
The dendrogram uses truncation:
```python
dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
```
Adjust `p` parameter to show more/fewer nodes.

---

## Evaluation Metrics Explained

### Silhouette Score (Range: -1 to 1)
- Measures how similar an object is to its own cluster vs. other clusters
- **Higher is better**
- > 0.7: Strong structure
- 0.5-0.7: Reasonable structure
- 0.25-0.5: Weak structure
- < 0.25: No substantial structure

### Homogeneity Score (Range: 0 to 1)
- Measures if clusters contain only members of a single class
- **Higher is better**
- 1.0: Perfect homogeneity
- Useful when true labels are known

### Completeness Score (Range: 0 to 1)
- Measures if all members of a class are in the same cluster
- **Higher is better**
- 1.0: Perfect completeness
- Complements homogeneity

---

## Additional Resources

### Documentation
- **Scikit-learn Clustering**: https://scikit-learn.org/stable/modules/clustering.html
- **Hierarchical Clustering**: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
- **DBSCAN**: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
- **Wine Dataset**: https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset

### Academic Papers
- **DBSCAN**: Ester, M., et al. (1996). "A density-based algorithm for discovering clusters"
- **Hierarchical Clustering**: Müllner, D. (2011). "Modern hierarchical, agglomerative clustering algorithms"

### Further Learning
- **Coursera**: Machine Learning by Andrew Ng
- **DataCamp**: Cluster Analysis in Python
- **YouTube**: StatQuest - Hierarchical Clustering and DBSCAN explained

### Related Topics
- K-Means Clustering
- Gaussian Mixture Models
- Spectral Clustering
- OPTICS (Ordering Points To Identify Clustering Structure)

---

## Extending the Lab

### Challenge Activities

1. **Try Other Linkage Methods**:
   ```python
   linkage_methods = ['ward', 'complete', 'average', 'single']
   # Compare results across different linkage methods
   ```

2. **Dimensionality Reduction First**:
   ```python
   from sklearn.decomposition import PCA
   pca = PCA(n_components=2)
   X_pca = pca.fit_transform(X_scaled)
   # Cluster on reduced dimensions
   ```

3. **Elbow Method for Optimal Clusters**:
   ```python
   # Calculate inertia for different k values
   # Plot elbow curve
   ```

4. **Apply to Other Datasets**:
   - Iris dataset
   - Breast Cancer dataset
   - Customer segmentation data

---

## Lab Submission Checklist

Before submitting your lab, ensure:

- [ ] Name, course title, and assignment title are in the first cell
- [ ] All cells run without errors
- [ ] All visualizations display correctly
- [ ] Metric tables show reasonable values
- [ ] Analysis section is complete with your insights
- [ ] Code includes comments explaining key steps
- [ ] Notebook is saved with outputs visible
- [ ] File is named appropriately (e.g., `LastName_FirstName_Clustering_Lab.ipynb`)

---

## Contact and Support

For questions about this lab:
- Check the troubleshooting section first
- Review scikit-learn documentation
- Ask your instructor or TA
- Post in the course discussion forum

---

## License

This lab is created for educational purposes. The Wine dataset is publicly available through the UCI Machine Learning Repository and scikit-learn.

---

## Version History

- **v1.0** (November 2025): Initial release
  - Complete implementation of Hierarchical and DBSCAN clustering
  - Comprehensive analysis and visualizations
  - Detailed documentation and troubleshooting guide

---

**Happy Clustering! 🎯📊**