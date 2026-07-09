**CO544: Machine Learning and Data Mining**   
**Lab 03: Decision Trees and k-Nearest Neighbors Classification**

E/21/245  
MADHUSHAN S.K.A.K.  
---

**Task 1:-  Build two decision tree classifiers with Gini index and entropy criteria for the given Wine.csv**

Duplicate and null checking :-

The dataset was initially verified for missing values and duplicates using Pandas functions  
The output confirmed that the dataset contains zero missing values and zero duplicate rows  
Therefore, no imputation or duplicate removal was required at this initial stage. The original   
dataset shape remains at 178 rows and 14 columns

Outlier Detection and Removal:-

To detect anomalies in the continuous features, boxplots were generated for all 13 chemical attributes. Based on visual observation of the boxplots, significant outliers (data points lying beyond the whiskers) were identified in several features, including `malic_acid`, `magnesium`, `proanth`, `color_intensity`, `hue`, `ash`, and `alcalinity`.

To handle these anomalies, the Interquartile Range (IQR) method with a standard 1.5 threshold was implemented. As a result, 17 outlier instances were systematically removed, reducing the total dataset size from 178 rows to 161 rows. This step ensures that the data is clean and prevents extreme values from negatively affecting distance-sensitive classifiers later in the practical.

Target Class Distribution:-

The target variable `wine_class` was analyzed to assess class distribution. The dataset contains three unique cultivator categories: Class 1, Class 2, and Class 3\. As observed from the distribution plot, the instances are relatively well-balanced across the categories—with Class 2 having the highest frequency followed by Class 1 and Class 3—ensuring that the classifiers are not systematically biased toward a single dominant majority class.

Baseline Decision Tree Classifiers (Gini vs Entropy):-

After cleaning, the dataset (161 instances) was split into a 75% training set and a 25% testing set using a fixed random state of 42 for reproducibility. Two baseline Decision Tree models were trained using different splitting criteria: Gini Impurity and Information Gain (Entropy).

The **Gini-based model** achieved a testing accuracy of **87.80%,** whereas the **Entropy-based model** achieved a higher accuracy of **92.68%**. This initial observation suggests that for the specific distribution of continuous chemical features in the Wine dataset, measuring the logarithmic reduction of uncertainty (Entropy) leads to more informative feature splits at the tree nodes compared to Gini impurity.

Model Evaluation :- 

To comprehensively evaluate the performance of both baseline classifiers beyond simple overall accuracy, a Confusion Matrix and a detailed Classification Report (comprising Precision, Recall, and F1-score) were generated for both the Gini Impurity and Information Gain (Entropy) models

A classification performance analysis was conducted on the test split using Confusion Matrices and Classification Reports to compare the Gini Impurity and Information Gain (Entropy) criteria.

* **Gini Criterion Model:** Correctly predicted 36 out of 41 test instances (5 misclassifications). It achieved a macro F1-score of 0.86. While it captured all Class 1 samples perfectly (Recall \= 1.00) and yielded no false positives for Class 2 (Precision \= 1.00), it suffered from minor class crossovers, reducing the classification accuracy of Class 3\.  
* **Information Gain (Entropy) Model:** Outperformed the Gini model by correctly classifying 38 out of 41 instances (only 3 misclassifications), achieving a higher macro F1-score of 0.93. This model minimized errors across all three categories, yielding a perfect Recall (1.00) for Class 1 and a perfect Precision (1.00) for Class 3\.

**Synthesis:** \> The evaluation confirms that the Entropy criterion is superior for this dataset. It produced higher, more balanced Precision and Recall metrics across all wine categories. This optimization is visually validated by the confusion matrix heatmaps, where the correct diagonal classification frequencies are higher and cleaner in the Entropy model compared to the Gini baseline.

Decision Tree Pruning and Visualization:-

To prevent the model from overfitting , the maximum depth of the tree was limited to 3 (`max_depth=3`). This constraint simplified the tree structure and improved its ability to work with new data, raising the test accuracy from 87.80% to 90.24%.

As shown in the tree diagram, `proline` is selected as the top feature (root node) to make the very first split. The tree then splits further down using clear boundaries based on other chemical features specifically `flavanoids`, `od280`, `color_intensity`, and `magnesium` to separate the wine classes accurately without becoming overly complicated

**Task 2:- Apply k-Nearest Neighbors to the same Wine.csv dataset**

(a) Feature Scaling Implementation

Since the kNN algorithm relies on computing the geometric distance (Euclidean distance) between data points, feature scaling is mandatory. Features with large magnitudes, such as `proline` (ranging up to 1600), would otherwise dominate features with smaller scales like `alcohol`. To ensure equal weight, `StandardScaler` was applied to normalize all 13 chemical features to a standard normal distribution.

##### b) Hyperparameter Tuning via Cross-Validation

To find the optimal parameters, a GridSearchCV was executed with a 5-fold cross-validation strategy. The search space evaluated $k$ values from 1 to 20 against Euclidean and Manhattan distance metrics.

* Optimal Configuration Found: k \= 17 using the Euclidean distance metric.

##### (c) Performance Evaluation & Algorithm Comparison

Using the optimal hyperparameters, the final kNN model was evaluated on the unseen testing dataset (41 samples). The model achieved a perfect testing accuracy of **100.00%**, with precision, recall, and F1-scores reaching a flawless **1.00** across all three wine classes. The model recorded an inference runtime of **0.037814 seconds**.

| Metric / Feature | Baseline Decision Tree (Gini) | Baseline Decision Tree (Entropy) | Pruned Decision Tree (max\_depth=3) | Optimized kNN Classifier (k=17) |
| :---- | :---- | :---- | :---- | :---- |
| **Testing Accuracy** | 87.80% | 92.68% | 90.24% | **100.00%** |
| **Macro F1-Score** | 0.86 | 0.93 | 0.89 | **1.00** |
| **Misclassifications** | 5 instances | 3 instances | 4 instances | **0 instances** |
| **Execution Speed** | Ultra-Fast (Direct logic) | Ultra-Fast (Direct logic) | Ultra-Fast (Direct logic) | Moderate (Needs to compute distances) |

##### 

##### **Comparison Synthesis:-**

The comparative analysis shows that the optimized kNN model significantly outperforms all configurations of the Decision Tree models on this specific dataset. The perfect accuracy is a result of effective data preprocessing (outlier removal and standard scaling) and hyperparameter tuning (k=17), which allowed kNN to form precise spatial boundaries around the distinct chemical clusters of the three wine types. While Decision Trees are inherently faster during inference and easier to interpret visually, kNN provides superior predictive performance for this continuous, non-linear chemical dataset.

											**E/21/245**

---

