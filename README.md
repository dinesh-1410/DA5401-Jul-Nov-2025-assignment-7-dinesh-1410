# DA5401 - Assignment 7: Multi-Class Model Selection using ROC and Precision-Recall Curves

## Student Information

- **Name:** Saggurthi Dinesh  
- **Roll Number:** BE21B032  
- **Email:** [be21b032@smail.iitm.ac.in](mailto:be21b032@smail.iitm.ac.in)  
- **Course:** DA5401 - Data Analytics Laboratory  
- **Assignment:** Assignment 6  

---

## Objective

The primary goal of this assignment is to **classify Landsat Satellite land-cover types** (6 classes) using various baseline machine learning models. The core task involves performing **model selection** by analyzing multi-class **Receiver Operating Characteristic (ROC)** and **Precision-Recall (PR)** curves, employing the **One-vs-Rest (OvR)** strategy with **macro-averaging**.

---

## Dataset

-   **Source:** Statlog (Landsat Satellite) Data Set from UCI Machine Learning Repository.
-   **Files:** `sat.trn` (training set), `sat.tst` (test set).
-   **Description:** Each data point consists of 36 numeric attributes representing spectral values, followed by a class label indicating the land-cover type (classes 1, 2, 3, 4, 5, 7).

---

## Methodology

The analysis follows these key steps:

1.  **Part A: Data Preparation & Baseline**
    * Load the training (`sat.trn`) and test (`sat.tst`) datasets.
    * Standardize features using `StandardScaler` (fit only on training data).
    * Train six baseline classification models:
        * K-Nearest Neighbors (KNN)
        * Decision Tree
        * Dummy Classifier (strategy='prior')
        * Logistic Regression (multinomial)
        * Gaussian Naive Bayes (GaussianNB)
        * Support Vector Classifier (SVC with `probability=True`)
    * Evaluate baseline models using **Overall Accuracy** and **Weighted F1-score** on the test set.

2.  **Part B: ROC Analysis (Multi-class, OvR, Macro-average)**
    * Explain the OvR approach for multi-class ROC and AUC calculation.
    * Compute and plot the **macro-averaged OvR ROC curves** for all models on a single plot.
    * Identify the model with the highest macro-averaged AUC and check for any models performing worse than random (AUC < 0.5).

3.  **Part C: Precision-Recall Analysis (Multi-class, OvR, Macro-average)**
    * Discuss the importance of PRC for potentially imbalanced datasets.
    * Compute and plot the **macro-averaged OvR Precision-Recall curves** for all models.
    * Report the **macro-averaged Average Precision (AP)** for each model.
    * Identify the best and worst models based on AP.

4.  **Part D: Synthesis & Final Recommendation**
    * Compare model rankings based on Weighted F1, ROC-AUC, and PRC-AP.
    * Discuss trade-offs between the evaluation metrics.
    * Provide a final recommendation for the best model based on the combined analysis.

5.  **Brownie Points (Optional):**
    * Train additional models: **RandomForest** and **XGBoost**.
    * Implement an **InvertedClassifier** to demonstrate AUC < 0.5 behavior.
    * Include these models in the comparative evaluation.

---

## Requirements

-   Python 3.x
-   NumPy
-   Pandas
-   Matplotlib
-   Scikit-learn
-   XGBoost

---

## Results Summary (Including Optional Models)

The performance of the models was evaluated across multiple metrics:

| Model              | Accuracy | F1 (Weighted) | ROC-AUC (Macro) | AP (Macro) |
| :----------------- | :------- | :------------ | :-------------- | :--------- |
| **RandomForest**   | 0.9115   | 0.9094        | 0.9901          | 0.9517     |
| **XGBoost**        | 0.9055   | 0.9034        | 0.9900          | 0.9513     |
| KNN                | 0.9045   | 0.9037        | 0.9786          | 0.9217     |
| SVC                | 0.8955   | 0.8925        | 0.9850          | 0.9177     |
| LogisticRegression | 0.8395   | 0.8296        | 0.9755          | 0.8711     |
| DecisionTree       | 0.8505   | 0.8509        | 0.9002          | 0.7366     |
| GaussianNB         | 0.7965   | 0.8036        | 0.9551          | 0.8105     |
| Dummy(Prior)       | 0.2305   | 0.0864        | 0.5000          | 0.1667     |
| InvertedLogistic   | 0.0005   | 0.0004        | 0.0245          | 0.0901     |

**Conclusion:** RandomForest and XGBoost significantly outperform the baseline models. **RandomForest** shows the slight edge overall. The **InvertedLogistic** model successfully demonstrates performance worse than random (AUC < 0.5).

**Final Recommendation:** **RandomForest Classifier** is recommended due to its top performance across all key metrics. XGBoost is a very close second.

---
