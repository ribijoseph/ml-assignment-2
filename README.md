# 🛒 Online Shopper Purchase Prediction – ML Assignment 2

**Live Streamlit app:** `https://ml-assignment-2-h6biaoxgu8x7gaycu3ecsh.streamlit.app/`

## a. Problem Statement

An online retailer wants to know, from the way a visitor browses the site during a single session, whether that session will end in a purchase. This is a **binary classification** problem: given 17 behavioural and session attributes, predict `Revenue` (1 = the visitor bought something, 0 = no purchase).

Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine) are trained on the same data and compared on Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC). The trained models are served through an interactive Streamlit app, where a user uploads test data, picks a model and sees its metrics, confusion matrix and classification report. Nothing is retrained in the app.

## b. Dataset Description

| Item | Detail |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository (ID 468): https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Instances | 12,330 sessions (12,205 after removing 125 exact duplicates) |
| Features | 17 (10 numerical, 7 categorical) |
| Target | `Revenue` (True/False) |
| Class balance | 10,422 no-purchase (84.5%) vs 1,908 purchase (15.5%), so the data is **imbalanced** |

Each row is one session by a different user over a one-year period.

**Numerical features**
- `Administrative`, `Informational`, `ProductRelated`: number of pages of each type visited
- `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`: time spent (seconds) on each page type
- `BounceRates`: average bounce rate of the pages visited (Google Analytics)
- `ExitRates`: average exit rate of the pages visited
- `PageValues`: average value of the pages visited before a transaction
- `SpecialDay`: closeness of the visit to a special day such as Valentine's Day or Mother's Day (0–1)

**Categorical features**
- `Month`: month of the visit
- `OperatingSystems`, `Browser`, `Region`, `TrafficType`: integer-coded categories
- `VisitorType`: Returning / New / Other
- `Weekend`: whether the visit was on a weekend

**Preprocessing (inside each saved pipeline)**
- Numerical columns go through `StandardScaler`.
- Categorical columns, including the integer-coded ones because their numbers are IDs and not quantities, go through `OneHotEncoder(handle_unknown="ignore")`.
- The split is stratified 80/20: 9,764 training rows and 2,441 test rows, with `random_state=461`.
- The 2,441 test rows are saved as `test_data.csv`.

## c. GitHub Repository Link

`<paste your GitHub repo URL here>`

```
ml-assignment-2/
├── app.py                 # Streamlit application (inference only)
├── train_models.py        # preprocessing, tuning, training, evaluation, saving
├── requirements.txt
├── README.md
├── test_data.csv          # held-out 20% test split (features + Revenue)
├── data/
│   └── online_shoppers_intention.csv   # raw UCI dataset
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── best_params.json   # hyper-parameters chosen by 5-fold CV
    └── test_metrics.csv   # metrics table below
```

**Reproduce**
```bash
pip install -r requirements.txt
python train_models.py      # retrains and overwrites model/ and test_data.csv
streamlit run app.py
```

## d. Models Used

Every model is a scikit-learn `Pipeline` (preprocessing followed by the classifier). Hyper-parameters were chosen by 5-fold `GridSearchCV` on the training set, using ROC-AUC as the score.

| Model | Tuned hyper-parameters (best) |
|---|---|
| Logistic Regression | `C = 0.1`, `class_weight = "balanced"` |
| Decision Tree | `max_depth = 6`, `min_samples_leaf = 50` |
| KNN | `n_neighbors = 31`, `weights = "distance"` |
| Naive Bayes (Gaussian) | `var_smoothing = 0.5` |
| SVM (RBF kernel) | `C = 0.5`, `class_weight = "balanced"` |

### Comparison Table (test set, 2,441 sessions)

Precision, Recall and F1 are for the positive class (purchase).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8488 | 0.9035 | 0.5118 | 0.7382 | 0.6045 | 0.5280 |
| Decision Tree | **0.9021** | **0.9245** | 0.7698 | 0.5340 | 0.6306 | 0.5891 |
| KNN | 0.8820 | 0.8704 | **0.8357** | 0.3063 | 0.4483 | 0.4611 |
| Naive Bayes | 0.8386 | 0.8291 | 0.4819 | 0.4188 | 0.4482 | 0.3554 |
| Support Vector Machine | 0.8710 | 0.9147 | 0.5626 | **0.7880** | **0.6565** | **0.5922** |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A strong linear baseline, with AUC 0.90. Because of `class_weight="balanced"`, it trades precision (0.51) for high recall (0.74): it finds most buyers but raises many false alarms. Its accuracy (0.85) is the second lowest, which shows that accuracy is misleading on a dataset that is 84% "no purchase". The best `C` was the smallest one tried (0.1), meaning strong regularisation helped. This suggests the useful signal comes from a few features. |
| Decision Tree | Highest accuracy (0.902) and AUC (0.925). Feature importance shows that **`PageValues` alone accounts for about 76% of the splits**, followed by `BounceRates` and `Month = Nov`. The tree captures the sharp threshold behaviour of `PageValues` that a linear model can only approximate. Limiting depth to 6 with ≥50 samples per leaf kept it from overfitting. Without class weighting it is conservative: precision is high (0.77) but recall is only 0.53. |
| KNN | Highest precision (0.84) but the lowest recall (0.31): it only predicts "purchase" when the nearest neighbours are clearly buyers. Because the classes are imbalanced, most neighbourhoods are dominated by non-buyers, so many real buyers are missed. The 65 one-hot columns (75 dimensions in total) also make Euclidean distance less informative. It has the third-lowest MCC (0.46), and it is the largest model file (about 6 MB) because KNN stores the whole training set. |
| Naive Bayes | The weakest model, with MCC 0.36 and AUC 0.83. Its independence assumption is clearly violated: page counts and their durations are strongly correlated (for example `ProductRelated` and `ProductRelated_Duration`, and `BounceRates` and `ExitRates`). Gaussian likelihoods also fit 0/1 one-hot columns and heavily skewed durations poorly. With the default smoothing it predicted almost every session as a purchase (accuracy about 0.25). Raising `var_smoothing` to 0.5 was necessary to make it usable. |
| Support Vector Machine | **Best F1 (0.657) and best MCC (0.592)**, with the highest recall (0.79) and AUC 0.91, second only to the tree. The RBF kernel models non-linear boundaries, and balanced class weights stop it from ignoring the minority class. Its downsides are the slowest training (most of the 3-minute training run) and the need for Platt scaling (`probability=True`) to produce AUC scores. |
| **Overall winner** | **Support Vector Machine.** On an imbalanced dataset, MCC and F1 are the fairest single numbers, and SVM leads on both. It also catches about 79% of buyers. The **Decision Tree** is a very close second (MCC 0.589) and wins on accuracy, AUC and interpretability, so it would be the practical choice if a business needed to explain predictions. Naive Bayes is clearly unsuitable for this data. |

## Streamlit App Features

- **CSV upload** of test data in the sidebar, plus a button to download the sample `test_data.csv`
- **Model selection dropdown** covering all five classifiers
- **Evaluation metrics** for the selected model: Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion matrix** heatmap and **classification report** table
- A **comparison tab** that runs all five saved models on the uploaded file, with a metric bar chart and the best model by MCC
- Checks that the uploaded file contains all required columns, and accepts `TRUE/FALSE`, `True/False` or `1/0` for the boolean columns
