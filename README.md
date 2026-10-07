# Telco Customer Churn Prediction

A machine learning project for predicting customer churn using the IBM Telco Customer Churn dataset. The project focuses on exploratory data analysis, categorical feature encoding, data cleaning, multicollinearity analysis, imbalanced classification, model comparison, threshold tuning, hyperparameter optimization with Optuna, and experiment tracking with MLflow.

## Project Overview

Customer churn prediction can help a business identify customers who are likely to leave and target them with retention strategies.

This project treats churn prediction as an imbalanced classification problem and places particular emphasis on **recall for the churn class**, since missing a customer who is likely to churn can be more costly than incorrectly flagging a loyal customer.

## Dataset

The project uses the **IBM Telco Customer Churn** dataset.

Each row represents a customer and the dataset contains:

- Customer demographic information
- Services subscribed to
- Account and billing information
- Contract information
- Monthly and total charges
- Customer tenure
- Churn status

The target variable is:

- `Churn` — whether the customer left the company

## Project Workflow

The notebook follows this general workflow:

```text
Raw Dataset
     │
     ▼
Data Loading & Exploration
     │
     ▼
Categorical Feature Encoding
     │
     ├── Binary Encoding
     │
     └── One-Hot Encoding
     │
     ▼
Data Cleaning
     │
     ├── Convert TotalCharges to numeric
     ├── Remove customerID
     └── Convert boolean features to integers
     │
     ▼
Correlation Analysis
     │
     ▼
Multicollinearity Analysis (VIF)
     │
     ▼
Train/Test Split
     │
     ▼
Class Imbalance Handling
     │
     ▼
Model Comparison
     │
     ├── Random Forest
     ├── LightGBM
     └── XGBoost
     │
     ▼
Threshold Tuning
     │
     ▼
XGBoost Selection
     │
     ▼
Optuna Hyperparameter Optimization
     │
     ▼
MLflow Experiment Tracking
```

## Exploratory Data Analysis & Preprocessing

### Binary Encoding

Binary categorical variables were converted to numerical values.

Examples include:

- `Yes` → `1`
- `No` → `0`
- `Male` → `1`
- `Female` → `0`

The binary columns include:

- `gender`
- `Partner`
- `Dependents`
- `PhoneService`
- `PaperlessBilling`
- `Churn`

### One-Hot Encoding

Categorical variables with more than two categories were one-hot encoded using `pandas.get_dummies()`.

Features include:

- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaymentMethod`

`drop_first=True` was used to avoid redundant dummy variables.

### Data Cleaning

The notebook also:

- Converts `TotalCharges` to numeric using `pd.to_numeric(..., errors='coerce')`
- Removes the `customerID` column
- Converts boolean columns to integer values (`0`/`1`)

## Feature Analysis

### Correlation Analysis

Feature correlations with `Churn` were examined to identify variables associated with customer churn.

The notebook highlights several notable relationships:

| Feature | Correlation with Churn | Interpretation |
|---|---:|---|
| `tenure` | -0.35 | Longer-tenured customers are less likely to churn |
| `Contract_Two year` | -0.30 | Two-year contract customers churn substantially less |
| `Contract_One year` | -0.18 | One-year contracts are also associated with lower churn |
| `InternetService_Fiber optic` | +0.31 | Fiber optic customers show higher churn in this dataset |
| `PaymentMethod_Electronic check` | +0.30 | Electronic-check customers show higher churn in this dataset |

Correlation is used here for exploratory analysis and should not be interpreted as proof of causation.

### Multicollinearity

Variance Inflation Factor (VIF) was used to investigate multicollinearity among the features.

Redundant `No internet service` dummy variables were consolidated into a single:

```text
No_internet_service
```

Similarly, the redundant:

```text
MultipleLines_No phone service
```

feature was converted into:

```text
No_phone_service
```

The notebook found evidence of multicollinearity and discusses several possible approaches:

- Removing redundant features
- Regularization such as Lasso or Ridge
- Using tree-based models, which are less sensitive to multicollinearity

## Handling Class Imbalance

The dataset contains an imbalanced target, with churners representing approximately **27%** of the data.

Instead of immediately applying aggressive oversampling, the project uses class weighting / positive-class weighting and decision-threshold tuning.

For churn prediction, recall is emphasized because:

> A false negative means failing to identify a customer who actually churns.

If retention campaigns are relatively inexpensive, maximizing recall can be preferable even if it results in more false positives.

## Models

Three tree-based approaches were evaluated.

### 1. Random Forest

The Random Forest model uses:

- `n_estimators=300`
- `class_weight='balanced'`
- `random_state=42`
- `n_jobs=-1`

The prediction threshold was lowered from the default `0.5` to improve churn recall.

The notebook reports approximately **71.7% recall for churners** for the selected Random Forest threshold.

### 2. LightGBM

The LightGBM model uses:

- `n_estimators=500`
- `learning_rate=0.05`
- `class_weight='balanced'`
- `random_state=42`
- `n_jobs=-1`

The notebook reports approximately **82% recall for churners**, improving on the Random Forest model.

The trade-off is lower precision, meaning more customers who ultimately remain loyal may be flagged as potential churners.

### 3. XGBoost

The initial XGBoost model uses:

- `n_estimators=500`
- `learning_rate=0.05`
- `max_depth=6`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `scale_pos_weight` calculated from the training-set class distribution
- `random_state=42`
- `n_jobs=-1`

The project selects XGBoost because the notebook reports that it provides comparable recall to LightGBM while training approximately **3× faster**.

## Threshold Tuning

Rather than relying exclusively on the default classification threshold of `0.50`, the project evaluates:

```text
0.25
0.30
0.35
0.40
0.45
0.50
```

For each threshold, the following metrics are evaluated for the churn class:

- Precision
- Recall
- F1-score

A threshold of:

```python
THRESHOLD = 0.30
```

is used in the main experiments to prioritize recall.

This is particularly relevant for churn prediction because the business may prefer to contact additional customers rather than miss customers who are likely to leave.

## Hyperparameter Optimization with Optuna

After selecting XGBoost, **Optuna** is used to search for better hyperparameters.

The optimization explores:

- `n_estimators`
- `learning_rate`
- `max_depth`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `gamma`
- `reg_alpha`
- `reg_lambda`

The study runs:

```text
30 trials
```

The objective is to **maximize recall for the churn class** while keeping the classification threshold at the selected value.

## Final Tuned Model

The tuned XGBoost model achieved the following results reported in the notebook:

| Metric | Churn Class |
|---|---:|
| Recall | **0.928** |
| Precision | **0.442** |
| F1-score | **0.599** |

The tuned model catches approximately **92.8% of actual churners**.

However, precision decreases to approximately **44.2%**, meaning a substantial number of customers flagged as potential churners do not actually churn.

### Business Interpretation

The model is well suited to a retention strategy where:

- Missing a potential churner is expensive
- Retention campaigns are relatively inexpensive
- The business is willing to contact additional customers to capture more true churners

If retention campaigns are expensive, a higher decision threshold or an objective that balances precision and recall may be more appropriate.

## Experiment Tracking with MLflow

MLflow is used to track the tuned XGBoost experiment.

The notebook logs:

### Parameters

- Optimized XGBoost hyperparameters
- Class weighting configuration
- Random state
- Evaluation metric

### Metrics

- Training time
- Prediction time
- Precision
- Recall
- F1-score
- ROC-AUC

### Model

The trained XGBoost model is logged through MLflow.

The experiment is configured under:

```text
Telco Churn - XGBoost
```

and uses a local `mlruns` directory for tracking.

## Technologies Used

- **Python**
- **Pandas** — data manipulation and preprocessing
- **NumPy** — numerical operations
- **Matplotlib** — visualization
- **Seaborn** — correlation visualization
- **Scikit-learn** — preprocessing, train/test split, Random Forest, metrics
- **Statsmodels** — VIF / multicollinearity analysis
- **LightGBM** — gradient boosting classifier
- **XGBoost** — gradient boosting classifier
- **Optuna** — hyperparameter optimization
- **MLflow** — experiment tracking and model logging
- **Jupyter Notebook** — experimentation and analysis

## Installation

Clone the repository and install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels lightgbm xgboost optuna mlflow jupyter
```

Then launch Jupyter:

```bash
jupyter notebook
```

Open the project notebook and run the cells sequentially.

## Data Paths

The notebook currently references local Windows paths such as:

```text
D:data/raw/Telco-Customer-Churn.csv
D:/data/processed/telco_churn_processed.csv
```

These paths are specific to the development environment.

If you clone this repository, update the paths or place the dataset in the corresponding project directories.

## MLflow

After running the MLflow experiment, the tracking data is stored in the project's:

```text
mlruns/
```

directory.

The experiment can be inspected using:

```bash
mlflow ui
```

Then open the MLflow UI in your browser.

## Key Takeaways

- Customer churn is treated as an imbalanced classification problem.
- Recall is prioritized because failing to identify a potential churner can be costly.
- Categorical variables are handled using binary encoding and one-hot encoding.
- VIF is used to identify multicollinearity.
- Random Forest, LightGBM, and XGBoost are compared.
- Decision-threshold tuning is used to improve churn detection.
- XGBoost is selected based on the notebook's reported balance of recall and training speed.
- Optuna is used to optimize XGBoost hyperparameters.
- The tuned XGBoost model achieves **0.928 recall** for the churn class in the notebook's reported evaluation.
- MLflow is used to track parameters, metrics, training/prediction time, and the trained model.

## Project Structure

A suitable repository structure for this project is:

```text
telco-customer-churn/
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   └── processed/
│       └── telco_churn_processed.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── mlruns/
│
├── README.md
└── requirements.txt
```

## Future Improvements

Potential next steps include:

- Add a reproducible preprocessing pipeline using `sklearn.pipeline`
- Separate EDA, preprocessing, training, and evaluation into reusable Python modules
- Perform cross-validation instead of relying on a single train/test split
- Optimize the decision threshold using an explicit business cost function
- Compare ROC-AUC and PR-AUC across models
- Add feature importance / SHAP analysis
- Create a model inference script or API
- Add automated model evaluation and experiment configuration
- Improve data-path handling so the project runs independently of the original local environment
