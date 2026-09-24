# Machine Learning Methodology & Engineering Decisions

**Project Title:** End-to-End MLOps Pipeline for House Price Prediction  
**Problem Formulation:** Supervised Tabular Regression

---

## 1. Problem Formulation

The objective is to learn a mapping function $f: \mathcal{X} \rightarrow \mathbb{R}^+$ that predicts the fair market sale price ($y \in \mathbb{R}^+$) of a residential property given an input feature vector $\mathbf{x} \in \mathcal{X}$.

Real estate appraisal is characterized by:
1. **Multi-collinearity:** Features like ground living area, basement square footage, and bathroom counts correlate strongly with one another.
2. **Non-linear Dynamics:** House prices compound non-linearly with overall quality rating and specific neighborhood premiums.
3. **Presence of Missing Values:** Certain physical amenities (e.g., basement area or garage capacity) may be absent or unrecorded in older homes.

---

## 2. Feature Schema & Selection Rationale

The feature space combines primary physical dimensions, construction dates, amenity counts, and categorical location descriptors:

| Feature Name | Type | Description | Rationale |
| :--- | :--- | :--- | :--- |
| `OverallQual` | Numerical (int 1-10) | Overall material & finish quality | Strongest single predictor of house price in real estate research. |
| `GrLivArea` | Numerical (float, sq ft) | Above ground living area | Direct measure of habitable interior space. |
| `TotalBsmtSF` | Numerical (float, sq ft) | Basement interior area | Adds substantial structural utility and storage value. |
| `GarageCars` | Numerical (int 0-4) | Garage car capacity | Important urban amenity strongly correlated with price. |
| `FullBath` | Numerical (int 1-4) | Number of full bathrooms | Functional accommodation metric for families. |
| `YearBuilt` | Numerical (int) | Original construction year | Captures building era, architectural standard, and depreciation. |
| `YearRemodAdd` | Numerical (int) | Remodel / renovation year | Adjusts valuation for modernized fixtures and structural updates. |
| `LotArea` | Numerical (float, sq ft) | Lot parcel size in sq ft | Measures parcel footprint and land value. |
| `Fireplaces` | Numerical (int 0-3) | Number of working fireplaces | Luxury amenity positively impacting buyer willingness to pay. |
| `Neighborhood` | Categorical (10 classes)| Physical location in Ames | Location dictates school districts, desirability, and baseline land values. |
| `BldgType` | Categorical (5 classes) | Dwelling type (1Fam, Townhouse, etc.) | Distinguishes detached homes from multi-family shared walls. |
| `HouseStyle` | Categorical (4 classes) | Architectural style (1-story, 2-story) | Reflects layout preference and living efficiency. |
| `CentralAir` | Categorical ('Y'/'N') | Central air conditioning | Essential climate control comfort amenity. |

---

## 3. Preprocessing & Data Leakage Prevention

Data leakage occurs when information from outside the training dataset is inadvertently used to fit the feature transformers. This artificially inflates offline validation scores but degrades online production performance.

### Anti-Leakage Protocol:
1. **Rigorous Split Prior to Fitting:** The dataset is partitioned into **Train (70%)**, **Validation (10%)**, and **Holdout Test (20%)** sets *before* computing summary statistics.
2. **Encapsulated Preprocessor:** The scikit-learn `ColumnTransformer` is fitted **exclusively on the training subset**:
   - **Numerical Imputation:** `SimpleImputer(strategy='median')` protects against skewness in square footage and year features.
   - **Standardization:** `StandardScaler()` centers numeric distributions to $\mu = 0, \sigma = 1$, vital for gradient stability and linear convergence.
   - **Categorical Imputation & One-Hot Encoding:** `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` ensures that previously unseen neighborhood or style categories encountered during live API inference do not cause runtime crashes.
3. **End-to-End Pipeline Artifact:** The fitted `preprocessor` and `regressor` are serialized together as a single composite `sklearn.pipeline.Pipeline`. Raw user inputs passed to FastAPI are transformed on-the-fly inside the pipeline.

---

## 4. Candidate Algorithms

To ensure rigorous model selection, four distinct regression paradigms were evaluated under identical cross-validation conditions:

1. **Ordinary Least Squares (Linear Regression):**
   - Serves as the transparent baseline.
   - Closed-form convex optimization; high interpretability with direct feature weight insights.
2. **Random Forest Regressor (Bagging Ensemble):**
   - Ensemble of unpruned decision trees trained on bootstrap samples of the data.
   - Reduces variance, naturally captures non-linear step functions, and resists outlier degradation.
3. **Gradient Boosting Regressor (Boosting Ensemble):**
   - Sequentially trains weak decision trees, where each subsequent tree minimizes the pseudo-residuals of the existing ensemble using gradient descent.
4. **XGBoost Regressor (Extreme Gradient Boosting):**
   - Advanced regularized gradient boosting library implementing exact tree split finding, $L_1$ and $L_2$ leaf weight penalties to mitigate overfitting, and cache-aware column block processing.

---

## 5. Evaluation Metrics

All candidate architectures were compared across three complementary statistical metrics:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
   - Measures average dollar error magnitude on an intuitive, linear scale without squaring deviations.

2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
   - Penalizes severe underestimations and overestimations heavily due to the quadratic term, essential for financial and valuation risk management.

3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
   - Quantifies the proportion of target variance explained by the model relative to a naive mean baseline.
