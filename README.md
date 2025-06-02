# House Prices Prediction

This project explores advanced regression models for predicting house prices using the **Ames Housing Dataset** from the [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). The goal is to identify key factors influencing house prices and build predictive models that generalize well on unseen data.

## 📊 Problem Statement

**Management Question**:  
How can we predict house sale prices accurately using property characteristics, and what are the key factors that influence these prices in Ames, Iowa?

This problem is critical for real estate agencies, property investors, and homeowners to make informed decisions about pricing and market strategy.

## 🔍 Dataset

The dataset includes:
- **Training Data**: 1,460 observations with 79 features (both numerical and categorical) and target variable `SalePrice`.
- **Test Data**: 1,459 observations without the target variable (for Kaggle evaluation).

Key features include:
- Property details (size, rooms, quality)
- Neighborhood and location attributes
- Year built, remodel status, and additional amenities

A detailed data dictionary is provided by Kaggle: [Data Description](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).

## 🧰 Methods and Workflow

### 1️⃣ Exploratory Data Analysis (EDA)
- **Missing Values Handling**:
  - Filled categorical missing values with `"None"` or mode.
  - Numerical missing values filled logically (e.g., `0` for area-based features, median `LotFrontage` by `Neighborhood`).
- **Outlier Detection**: Reviewed `GrLivArea` vs. `SalePrice` scatterplots.
- **Distribution Analysis**: Visualized target and feature distributions using histograms and scatterplots.

### 2️⃣ Feature Engineering
- Created new features:
  - `TotalBath` (combined full and half baths)
  - `TotalPorchArea`
  - `HouseAge`, `HasFireplace`, `Remodeled`
- Added polynomial features:
  - `GrLivArea^2`, `TotalBsmtSF^2`, `GrLivArea * TotalBsmtSF`
- Handled multicollinearity using **VIF** analysis and feature selection.

### 3️⃣ Feature Selection
- Selected numerical features based on correlation and F-test.
- Selected categorical features via target encoding, correlation analysis, and ANOVA significance testing.

### 4️⃣ Model Development
- **Baseline**: Linear Regression (OLS)
- **Advanced Models**:
  - Lasso Regression
  - Ridge Regression
  - ElasticNet (with hyperparameter tuning via Grid Search)
- **Cross-Validation**: 20-fold cross-validation for model evaluation.

### 5️⃣ Evaluation
- **Metrics**: RMSE on log-transformed target (`SalePrice`), R², residual plots.
- Kaggle metric: RMSE of log-transformed predictions.

## 📈 Key Results

- **OLS Model**: Mean CV RMSE ≈ 0.1352; R² ≈ 0.912 (strong baseline).
- **Feature Importance**: `OverallQual`, `GrLivArea`, `TotalBsmtSF`, and neighborhood-related features were among the top predictors.
