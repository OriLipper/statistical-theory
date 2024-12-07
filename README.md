# House Price Prediction Project

This repository contains an in-depth analysis of the determinants of house prices in the Indian real estate market. The project involves data preprocessing, exploratory data analysis (EDA), statistical testing, regression modeling, and diagnostics to understand the factors influencing house prices and predict them effectively.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Project Objectives](#project-objectives)
4. [Methodology](#methodology)
5. [Key Features and Determinants](#key-features-and-determinants)
6. [Model Performance](#model-performance)
7. [Limitations](#limitations)
8. [Future Work](#future-work)
9. [Repository Structure](#repository-structure)
10. [Setting Up the Environment](#setting-up-the-environment)
11. [How to Run](#how-to-run)
12. [Results and Visualizations](#results-and-visualizations)

---

## Introduction

The Indian real estate market has experienced significant growth, driven by urbanization, economic development, and increasing disposable incomes. Understanding the determinants of house prices is crucial for policymakers, investors, and homebuyers. This project aims to explore these factors and build a predictive model for house prices.

---

## Dataset Description

The dataset used in this project contains detailed information about properties, including:
- Number of bedrooms, bathrooms, floors, etc.
- Living area, lot area, and basement area.
- Condition, grade, and renovation details.
- Geographic features like latitude and longitude.
- Waterfront presence and views.

**Target Variable:**
- `Price`: The selling price of the house.

**Key Characteristics:**
- Contains **11,695 observations** and **20 features** after preprocessing.
- Missing values were handled using median and mode imputation.

---

## Project Objectives

1. **Exploratory Analysis:**
   - Understand the distribution and relationships among features.
   - Identify key determinants of house prices.

2. **Statistical Analysis:**
   - Perform hypothesis testing and compute confidence intervals for feature correlations.

3. **Modeling:**
   - Build an Ordinary Least Squares (OLS) regression model to quantify the impact of features on house prices.

4. **Diagnostics:**
   - Validate regression assumptions (e.g., normality, homoscedasticity, multicollinearity).
   - Identify influential observations using Cook's Distance.

---

## Methodology

1. **Data Cleaning:**
   - Removed irrelevant columns (e.g., `id`, `Date`).
   - Imputed missing values and standardized numerical features.

2. **Feature Engineering:**
   - Created new features:
     - `House_Age`: Difference between current year and built year.
     - `Time_Since_Renovation`: Years since the last renovation.
     - `Total_Area`: Combined living, basement, and lot areas.

3. **Exploratory Data Analysis:**
   - Visualized distributions, relationships, and correlations among features.

4. **Statistical Testing:**
   - Calculated Pearson correlations and applied Bonferroni correction.
   - Constructed confidence intervals for correlation coefficients.

5. **Regression Modeling:**
   - Built an OLS regression model with statistically significant features.
   - Evaluated the model using metrics such as MAE, RMSE, and \( R^2 \).

6. **Diagnostics:**
   - Assessed residuals for linearity, normality, and homoscedasticity.
   - Calculated Variance Inflation Factor (VIF) to detect multicollinearity.
   - Identified influential points using Cook's Distance.

---

## Key Features and Determinants

**Significant Features:**
- `Total_Area`, `Living Area`, `Grade of the House`, `Number of Bathrooms`, and `Waterfront Presence` are strong positive contributors.
- `Time_Since_Renovation` and `House_Age` negatively impact house prices.

**Insignificant Features:**
- `Longitude`, `Distance from the Airport`, and `Number of Schools Nearby` showed no significant correlation with `Price`.

---

## Model Performance

**Training Results:**
- **R-squared (R²):** 0.642
- **Adjusted R²:** 0.641

**Test Set Results:**
- **Mean Absolute Error (MAE):** \$130,063.70
- **Root Mean Squared Error (RMSE):** \$234,641.76
- **R-squared (R²):** 0.63

The model explains 63% of the variability in house prices and performs reasonably well, though improvements are needed for higher-priced properties.

---

## Limitations

1. **Multicollinearity:**
   - High VIF values for features like `Total_Area` and `Living Area` indicate redundancy.

2. **Heteroscedasticity:**
   - Residuals show increasing variance for higher prices.

3. **Non-Normal Residuals:**
   - Q-Q plot reveals deviations from normality, particularly in the tails.

4. **Influential Observations:**
   - Cook's Distance identified 499 influential points, potentially skewing the model.

---

## Future Work

1. **Address Multicollinearity:**
   - Remove or combine redundant features.
   - Explore dimensionality reduction techniques (e.g., PCA).

2. **Improve Assumptions:**
   - Apply log or Box-Cox transformations to stabilize variance and normalize residuals.

3. **Explore Advanced Models:**
   - Experiment with tree-based models (e.g., Random Forest, Gradient Boosting) to capture non-linear relationships.

4. **Incorporate Additional Features:**
   - Neighborhood-level details, market trends, and proximity to amenities.

---

## Repository Structure

House-Price-India-Analysis/
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── House Price India.csv
│   └── House Price India_clean.csv
├── notebooks/
│   └── Statistical_Theory.ipynb
├── results/
│   └── figures/
│       ├── bathrooms_distribution.png
│       ├── bedrooms_distribution.png
│       ├── cooks_distance.png
│       ├── correlation_matrix.png
│       ├── feature_correlations.png
│       ├── living_area_boxplot.png
│       ├── price_boxplot.png
│       ├── price_by_bedrooms.png
│       ├── price_distribution.png
│       ├── price_vs_living_area.png
│       ├── qq_plot_train.png
│       └── residuals_vs_fitted_train.png
└── scripts/
    └── data_cleaning.py



---

## Setting Up the Environment

1. **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    ```

2. **Activate the Virtual Environment:**

    - **On Windows:**
        ```bash
        venv\Scripts\activate
        ```

    - **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run Jupyter Notebooks:**
    ```bash
    pip install notebook
    jupyter notebook
    ```

5. **Deactivate the Environment (When Done):**
    ```bash
    deactivate
    ```

---

## How to Run

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/username/house-price-prediction.git
    cd house-price-prediction
    ```

2. **Install Dependencies:**
   
   Follow the steps outlined in the [Setting Up the Environment](#setting-up-the-environment) section.

3. **Run the Jupyter Notebooks:**
   
   Navigate to the `notebooks/` directory and open the desired notebook for analysis.

4. **View Results:**
   
   Visualizations and diagnostic plots are saved in the `results/figures/` directory.

---

## Results and Visualizations

Key visualizations include:

- **Price Distribution:** Shows the skewness of house prices.
- **Correlation Matrix:** Highlights relationships between features and Price.
- **Cook's Distance Plot:** Identifies influential observations.
- **Q-Q Plot:** Assesses normality of residuals.
- **Residuals vs Fitted Values:** Evaluates linearity and homoscedasticity.

By addressing the limitations and incorporating the proposed improvements, this project aims to enhance the predictive power and interpretability of the house price model.