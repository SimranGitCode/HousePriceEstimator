# House Price Estimator

A machine learning project for predicting **house prices in Bangalore**, based on area type, location, square footage, number of bathrooms, balconies, and BHK. The model was trained using a **Random Forest Regressor**.

---

## Dataset Overview

- **Initial Shape:** 13,320 rows × 9 columns  
  Columns: `area_type`, `availability`, `location`, `size`, `society`, `total_sqft`, `bath`, `balcony`, `price`  

- **Missing Values Count:**  
  - `location`: 1  
  - `size`: 16  
  - `society`: 5,502  
  - `bath`: 73  
  - `balcony`: 609  

---

## Data Cleaning

1. **Column Removal:** Dropped `availability` and `society` due to high null values.  
2. **Handling Missing Values:**  
   - Dropped rows with null `location`, `size`, or `bath`.  
   - Filled 536 null `balcony` values with mean.  
3. **Feature Formatting:** Extracted BHK from `size` column.  
4. **Location Simplification:** Grouped locations with <20 occurrences as `"others"`.  
5. **Total Sqft Cleaning:**  
   - Converted ranges and units to numeric square feet.  
   - Removed invalid entries.  
6. **Column Renaming:** `bath` → `bathrooms`.  
7. **New Feature:** `price_per_sqft = price / total_sqft`.

---

## Outlier Removal

- Removed unrealistic values for `total_sqft`, `balcony`, `BHK`, and `bathrooms`.  
- Applied BHK-to-Sqft rule: rows where `total_sqft / BHK < 300` were removed.  
- **Final Cleaned Dataset:** 12,335 rows × 8 columns  

---

## Correlation Analysis

- `total_sqft` most correlated with `bathrooms`, then `BHK`, then `balcony`.  
- Pearson correlation with price:  
  - `total_sqft`: 0.777  
  - `bathrooms`: 0.695  
  - `balcony`: 0.124  
  - `BHK`: 0.628  

---

## Additional Analysis

- Calculated average price for each `area_type` and `location`.  
- Created a pivot table of **area_type vs location** with average prices.  

---

## Model Preparation

- **Encoding:** One-Hot Encoding for `area_type` and `location`.  
- **Features Used:** `area_type`, `location`, `total_sqft`, `balcony`, `bathrooms`, `BHK`, `price_per_sqft`  
- **Target:** `price`  

---

## Model Training & Evaluation

| Model                     | MSE      | R²       | Cross-Val Score |
|----------------------------|----------|----------|----------------|
| Linear Regression          | 1033.06  | 0.6849   | -1.4035        |
| Decision Tree Regressor    | 329.13   | 0.8996   | 0.9613         |
| Random Forest Regressor    | 248.33   | 0.9242   | 0.9363         |

**Best Model:** Random Forest Regressor (highest R², lowest MSE)

---
**Dataset:** Kaggle dataset for Bangalore house prices.
