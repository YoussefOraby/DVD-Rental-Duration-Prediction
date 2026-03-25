# DVD-Rental-Duration-Prediction

## Project Overview

This project builds a machine learning model to predict how many days a customer will keep a rented DVD.
Accurate predictions help the rental company improve inventory planning and availability.

The target requirement for the project is to build a regression model that achieves:

**Mean Squared Error (MSE) ≤ 3 on the test set**

---

## Dataset

The dataset **`rental_info.csv`** contains historical DVD rental records.

### Features

| Column              | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| rental_date         | Date and time when the DVD was rented                      |
| return_date         | Date and time when the DVD was returned                    |
| amount              | Price paid by the customer                                 |
| rental_rate         | Standard rental rate                                       |
| release_year        | Year the movie was released                                |
| length              | Movie duration in minutes                                  |
| replacement_cost    | Cost to replace the DVD                                    |
| special_features    | Additional DVD features such as trailers or deleted scenes |
| NC-17, PG, PG-13, R | Dummy variables representing the movie rating              |

Additional engineered features already included in the dataset:

* `amount_2`
* `rental_rate_2`
* `length_2`

---

## Feature Engineering

Several preprocessing steps were performed before modeling.

### 1. Datetime Conversion

The columns `rental_date` and `return_date` were converted to datetime format.

### 2. Target Variable Creation

Rental duration was computed as:

```
rental_length_days = return_date − rental_date
```

This represents the number of days a DVD was rented.

### 3. Special Feature Encoding

The `special_features` column was converted into binary indicators:

* `deleted_scenes`
* `behind_the_scenes`

### 4. Column Removal

After feature engineering the following columns were removed:

* rental_date
* return_date
* rental_length
* special_features

### 5. Feature Scaling

Numerical variables were standardized using **StandardScaler**.

---

## Feature Selection

**Lasso Regression** was used to perform feature selection.

Lasso automatically reduces less important feature coefficients to zero, allowing the model to keep only the most relevant predictors.

---

## Models Tested

### Linear Regression

After feature selection using Lasso, a Linear Regression model was trained.

Performance:

```
MSE = 4.8459
```

This result did not meet the project requirement.

---

### Random Forest Regressor

A Random Forest model was trained and optimized using **RandomizedSearchCV**.

Hyperparameters searched:

* `n_estimators`
* `max_depth`

Best parameters found:

```
n_estimators = 51
max_depth = 10
```

---

## Final Model

Random Forest Regressor

```
RandomForestRegressor(
    n_estimators=51,
    max_depth=10,
    random_state=9
)
```

Model performance:

```
Test MSE = 2.2252
```

This satisfies the project requirement of **MSE ≤ 3**.

---

## Libraries Used

* pandas
* numpy
* scikit-learn

---

## Results

The Random Forest model significantly improved performance compared to Linear Regression and achieved an MSE well below the required threshold.

This model can help the company:

* estimate how long DVDs will stay rented
* optimize inventory planning
* improve availability of popular titles
