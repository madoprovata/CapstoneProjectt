import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Loading the data from previously processed CSV file
monthly_data = pd.read_csv("monthly_data.csv")
monthly_data['Date'] = pd.to_datetime(monthly_data['Date'])

# Remove pre-existing one hot encoded columns
columns_to_remove = ['Season_Spring', 'Season_Summer', 'Season_Winter', 'Season_Spring.1', 'Season_Summer.1', 'Season_Winter.1', 'Season_Spring.2', 'Season_Summer.2', 'Season_Winter.2', 'Season_Spring.3', 'Season_Summer.3', 'Season_Winter.3']
for col in columns_to_remove:
    if col in monthly_data.columns:
        monthly_data.drop(col, axis=1, inplace=True)

# Feature Engineering for Model Input
monthly_data['Date'] = pd.to_datetime(monthly_data['Date'])
monthly_data['Month_Num'] = monthly_data['Date'].dt.month
if 'Season' not in monthly_data.columns:
    def get_season(month):
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
        else:
            return 'Winter'
    monthly_data['Season'] = monthly_data['Month_Num'].apply(get_season)

monthly_data = pd.get_dummies(monthly_data, columns=['Season'], drop_first=True)

# Define the feature and target variables
features = ['Year', 'Month_Num', 'Season_Spring', 'Season_Summer', 'Season_Winter']
X = monthly_data[features]
y = monthly_data['Temp_Anomaly']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cross-validation setup
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# MODEL TRAINING AND INITIAL EVALUATION
# 1. Linear Regression
lr_model = LinearRegression()
lr_scores = cross_val_score(lr_model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# 2. Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_scores = cross_val_score(rf_model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# 3. XGBoost
xgb_model = xgb.XGBRegressor(random_state=42)
X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values
xgb_scores = cross_val_score(xgb_model, X_train_np, y_train_np, cv=kfold, scoring='neg_mean_squared_error')
xgb_model.fit(X_train_np, y_train_np)
xgb_predictions = xgb_model.predict(X_test_np)

# Evaluation Function (Initial)
def evaluate_model(model_name, model, predictions, scores):
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    avg_rmse = (-scores.mean())**0.5
    print(f"{model_name}:")
    print(f"  RMSE: {mse**0.5:.4f}")
    print(f"  R-squared: {r2:.4f}")
    print(f"  Avg. CV RMSE: {avg_rmse:.4f}")
    print("-" * 30)

# Evaluate the initial models
evaluate_model("Linear Regression", lr_model, lr_predictions, lr_scores)
evaluate_model("Random Forest", rf_model, rf_predictions, rf_scores)
evaluate_model("XGBoost", xgb_model, xgb_predictions, xgb_scores)

# Hyperparameter Tuning with RandomizedSearchCV
# 2. Random Forest Regressor (Hyperparameter tuning with RandomizedSearchCV)
rf_model = RandomForestRegressor(random_state=42)
rf_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}
rf_random_search = RandomizedSearchCV(rf_model, rf_param_grid, n_iter=20, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
rf_random_search.fit(X_train, y_train)
best_rf_model = rf_random_search.best_estimator_
rf_predictions = best_rf_model.predict(X_test)

# Saving the model using joblib
joblib.dump(best_rf_model, "rf_model.joblib")
print("Random Forest model saved successfully using joblib!")

# Test load using joblib
try:
    test_rf_model = joblib.load("rf_model.joblib")
    print("Test load of rf_model.joblib successful!")
except Exception as e:
    print(f"Test load of rf_model.joblib failed: {e}")

# 3. XGBoost (Hyperparameter tuning)
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.3, 0.5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}
xgb_random_search = RandomizedSearchCV(xgb_model, xgb_param_grid, n_iter=20, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
xgb_random_search.fit(X_train_np, y_train_np)
best_xgb_model = xgb_random_search.best_estimator_
xgb_predictions = best_xgb_model.predict(X_test_np)

# Evaluation FINAL (using MAE, MSE, R-squared)
def evaluate_model(model_name, model, predictions):
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5
    r2 = r2_score(y_test, predictions)
    print(f"{model_name}:")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R-squared: {r2:.4f}")
    print("-" * 30)

print("Linear Regression Evaluation:")
evaluate_model("Linear Regression", lr_model, lr_predictions)

print("\nRandom Forest Evaluation:")
evaluate_model("Random Forest", best_rf_model, rf_predictions)

print("\nXGBoost Evaluation:")
evaluate_model("XGBoost", best_xgb_model, xgb_predictions)

# Model Evaluation and Comparison
def evaluate_model_with_plots(model_name, model, predictions):
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5
    r2 = r2_score(y_test, predictions)
    print(f"{model_name}:")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R-squared: {r2:.4f}")
    print("-" * 30)

    # Residual Plot
    residuals = y_test - predictions
    plt.figure(figsize=(8, 6))
    sns.residplot(x=predictions, y=residuals, lowess=True, line_kws=dict(color="r"))
    plt.title(f"{model_name} - Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.show()

    # Prediction vs. Actual Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(f"{model_name} - Prediction vs. Actual")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()

# Use the plotting function
print("Linear Regression Evaluation:")
evaluate_model_with_plots("Linear Regression", lr_model, lr_predictions)

print("\nRandom Forest Evaluation:")
evaluate_model_with_plots("Random Forest", best_rf_model, rf_predictions)

print("\nXGBoost Evaluation:")
evaluate_model_with_plots("XGBoost", best_xgb_model, xgb_predictions)

# Load and save linear and random forest models.
joblib.dump(lr_model, "lr_model.joblib") # Using joblib for consistency
joblib.dump(best_rf_model, "rf_model.joblib")
best_xgb_model.get_booster().save_model("xgb_model.xgb")

# Save the preprocessed data
X.to_csv("X_data.csv", index=False)
y.to_csv("y_data.csv", index=False)
# If you need the original monthly_data for plotting:
monthly_data.to_csv("monthly_data.csv", index=False)

# Test Random Forest Model (Simplified)
import joblib
from sklearn.ensemble import RandomForestRegressor

# Create a simple model
rf_model_test = RandomForestRegressor()
rf_model_test.fit([[0, 0], [1, 1]], [0, 1])

# Save the model
joblib.dump(rf_model_test, "test_rf_model.joblib")
print("Test model saved.")

# Load the model
try:
    loaded_model = joblib.load("test_rf_model.joblib")
    print("Test model loaded.")
except Exception as e:
    print(f"Error loading test model: {e}")

