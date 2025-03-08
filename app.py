import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# --- 1. Model and Data Loading ---
# Load the trained models and data.
try:
    lr_model = joblib.load("lr_model.pkl")  # Load Linear Regression model
    rf_model = joblib.load("rf_model.joblib")  # Load Random Forest model

    # Load XGBoost model
    xgb_booster = xgb.Booster()  # Create an XGBoost Booster object
    xgb_booster.load_model("xgb_model.xgb")  # Load the trained XGBoost model
    xgb_model = xgb.XGBRegressor()  # Create an XGBoost Regressor object
    xgb_model._Booster = xgb_booster  # Assign the loaded Booster to the Regressor

    monthly_data = pd.read_csv("monthly_data.csv")  # Load monthly data
    X_data = pd.read_csv("X_data.csv")  # Load feature data
    y_data = pd.read_csv("y_data.csv")  # Load target data
    monthly_data['Date'] = pd.to_datetime(monthly_data['Date'])  # Convert 'Date' to datetime
    seasonal_data = pd.read_csv("seasonal_averages.csv") #Load seasonal data.

except FileNotFoundError as e:
    st.error(f"Error loading model or data: {e}. Please ensure model and data files are in the correct location.")
    st.stop() # Stop the app if model or data is not found.
# Assumption: The model and data files are in the same directory as the script.
# Explanation: This section loads the trained models and necessary data for the application.
# If any file is missing, an error message is displayed, and the app is stopped.

# --- 2. Streamlit App Setup ---
# Streamlit App
st.title("Global Temperature Anomaly Prediction")  # Set the app title

# --- 3. User Input ---
# User Input
year = st.slider("Select Year", min_value=1880, max_value=2050, value=monthly_data['Year'].max()) # Year selection slider
month = st.selectbox("Select Month", monthly_data['Month'].unique()) # Month selection dropdown
model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest", "XGBoost"]) # Model selection dropdown
# Explanation: This section provides input widgets for the user to select the year, month, and model.

# --- 4. Data Preparation ---
# Prepare Input Data
month_map = { # Mapping month names to numbers
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}
month_num = month_map[month] # Get the month number

def get_season_values(month_num): # Function to determine season values
    if month_num in [3, 4, 5]:
        return 1, 0, 0 # Spring
    elif month_num in [6, 7, 8]:
        return 0, 1, 0 # Summer
    elif month_num in [9, 10, 11]:
        return 0, 0, 0 # Autumn
    else:
        return 0, 0, 1 # Winter

season_spring, season_summer, season_winter = get_season_values(month_num) # Get season values

input_features = pd.DataFrame({ # Create a DataFrame with input features
    'Year': [year],
    'Month_Num': [month_num],
    'Season_Spring': [season_spring],
    'Season_Summer': [season_summer],
    'Season_Winter': [season_winter]
})
# Explanation: This section prepares the input data for the selected year and month.
# It converts the month name to a numerical value and determines the season values.

# --- 5. Prediction ---
# Initialize future_data outside the if block
future_data = [] # Initialize list to store future predictions

#Always predict.
# Predict
if model_choice == "Linear Regression":
    prediction = lr_model.predict(input_features)[0] # Predict using Linear Regression
elif model_choice == "Random Forest":
    prediction = rf_model.predict(input_features)[0] # Predict using Random Forest
else:  # XGBoost
    prediction = xgb_model.predict(input_features.values)[0] # Predict using XGBoost
# Explanation: This section makes predictions based on the selected model and input features.

# --- 6. Display Prediction ---
# Display Prediction
st.subheader(f"Predicted Temperature Anomaly ({model_choice})") # Display the model used
st.write(f"The predicted temperature anomaly for {month} {year} is: {prediction:.4f}") # Display the prediction
# Explanation: This section displays the predicted temperature anomaly.

# --- 7. Plot Predicted vs. Actual ---
# Plot Predicted vs. Actual
st.subheader("Predicted vs. Actual Temperature Anomalies")
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Temp_Anomaly', data=monthly_data, label='Actual') # Plot actual data

# Predict for all years and months
if model_choice == "Linear Regression":
    predictions = lr_model.predict(X_data) # Predict using Linear Regression
elif model_choice == "Random Forest":
    predictions = rf_model.predict(X_data) # Predict using Random Forest
else:  # XGBoost
    predictions = xgb_model.predict(X_data.values) # Predict using XGBoost

monthly_data['Predicted_Anomaly'] = predictions # Add predicted values to the DataFrame

sns.lineplot(x='Date', y='Predicted_Anomaly', data=monthly_data, label='Predicted') # Plot predicted data
plt.title("Actual vs. Predicted Temperature Anomalies")
plt.xlabel("Date")
plt.ylabel("Temperature Anomaly")
st.pyplot(plt) # Display the plot
# Explanation: This section plots the actual and predicted temperature anomalies over time.

# --- 8. Insights (Future Predictions) ---
# Insights
st.subheader("Insights")
future_years = [year + i for i in range(1, 4)] # Generate future years
future_predictions = []

for future_year in future_years: # Loop through future years
    future_input_features = pd.DataFrame({ # Create input features for future year
        'Year': [future_year],
        'Month_Num': [month_num],
        'Season_Spring': [season_spring],
        'Season_Summer': [season_summer],
        'Season_Winter': [season_winter]
    })
    if model_choice == "Linear Regression":
        future_prediction = lr_model.predict(future_input_features)[0] # Predict using Linear Regression
    elif model_choice == "Random Forest":
        future_prediction = rf_model.predict(future_input_features)[0] # Predict using Random Forest
    else:
        future_prediction = xgb_model.predict(future_input_features.values)[0] # Predict using XGBoost
    future_predictions.append(future_prediction) # Append prediction to list
    future_data.append({'Year': future_year, 'Month': month, 'Predicted_Anomaly': future_prediction}) # Append data to list

st.write(f"Expected temperature anomaly changes in the next few years ({', '.join(map(str, future_years))}): {', '.join(map(lambda x: f'{x:.4f}', future_predictions))}")
# Explanation: This section generates future predictions for the next three years and displays them.

# --- 9. Model Confidence (Feature Importance) ---
# Model Confidence (simplified - using feature importance)
st.subheader("Model Confidence")
if model_choice == "Random Forest":
    feature_importance = pd.Series(rf_model.feature_importances_, index=input_features.columns) # Get feature importance
    st.write("Feature Importance (Random Forest):")
    st.write(feature_importance) # Display feature importance
elif model_choice == 'XGBoost':
    feature_importance = pd.Series(xgb_model.feature_importances_, index=input_features.columns) # Get feature importance
    st.write("Feature Importance (XGBoost):")
    st.write(feature_importance) # Display feature importance
else: # Linear Regression
    coefficients = pd.Series(lr_model.coef_, index=input_features.columns)
    st.write("Coefficients (Linear Regression):")
    st.write(coefficients)

# Download CSVs
st.subheader("Download Data")

# Future Predictions CSV
future_df = pd.DataFrame(future_data)
future_df['Date'] = pd.to_datetime(future_df['Year'].astype(str) + '-' + future_df['Month'], format='%Y-%b')

csv_future = future_df.to_csv(index=False).encode('utf-8')
future_df.to_csv("future_predictions.csv", index=False)
print("future_predictions.csv saved by app.py")
st.download_button(
    label="Download Future Predictions (CSV)",
    data=csv_future,
    file_name='future_predictions.csv',
    mime='text/csv',
    key="future_predictions_download"
)

# Seasonal Data CSV
csv_seasonal = seasonal_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Seasonal Averages (CSV)",
    data=csv_seasonal,
    file_name='seasonal_averages.csv',
    mime='text/csv',
    key="seasonal_averages_download"
)

# Monthly Data CSV
csv_monthly = monthly_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Monthly Data (CSV)",
    data=csv_monthly,
    file_name='monthly_data.csv',
    mime='text/csv',
    key="monthly_data_download"
)