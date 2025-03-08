import pandas as pd


#Loading the processed data
data = pd.read_csv('updated_data.csv')

# 1. Melt the monthly data ONLY
# Reshape the DataFrame from wide to long format, focusing on monthly temperature anomalies.
# 'id_vars' specifies the columns to keep as identifiers (Year).
# 'value_vars' specifies the columns to unpivot (Jan, Feb, ..., Dec).
# 'var_name' renames the unpivoted columns to 'Month'.
# 'value_name' renames the values to 'Temp_Anomaly'.
monthly_data = data.melt(id_vars=["Year"], value_vars=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], var_name="Month", value_name="Temp_Anomaly")

# 2. Create the Date column
# Create a 'Date' column by combining 'Year' and 'Month' and converting to datetime objects.
# The format '%Y-%b' ensures that the year and abbreviated month names are correctly parsed.
monthly_data['Date'] = pd.to_datetime(monthly_data['Year'].astype(str) + '-' + monthly_data['Month'], format='%Y-%b')  # Correct format!

# 3. Create Season (from the Date)
# Create a 'Season' column based on the month extracted from the 'Date' column.
# A lambda function is used to assign seasons based on month numbers.
# The 'Season' column is then converted to a categorical data type.
monthly_data["Season"] = monthly_data["Date"].dt.month.apply(lambda x: "Winter" if x in [12, 1, 2] else "Spring" if x in [3, 4, 5] else "Summer" if x in [6, 7, 8] else "Autumn")
monthly_data["Season"] = monthly_data["Season"].astype("category")

# 4. Moving averages (using the Date)
# Calculate 5-year and 10-year moving averages of the 'Temp_Anomaly' column.
# 'rolling(window=n, center=True)' calculates the rolling mean with a window of 'n' and centers the window.
# 'fillna(method='bfill')' fills NaN values at the beginning and end of the series using backward fill.
monthly_data["5yr_moving_avg"] = monthly_data["Temp_Anomaly"].rolling(window=5, center=True).mean().fillna(method='bfill')
monthly_data["10yr_moving_avg"] = monthly_data["Temp_Anomaly"].rolling(window=10, center=True).mean().fillna(method='bfill')

# 5. Keep seasonal averages from the original data
seasonal_data = data[["Year", "DJF", "MAM", "JJA", "SON", "J-D", "D-N"]]  # Include all seasonal averages


# Saving the data
# 'index=False' prevents the index from being written to the CSV file.
monthly_data.to_csv("monthly_data.csv", index=False)
seasonal_data.to_csv("seasonal_averages.csv", index=False)

print(monthly_data.head())
print(seasonal_data.head())



