import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Loading the data
monthly_data = pd.read_csv("monthly_data.csv")
seasonal_data = pd.read_csv("seasonal_averages.csv")

# Set the Seaborn style to 'whitegrid' for a clean background with gridlines.
sns.set_style('whitegrid')

# Print the data types of the columns in the DataFrame.
print(monthly_data.dtypes)

# Convert the 'Date' column to datetime format.
monthly_data['Date'] = pd.to_datetime(monthly_data['Date'])

# 1. Global Temperature Anomaly Trends
# Create a figure and axes for the plot.
plt.figure(figsize=(12, 8))

# Plot the 'Temp_Anomaly' column against the 'Date' column.
plt.plot(monthly_data['Date'], monthly_data['Temp_Anomaly'], label='Monthly Anomalies')

# Plot the '5yr_moving_avg' column against the 'Date' column.
plt.plot(monthly_data['Date'], monthly_data['5yr_moving_avg'], label='5-Year Moving Avg')

# Plot the '10yr_moving_avg' column against the 'Date' column.
plt.plot(monthly_data['Date'], monthly_data['10yr_moving_avg'], label='10-Year Moving Avg')

# Set the x-axis label.
plt.xlabel('Date')

# Set the y-axis label.
plt.ylabel('Temperature Anomaly')

# Set the x-axis major tick locator to show every 20 years.
plt.gca().xaxis.set_major_locator(mdates.YearLocator(20))

# Set the x-axis major tick formatter to display the year in YYYY format.
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Set the title of the plot.
plt.title('Global Temperature Anomaly Trends')

# Add a legend to the plot.
plt.legend(loc='upper left')

# Add a super title to the plot.
plt.suptitle('Global Temperature Anomaly Trends\n(1880-2024)', fontsize=12)

# Turn off gridlines.
plt.grid(False)

# Adjust the layout of the plot to prevent labels from overlapping.
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Set the x-axis limits to the minimum and maximum dates in the 'Date' column.
start_year = monthly_data['Date'].min().year
end_year = monthly_data['Date'].max().year
plt.xlim(pd.to_datetime(f'{start_year}-01-01'), pd.to_datetime(f'{end_year}-12-31'))

# Rotate the x-axis tick labels by 45 degrees.
plt.xticks(rotation=45)

# Display the plot.
plt.show()

# 2. Comparing Anomalies for Different Months and Seasons

# Months
plt.figure(figsize=(12, 6))
for month in monthly_data['Month'].unique():
    month_data = monthly_data[monthly_data['Month'] == month]
    plt.plot(month_data['Date'], month_data['Temp_Anomaly'], label=month)
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly')
plt.title('Temperature Anomalies by Month')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(False)
plt.tight_layout()
plt.show()

# Seasons
plt.figure(figsize=(12, 6))
for season in monthly_data['Season'].unique():
    season_data = monthly_data[monthly_data['Season'] == season]
    plt.plot(season_data['Date'], season_data['Temp_Anomaly'], label=season)
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly')
plt.title('Temperature Anomalies by Season')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Place legend outside
plt.grid(False)
plt.tight_layout()
plt.show()

# 3. Distribution of Temperature Anomalies
plt.figure(figsize=(10, 6))
sns.histplot(monthly_data['Temp_Anomaly'], kde=True)
plt.xlabel('Temperature Anomaly')
plt.ylabel('Frequency')
plt.title('Distribution of Temperature Anomalies')
plt.tight_layout()
plt.grid(False)
plt.show()

# Distribution over years
plt.figure(figsize=(12, 6))
years = [1980, 1990, 2000, 2010, 2021]
for year in years:
    year_data = monthly_data[monthly_data['Year'] == year]
    sns.kdeplot(year_data['Temp_Anomaly'], label=year)
plt.xlabel('Temperature Anomaly')
plt.ylabel('Density')
plt.title(f'Distribution of Temperature Anomalies in Selected Years')
plt.legend()
plt.tight_layout()
plt.grid(False)
plt.show()


# 4. Correlation Heatmap

monthly_corr_data = monthly_data.pivot_table(index='Year', columns='Month', values='Temp_Anomaly')
monthly_corr = monthly_corr_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(monthly_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Monthly Temperature Anomalies')
plt.tight_layout()
plt.grid(False)
plt.show()

# Seasonal Correlations
seasonal_corr = seasonal_data[["DJF", "MAM", "JJA", "SON"]].corr() # Select only the seasonal columns
plt.figure(figsize=(8, 6))
sns.heatmap(seasonal_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Seasonal Temperature Anomalies')
plt.tight_layout()
plt.show()