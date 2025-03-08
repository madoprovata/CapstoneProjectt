import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#DATA LOADING
data = pd.read_csv(r"C:\Users\proma\OneDrive\DataScience\PycharmProjects\Capstone_Project\Global_Temp.csv")

#CONVERTING object columns to numeric data types.
#'errors = 'coerce'' will replace invalid parsing with NaN.
data['D-N'] = pd.to_numeric(data['D-N'], errors = 'coerce')
data['DJF'] = pd.to_numeric(data['DJF'], errors = 'coerce')

#HANDLING MISSING VALUES
#Calculate the mean of 'D-N' and 'DJF' columns
#Assumptions: missing values in these columns can be imputed with the mean.
mean_dn = data['D-N'].mean()
mean_djf = data['DJF'].mean()

#Fillinf NaN values with their respective means.
data['D-N'].fillna(mean_dn, inplace=True)
data['DJF'].fillna(mean_djf, inplace=True)

#DATA INSPECTION
print(data.info())
print('First 5 rows:')
print(data.head())

#DATA CLEANING
#Removing duplicate rows for DataFrame
data.drop_duplicates(inplace = True)
#Assumptions: Forward fill is appropriate for the time-series nature of the data
data.fillna(method='ffill', inplace=True)

#REMOVING OUTLIERS
from sklearn.ensemble import IsolationForest

# Initialize the IsolationForest model with a contamination rate of 5%.
# Contamination represents the proportion of outliers in the dataset.
model = IsolationForest(contamination = 0.05)
model.fit(data[['Jan']])

#Predict outliers and create a new 'outlier' column indicating whether a row is an outlier (-1) or not (1)
data['outlier'] = model.predict(data[['Jan']])

#Extracting rows identified as outliers
outliers = data[data['outlier'] == -1]
print(f"Number of outliers (Isolation Forest): {len(outliers)}")
print(outliers)

#Removing outlier column from DataFrame
data.drop('outlier', axis=1, inplace=True)

#REMOVING ROWS identified as outliers
data = data.drop(outliers.index)
print(f'DataFrame shape after outlier removal: {data.shape}')


#Saving File
output_filename = 'updated_data.csv'
data.to_csv(output_filename, index = False)
print(f'File saved to {output_filename}')