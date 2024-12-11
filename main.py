import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm  # Importing statsmodels
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sns.set()

# Load data
cardata = pd.read_csv('Midterm/cars.csv')

# Basic data description
cardata.describe(include='all')

# Dropping the 'Model' column
data = cardata.drop(['Model'], axis=1)
data.describe(include='all')

# Drop rows with missing values
data_no_rv = data.dropna(axis=0)

# Reset the index after dropping rows
data_no_rv = data_no_rv.reset_index(drop=True)
data_no_rv.describe(include='all')

# Deal with outliers in Price
q = data_no_rv['Price'].quantile(0.99)
data_price_in = data_no_rv[data_no_rv['Price'] < q]
data_price_in = data_price_in.reset_index(drop=True)  # Reset index after filtering

# Deal with outliers in Mileage
q = data_no_rv['Mileage'].quantile(0.99)
data_mileage_in = data_no_rv[data_no_rv['Mileage'] < q]
data_mileage_in = data_mileage_in.reset_index(drop=True)  # Reset index after filtering

# Deal with outliers in EngineV
q = data_no_rv['EngineV'].quantile(0.99)
data_enginev_in = data_no_rv[data_no_rv['EngineV'] < q]
data_enginev_in = data_enginev_in.reset_index(drop=True)  # Reset index after filtering

# Deal with outliers in Year
q = data_no_rv['Year'].quantile(0.99)
data_year_in = data_no_rv[data_no_rv['Year'] < q]
data_year_in = data_year_in.reset_index(drop=True)  # Reset index after filtering

# Now that the data is cleaned (dropped missing values, handled outliers, and reset index),
# assign it to 'data_cleaned' (after all modifications)
data_cleaned = data_year_in

# Apply a logarithmic transformation to 'Price'
log_price = np.log(data_cleaned['Price'])  # Apply logarithmic transformation
data_cleaned['log_price'] = log_price  # Add the transformed column to the DataFrame

# Data summary after cleaning and transformation
print(data_cleaned.describe(include='all'))

# Visualize relationships between 'Price' and selected features

# Create subplots to visualize relationships
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 3))

# Scatter plot: Price vs Year
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and Year')

# Scatter plot: Price vs EngineV
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('Price and EngineV')

# Scatter plot: Price vs Mileage
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Price and Mileage')

# Show the plots
plt.show()

# Prepare the data for model training
targets = data_cleaned['log_price']  # Set log_price as the target variable
inputs = data_cleaned.drop(['log_price'], axis=1)  # Remove log_price from inputs

# Scale the data
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)

# Now let's implement a regression analysis using statsmodels

# Choose independent variables (features) and dependent variable (target)
X = data_cleaned[['Mileage', 'EngineV', 'Year']]  # Independent variables
y = data_cleaned['Price']  # Dependent variable

# Add a constant to the independent variables matrix (for the intercept in the model)
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print the summary of the regression results
print(model.summary())