#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This script downloads a dataset from a given URL and saves it as a CSV file

import requests

# URL of the CSV file
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

# Download the file
response = requests.get(url)

# Save the downloaded file as 'auto.csv'
with open('auto.csv', 'wb') as file:
    file.write(response.content)


# In[2]:


# This script loads and displays the dataset

import pandas as pd
import numpy as np

# Load the dataset from the CSV file
df = pd.read_csv('auto.csv', header=None)
# Display the first few rows of the dataset
df.head()


# In[3]:


# Display the last few rows of the dataset
df.tail()


# In[4]:


# This script adds descriptive headers to the dataset that initially lacks headers.
# Create a list of headers based on the information provided
headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", 
           "num-of-doors", "body-style", "drive-wheels", "engine-location", 
           "wheel-base", "length", "width", "height", "curb-weight", 
           "engine-type", "num-of-cylinders", "engine-size", "fuel-system", 
           "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", 
           "city-mpg", "highway-mpg", "price"]

# Replace the existing headers (integers) with the descriptive headers we created
df.columns = headers
df.head()


# In[5]:


# Replace "?" with NaN to handle missing values
df.replace("?", np.nan, inplace=True)
df.head()


# In[6]:


# Save the DataFrame to a CSV file
df.to_csv('automobile.csv', index=False)


# In[7]:


# Display the data types of each column in the DataFrame
df.dtypes


# In[8]:


# Display summary statistics for the numerical columns in the DataFrame
df.describe()


# In[9]:


# Display summary statistics for all columns, including non-numerical ones
df.describe(include="all")


# In[10]:


# Display a concise summary of the DataFrame, including data types and non-null counts
df.info()


# In[11]:


# Create a DataFrame indicating the presence of missing values
# Each cell will be True if the value is missing, and False otherwise
missing_data = df.isnull()

# Display the first few rows of the DataFrame showing missing values
missing_data.head()


# In[12]:


# Iterate over each column in the DataFrame that tracks missing values
for column in missing_data.columns.values.tolist():
    # Print the name of the current column
    print(column)
    
    # Print the count of True (missing) and False (not missing) values for the current column
    print(missing_data[column].value_counts())
    
    # Print a blank line for better readability between columns
    print("")


# In[13]:


# Convert the "normalized-losses" column to float type and calculate the mean value
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
# Print the average of the "normalized-losses" column
print("Average of normalized-losses:", avg_norm_loss)

# Convert the "bore" column to float type and calculate the mean value
avg_bore = df['bore'].astype('float').mean(axis=0)
# Print the average of the "bore" column
print("Average of bore:", avg_bore)

# Convert the "stroke" column to float type and calculate the mean value
avg_stroke = df["stroke"].astype("float").mean(axis=0)
# Print the average of the "stroke" column
print("Average of stroke:", avg_stroke)

# Convert the "horsepower" column to float type and calculate the mean value
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
# Print the average of the "horsepower" column
print("Average horsepower:", avg_horsepower)

# Convert the "peak-rpm" column to float type and calculate the mean value
avg_peakrpm = df['peak-rpm'].astype('float').mean(axis=0)
# Print the average of the "peak-rpm" column
print("Average peak rpm:", avg_peakrpm)


# In[14]:


# Replace missing values in the "normalized-losses" column with the calculated average
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

# Replace missing values in the "bore" column with the calculated average
df["bore"].replace(np.nan, avg_bore, inplace=True)

# Replace missing values in the "stroke" column with the calculated average
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

# Replace missing values in the "horsepower" column with the calculated average
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

# Replace missing values in the "peak-rpm" column with the calculated average
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)


# In[15]:


# Count the occurrences of each value in the "num-of-doors" column
print(df['num-of-doors'].value_counts())

# Identify the most frequent value in the "num-of-doors" column
print(df['num-of-doors'].value_counts().idxmax())

# Replace missing values in the "num-of-doors" column with the most frequent value ("four")
df["num-of-doors"].replace(np.nan, "four", inplace=True)


# In[16]:


# Drop any row in the DataFrame where the "price" column has a missing value (NaN)
df.dropna(subset=["price"], axis=0, inplace=True)

# Reset the index of the DataFrame after dropping rows, removing the old index
df.reset_index(drop=True, inplace=True)


# In[17]:


df.head()


# In[18]:


# Display a concise summary of the DataFrame, including data types and non-null counts
# Initially, there were 205 entries, but after dropping 4 rows with missing 'price' values, there are now 201 entries.
# The following code shows that each column has 201 non-null entries.
df.info()


# In[19]:


# Some columns have incorrect data types. 
# Numerical columns like 'bore' and 'stroke' should be 'float' or 'int' but are currently 'object'.
# We'll convert these to the correct types using the 'astype()' method.

# Convert 'bore' and 'stroke' columns to float type
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")

# Convert 'normalized-losses' column to int type
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")

# Convert 'price' column to float type
df[["price"]] = df[["price"]].astype("float")

# Convert 'peak-rpm' column to float type
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

# Verify the changes by displaying the data types of each column
df.dtypes


# In[20]:


# Convert 'city-mpg' to L/100km using the formula: L/100km = 235 / mpg
df['city-L/100km'] = 235 / df["city-mpg"]

# Convert 'highway-mpg' to L/100km using the formula: L/100km = 235 / mpg
df["highway-L/100km"] = 235 / df["highway-mpg"]

# Display the first few rows of the DataFrame to verify the changes
df.head()


# In[21]:


# Normalize 'length', 'width', and 'height' columns by scaling values to the range 0 to 1
df['length'] = df['length'] / df['length'].max()
df['width'] = df['width'] / df['width'].max()
df['height'] = df['height'] / df['height'].max()

# Display the first few rows of the normalized columns
df[["length", "width", "height"]].head()


# In[22]:


# Convert the 'horsepower' column to integer type to prepare for binning
df["horsepower"] = df["horsepower"].astype(int, copy=True)


# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[24]:


import numpy as np

# Define bin edges
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
print(bins)

# Define bin labels
group_names = ['Low', 'Medium', 'High']

# Create a new column 'horsepower-binned' with bin labels
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
print(df[['horsepower','horsepower-binned']].head())
df["horsepower-binned"].value_counts()


# In[25]:


# Create a bar chart of the horsepower bins
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[26]:


# Using 3 bins to categorize the 'horsepower' values into three distinct ranges
plt.pyplot.hist(df["horsepower"], bins=3, edgecolor='black')

# Set the label for the x-axis to describe the variable plotted
plt.pyplot.xlabel("Horsepower")

# Set the label for the y-axis to show the count of occurrences for each bin
plt.pyplot.ylabel("Count")

# Set the title of the plot to describe what is being visualized
plt.pyplot.title("Horsepower Bins")


# In[27]:


# Create dummy variables for the "fuel-type" column
dummy_variable_1 = pd.get_dummies(df["fuel-type"])

# Rename the columns to more descriptive names
dummy_variable_1.rename(columns={'gas': 'fuel-type-gas', 'diesel': 'fuel-type-diesel'}, inplace=True)

# Display the updated dummy variables for "fuel-type"
print(dummy_variable_1.head())

# Create dummy variables for the "aspiration" column
dummy_variable_2 = pd.get_dummies(df["aspiration"])

# Rename the columns to more descriptive names
dummy_variable_2.rename(columns={'std': 'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)
# Display the updated dummy variables for "aspiration"
print("")
print(dummy_variable_2.head())


# In[28]:


# Merge the new dummy variable dataframes with the original dataframe
df = pd.concat([df, dummy_variable_1, dummy_variable_2], axis=1)

# Display the updated dataframe with new dummy variables
df.head()


# In[29]:


# Save the cleaned and updated DataFrame to a CSV file
df.to_csv('clean_df.csv', index=False)


# In[30]:


df.head()


# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt  # Import matplotlib.pyplot for plotting functions

# Create a regression plot to visualize the relationship between "engine-size" and "price"
sns.regplot(x="engine-size", y="price", data=df)

# Set the lower limit of the y-axis to 0 for better visibility
plt.ylim(0,)

# Display the plot
plt.show()


# In[32]:


# Calculate and display the correlation matrix between "engine-size" and "price"
df[["engine-size", "price"]].corr()


# In[33]:


# Plot regression line and scatter plot for "highway-mpg" vs. "price" 
sns.regplot(x="highway-mpg", y="price", data=df)


# In[34]:


# Compute and display the correlation matrix between "highway-mpg" and "price"
df[['highway-mpg', 'price']].corr()


# In[35]:


# Create a regression plot to visualize the relationship between "peak-rpm" and "price"
sns.regplot(x="peak-rpm", y="price", data=df)

# Set the lower limit of the y-axis to 0 for better visibility
plt.ylim(0,)

# Display the plot
plt.show()


# In[36]:


# Calculate and display the correlation matrix between "peak-rpm" and "price"
df[['peak-rpm', 'price']].corr()


# In[37]:


# Create a regression plot to visualize the relationship between "stroke" and "price"
sns.regplot(x="stroke", y="price", data=df)


# In[38]:


# Calculate and display the correlation matrix between "stroke" and "price"
df[["stroke", "price"]].corr()


# In[39]:


# Create a boxplot to visualize the distribution of "price" across different "body-style" categories
sns.boxplot(x="body-style", y="price", data=df)


# In[40]:


# Create a boxplot to visualize the distribution of "price" across different "engine-location" categories
sns.boxplot(x="engine-location", y="price", data=df)


# In[41]:


# Create a boxplot to visualize the distribution of "price" across different "drive-wheels" categories
sns.boxplot(x="drive-wheels", y="price", data=df)


# In[42]:


# Generate descriptive statistics for the dataset after data wrangling
df.describe()


# In[43]:


# Generate descriptive statistics for categorical variables in the dataset
df.describe(include=['object'])


# In[44]:


# Count the frequency of each unique value in the 'drive-wheels' column and convert the result to a DataFrame
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()

# Rename the column 'drive-wheels' to 'value_counts' for clarity
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)

# Set the index name to 'drive-wheels' for better readability in the output
drive_wheels_counts.index.name = 'drive-wheels'

# Display the DataFrame with the count of each 'drive-wheels' category
drive_wheels_counts


# In[45]:


# Count the frequency of each unique value in the 'engine-location' column and convert the result to a DataFrame
engine_loc_counts = df['engine-location'].value_counts().to_frame()

# Rename the column 'engine-location' to 'value_counts' for clarity
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)

# Set the index name to 'engine-location' for better readability in the output
engine_loc_counts.index.name = 'engine-location'

# Display the DataFrame with the count of each 'engine-location' category
engine_loc_counts.head()


# In[46]:


# Select relevant columns for analysis
df_group_one = df[['drive-wheels', 'price']]

# Group by 'drive-wheels' and calculate the mean 'price' for each group
df_group_one = df_group_one.groupby(['drive-wheels'], as_index=False).mean()

# Display the resulting DataFrame
df_group_one


# In[47]:


# Select relevant columns for analysis: 'body-style' and 'price'
df_gptest2 = df[['body-style', 'price']]

# Group the data by 'body-style' and calculate the mean 'price' for each 'body-style'
# The parameter 'as_index=False' ensures that 'body-style' remains a column in the result DataFrame
grouped_test_bodystyle = df_gptest2.groupby(['body-style'], as_index=False).mean()

# Display the resulting DataFrame showing the mean price for each body style
grouped_test_bodystyle


# In[48]:


# Select relevant columns for analysis: 'drive-wheels', 'body-style', and 'price'
df_gptest = df[['drive-wheels', 'body-style', 'price']]

# Group the data by 'drive-wheels' and 'body-style', and calculate the mean 'price' for each combination
# The parameter 'as_index=False' ensures that 'drive-wheels' and 'body-style' remain columns in the result DataFrame
grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'], as_index=False).mean()

# Display the resulting DataFrame showing the mean price for each combination of 'drive-wheels' and 'body-style'
grouped_test1


# In[49]:


# Pivot the grouped DataFrame to reformat it, with 'drive-wheels' as rows and 'body-style' as columns
# This transformation makes it easier to analyze and compare mean prices across different body styles and drive wheels
grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style')

# Fill missing values in the pivoted DataFrame with 0
# This step is important to handle any gaps in the data where certain combinations of 'drive-wheels' and 'body-style' do not exist
grouped_pivot = grouped_pivot.fillna(0)

# Display the pivoted DataFrame with missing values filled
grouped_pivot


# In[50]:


# Use the grouped results to create a heatmap-like plot
# 'pcolor' is used to display the values in the pivoted DataFrame as a color-coded grid
plt.pcolor(grouped_pivot, cmap='RdBu')

# Add a color bar to the plot for reference, showing the mapping between color and value
plt.colorbar()

# Display the plot
plt.show()


# In[51]:


# Create a figure and axis object for the plot
fig, ax = plt.subplots()

# Generate a pseudocolor plot (heatmap) using the grouped pivot data
im = ax.pcolor(grouped_pivot, cmap='RdBu')

# Define the labels for the columns and rows
row_labels = grouped_pivot.columns.levels[1]  # Get the body-style labels (columns)
col_labels = grouped_pivot.index  # Get the drive-wheels labels (rows)

# Position the ticks and labels in the center of each cell
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)  # X-axis (body-style)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)  # Y-axis (drive-wheels)

# Set the labels for the X and Y ticks
ax.set_xticklabels(row_labels, minor=False)  # Apply body-style labels to X-axis
ax.set_yticklabels(col_labels, minor=False)  # Apply drive-wheels labels to Y-axis

# Rotate the X-axis labels if they are too long for better readability
plt.xticks(rotation=90)

# Add a color bar to provide a reference for the color mapping
fig.colorbar(im)

# Display the plot
plt.show()


# In[52]:


from scipy import stats

# Calculate Pearson correlation coefficient and P-value for 'wheel-base' and 'price'
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient for Wheel-Base vs. Price is", pearson_coef, " with a P-value of P =", p_value)

# Calculate Pearson correlation coefficient and P-value for 'horsepower' and 'price'
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient for Horsepower vs. Price is", pearson_coef, " with a P-value of P =", p_value)

# Calculate Pearson correlation coefficient and P-value for 'length' and 'price'
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient for Length vs. Price is", pearson_coef, " with a P-value of P = ", p_value)

# Calculate Pearson correlation coefficient and P-value for 'width' and 'price'
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient for Width vs. Price is", pearson_coef, " with a P-value of P =", p_value)

# Calculate Pearson correlation coefficient and P-value for 'curb-weight' and 'price'
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print("The Pearson Correlation Coefficient for Curb-Weight vs. Price is", pearson_coef, " with a P-value of P =", p_value)

# Calculate Pearson correlation coefficient and P-value for 'engine-size' and 'price'
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient for Engine-Size vs. Price is", pearson_coef, " with a P-value of P =", p_value)

# Calculate Pearson correlation coefficient and P-value for 'bore' and 'price'
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient for Bore vs. Price is", pearson_coef, " with a P-value of P =  ", p_value)

# Calculate Pearson correlation coefficient and P-value for 'city-mpg' and 'price'
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient for City-mpg vs. Price is", pearson_coef, " with a P-value of P = ", p_value)

# Calculate Pearson correlation coefficient and P-value for 'highway-mpg' and 'price'
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print("The Pearson Correlation Coefficient for Highway-mpg vs. Price is", pearson_coef, " with a P-value of P = ", p_value)


# In[53]:


# Group the dataset by 'drive-wheels' and select the 'price' column for analysis
grouped_test2 = df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])

# Display the first few rows of the grouped data for verification
grouped_test2.head()

# Access the 'price' values for the '4wd' drive-wheels group
grouped_test2.get_group('4wd')['price']

# Perform ANOVA to compare the means of 'price' across different 'drive-wheels' groups
# We use the 'f_oneway' function from scipy.stats to calculate the F-test score and p-value
f_val, p_val = stats.f_oneway(
    grouped_test2.get_group('fwd')['price'], 
    grouped_test2.get_group('rwd')['price'], 
    grouped_test2.get_group('4wd')['price']
)  

# Print the results of the ANOVA test
print("ANOVA results for 'drive-wheels' and 'price': F=", f_val, ", P =", p_val)

# Group the dataset by 'drive-wheels' and select the 'price' column for analysis
grouped_test2 = df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])

# Display the first few rows of the grouped data for verification
grouped_test2.head()

# Access the 'price' values for the '4wd' drive-wheels group
grouped_test2.get_group('4wd')['price']

# Perform ANOVA to compare the means of 'price' across 'fwd' and 'rwd' drive-wheels groups
# We use the 'f_oneway' function from scipy.stats to calculate the F-test score and p-value
f_val, p_val = stats.f_oneway(
    grouped_test2.get_group('fwd')['price'], 
    grouped_test2.get_group('rwd')['price']
)  

# Print the results of the ANOVA test comparing 'fwd' and 'rwd' drive-wheels groups
print("ANOVA results for 'fwd' vs 'rwd' drive-wheels and 'price': F =", f_val, ", P =", p_val)  
# F-test score: Indicates the ratio of variance between 'fwd' and 'rwd' groups to variance within these groups.
# P-value: Shows the probability that the observed differences between 'fwd' and 'rwd' groups are due to chance.

# Perform ANOVA to compare the means of 'price' across '4wd' and 'rwd' drive-wheels groups
f_val, p_val = stats.f_oneway(
    grouped_test2.get_group('4wd')['price'], 
    grouped_test2.get_group('rwd')['price']
)  

# Print the results of the ANOVA test comparing '4wd' and 'rwd' drive-wheels groups
print("ANOVA results for '4wd' vs 'rwd' drive-wheels and 'price': F =", f_val, ", P =", p_val)  
# F-test score: Indicates the ratio of variance between '4wd' and 'rwd' groups to variance within these groups.
# P-value: Shows the probability that the observed differences between '4wd' and 'rwd' groups are due to chance.

# Perform ANOVA to compare the means of 'price' across '4wd' and 'fwd' drive-wheels groups
f_val, p_val = stats.f_oneway(
    grouped_test2.get_group('4wd')['price'], 
    grouped_test2.get_group('fwd')['price']
)  

# Print the results of the ANOVA test comparing '4wd' and 'fwd' drive-wheels groups
print("ANOVA results for '4wd' vs 'fwd' drive-wheels and 'price': F =", f_val, ", P =", p_val)  
# F-test score: Indicates the ratio of variance between '4wd' and 'fwd' groups to variance within these groups.
# P-value: Shows the probability that the observed differences between '4wd' and 'fwd' groups are due to chance.


# In[54]:


# Import the LinearRegression class from sklearn
from sklearn.linear_model import LinearRegression

# Create an instance of the LinearRegression model
lm = LinearRegression()

# Display the created instance
lm


# In[55]:


# Select 'highway-mpg' as the predictor variable (independent variable) and assign it to X
X = df[['highway-mpg']]

# Select 'price' as the response variable (dependent variable) and assign it to Y
Y = df['price']

# Fit the linear regression model using the selected predictor (X) and response (Y)
lm.fit(X, Y)

# Use the fitted model to predict the price (Yhat) based on the values in X
Yhat = lm.predict(X)

# Display the first 5 predicted values
print("The first predicted values are:", Yhat[0:5])

# Get the intercept (a) of the linear regression line
intercept = lm.intercept_
print("The intercept is: a =", intercept)

# Get the coefficient (b) of the linear regression line, indicating the slope
coef = lm.coef_[0]
print("The coefficient is: b =", coef)

# Display the linear equation based on the calculated intercept and coefficient
print(f'The linear equation is: Y = {intercept:.2f} + {coef:.2f}X')


# In[56]:


# Define the predictor variables (independent variables) for the Multiple Linear Regression model
# We are using 'highway-mpg', 'engine-size', 'horsepower', and 'curb-weight' as predictors
Z = df[['highway-mpg', 'engine-size', 'horsepower', 'curb-weight']]

# Fit the Multiple Linear Regression model using the predictor variables (Z) and the response variable ('price')
lm.fit(Z, df['price'])

# Predict the values of 'price' using the fitted model
Yhat = lm.predict(Z)

# Display the first 5 predicted values
print("The first predicted values are: ", Yhat[:5])

# Display the coefficients for each predictor variable
print("The coefficients for each predictor are: ", lm.coef_)

# Display the intercept of the Multiple Linear Regression model
print("The intercept is: a =", lm.intercept_)

# Display the linear equation of the Multiple Linear Regression model
# Note: The coefficients and intercept will be used to construct the equation
# For example, the equation format is: Y = a + b1*X1 + b2*X2 + ... + bn*Xn
coefficients = lm.coef_
intercept = lm.intercept_
equation = f"Y = {intercept:.2f} + " + " + ".join([f"{coef:.2f}*{feature}" for coef, feature in zip(coefficients, Z.columns)])
print("The linear equation is: ", equation)


# In[57]:


# Generate a regression plot with 'highway-mpg' on the x-axis and 'price' on the y-axis using the seaborn library
sns.regplot(x="highway-mpg", y="price", data=df)

# Set the lower limit for the y-axis to 0, ensuring the plot starts from 0 on the y-axis
plt.ylim(0,)


# In[58]:


# Create a regression plot using Seaborn to visualize the relationship between 'peak-rpm' and 'price'
sns.regplot(x="peak-rpm", y="price", data=df)

# Set the y-axis limit to start from 0 to ensure the plot starts at the origin
plt.ylim(0,)


# In[59]:


# Calculate the correlation matrix for the selected columns: 'peak-rpm', 'highway-mpg', and 'price'
# This will show the correlation coefficients between each pair of variables in the dataframe.
df[["peak-rpm", "highway-mpg", "price"]].corr()


# In[60]:


# Generate a residual plot to visualize the residuals of the regression model
# Residuals are the differences between observed and predicted values, plotted against 'highway-mpg'
# This plot helps evaluate the model's fit and identify patterns or issues.

sns.residplot(x=df['highway-mpg'], y=df['price'])
plt.show()


# In[61]:


# Predicts the car prices using the linear model and stores the results in Y_hat.
Y_hat = lm.predict(Z)

# Creates a kernel density estimate plot comparing the actual car prices (in red) with the fitted values predicted by the model (in blue).
ax1 = sns.kdeplot(df['price'], color="r", label="Actual Value", fill=True)
sns.kdeplot(Y_hat, color="b", label="Fitted Values", ax=ax1, fill=True)

# Sets the title and labels for the plot, displaying the comparison of actual vs fitted prices.
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

# Displays the plot.
plt.show()

# Closes the plot to free up resources.
plt.close()


# In[62]:


def PlotPolly(model, independent_variable, dependent_variable, Name):
    # Generate a range of new values for the independent variable to evaluate the polynomial model.
    x_new = np.linspace(15, 55, 100)
    # Compute the corresponding predicted values using the polynomial model.
    y_new = model(x_new)

    # Plot the original data points as dots.
    plt.plot(independent_variable, dependent_variable, '.', label='Actual Data')
    # Plot the polynomial fit as a line.
    plt.plot(x_new, y_new, '-', label='Polynomial Fit')
    # Set the title of the plot.
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    # Get the current axes of the plot.
    ax = plt.gca()
    # Set the background color of the plot.
    ax.set_facecolor((0.898, 0.898, 0.898))
    # Get the current figure.
    fig = plt.gcf()
    # Label the x-axis with the name of the independent variable.
    plt.xlabel(Name)
    # Label the y-axis with 'Price of Cars'.
    plt.ylabel('Price of Cars')
    # Display the plot.
    plt.show()
    # Close the plot to free up resources.
    plt.close()


# In[63]:


# Extract the 'highway-mpg' and 'price' columns from the DataFrame into separate variables.
x = df['highway-mpg']  # Independent variable: highway-mpg
y = df['price']        # Dependent variable: price

# Fit a 3rd-order polynomial (cubic) to the data.
f = np.polyfit(x, y, 3)  # np.polyfit fits a polynomial of the specified order (3rd order) to the data.

# Create a polynomial function from the coefficients obtained.
p = np.poly1d(f)  # np.poly1d generates a polynomial function from the coefficients.

# Extract coefficients from the polynomial function
b3, b2, b1, a = f

# Print the polynomial equation with labeled coefficients.
print(f"Polynomial equation: ")
print("")
print(f"Price = {a:.2f} + {b1:.2f} * (highway-mpg) + {b2:.2f} * (highway-mpg)^2 + {b3:.2f} * (highway-mpg)^3  ")


# In[64]:


PlotPolly(p, x, y, 'highway-mpg')


# In[65]:


# Import the mean_squared_error function from sklearn.metrics to calculate the Mean Squared Error (MSE).
from sklearn.metrics import mean_squared_error

# Fit the linear regression model to the data (X as independent variable and Y as dependent variable).
lm.fit(X, Y)

# Calculate and print the R-squared value of the model, which indicates the proportion of variance explained by the model.
print('The R-square is: ', lm.score(X, Y))

# Predict the dependent variable values (Yhat) using the fitted model.
Yhat = lm.predict(X)

# Print the first four predicted values to observe the model's output.
print('The output of the first four predicted value is: ', Yhat[0:4])

# Calculate the Mean Squared Error (MSE) between the actual values (df['price']) and the predicted values (Yhat).
mse = mean_squared_error(df['price'], Yhat)

# Print the Mean Squared Error, which provides an indication of the average squared difference between actual and predicted values.
print('The mean square error of price and predicted value is: ', mse)


# In[66]:


# Fit the model for Multiple Linear Regression
lm.fit(Z, df['price'])  
# The model is being fitted using Multiple Linear Regression, where Z represents the independent variables and df['price'] is the dependent variable. This trains the regression model to understand the relationship between multiple predictors and car prices.

# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))  
# The R-squared value of the Multiple Linear Regression model is computed and printed. It measures how well the model explains the variability in car prices based on the independent variables in Z. A higher R-squared indicates a better fit.

# Predict the prices using the fitted Multiple Linear Regression model
Y_predict_multifit = lm.predict(Z)  
# The fitted Multiple Linear Regression model is used to predict car prices based on the independent variables in Z. The predicted values are stored in Y_predict_multifit.

# Compute and print the Mean Squared Error (MSE) between the actual prices and the predicted values
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))


# In[67]:


from sklearn.metrics import r2_score, mean_squared_error

# Calculate the R-squared value for the polynomial fit
r_squared = r2_score(df['price'], p(x))  

# Print the R-squared value, which provides an indication of the goodness of fit for the polynomial model.
print('The R-square value is: ', r_squared)


# Calculate the Mean Squared Error (MSE) for the polynomial fit
mse = mean_squared_error(df['price'], p(x))  

print('The Mean Squared Error is: ', mse)  


# In[68]:


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    
    # Plot the KDE for the red function with its label
    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    
    # Plot the KDE for the blue function with its label
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)
    
    # Add title and labels to the plot
    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    
    # Add a legend to the plot
    plt.legend()
    
    # Display the plot
    plt.show()
    plt.close()


# In[69]:


def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):

    # xtrain, xtest: Training and testing data for the independent variable
    # y_train, y_test: Training and testing data for the dependent variable
    # lr: Linear regression object that has been trained
    # poly_transform: Polynomial transformation object used for transforming the input data

    # Determine the range of x values by finding the minimum and maximum of both training and testing sets
    xmax = max([xtrain.values.max(), xtest.values.max()])
    xmin = min([xtrain.values.min(), xtest.values.min()])

    # Create a range of values from xmin to xmax with a step size of 0.1
    x = np.arange(xmin, xmax, 0.1)

    # Plot the training data as red dots
    plt.plot(xtrain, y_train, 'ro', label='Training Data')

    # Plot the testing data as green dots
    plt.plot(xtest, y_test, 'go', label='Test Data')

    # Plot the predicted polynomial function line using the polynomial transformation on the x values
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')

    # Set the limits for the y-axis to range from -10000 to 60000
    plt.ylim([-10000, 60000])

    # Set the label for the y-axis
    plt.ylabel('Price')

    # Display the legend to identify the different data series
    plt.legend()


# In[70]:


# Extract only the numeric columns from the DataFrame and assign them to a new DataFrame 'df1'
df1 = df._get_numeric_data()

# Display the first few rows of the 'df1' DataFrame to inspect the numeric data
df1.head()


# In[71]:


from sklearn.model_selection import train_test_split

# The target variable is 'price' from the dataset.
y_data = df1['price']

# All other numeric columns except 'price' are used as input features.
x_data = df1.drop('price', axis=1)

# Split the dataset into training and testing sets.
# 10% of the data is used for testing, and the random_state is set for reproducibility.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

# Print the number of samples in the test set.
print("Number of test samples:", x_test.shape[0])

# Print the number of samples in the training set.
print("Number of training samples:", x_train.shape[0])

# Create a LinearRegression object.
lre = LinearRegression()

# Fit the model using 'horsepower' as the predictor and the training data.
lre.fit(x_train[['horsepower']], y_train)

# Calculate and print the R-squared value for the test set.
r_squared_test = lre.score(x_test[['horsepower']], y_test)
print("R-squared value for the test set with 10% split:", r_squared_test)

# Calculate and print the R-squared value for the training set.
r_squared_train = lre.score(x_train[['horsepower']], y_train)
print("R-squared value for the training set with 10% split:", r_squared_train)


# In[72]:


# Split the data into training and test sets with 40% of the data reserved for testing.
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.4, random_state=0)

# Print the number of samples in the training and test sets.
print("Number of test samples:", x_test1.shape[0])
print("Number of training samples:", x_train1.shape[0])

# Create a LinearRegression object.
lre = LinearRegression()

# Fit the model using 'horsepower' as the predictor with the new training data.
lre.fit(x_train1[['horsepower']], y_train1)

# Evaluate the model by calculating the R-squared value for the test set.
test_r_squared = lre.score(x_test1[['horsepower']], y_test1)
print("R-squared value for the test set with 40% split:", test_r_squared)

# Evaluate the model by calculating the R-squared value for the training set.
train_r_squared = lre.score(x_train1[['horsepower']], y_train1)
print("R-squared value for the training set with 40% split:", train_r_squared)


# In[73]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# Perform cross-validation to evaluate the model using the 'horsepower' feature.
# This calculates the R-squared scores for each fold in the cross-validation.
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

# Calculate the negative Mean Squared Error (MSE) for each fold in the cross-validation.
# The negative sign is used because cross_val_score treats higher scores as better,
# and MSE should be minimized.
neg_mse = -1 * cross_val_score(lre, x_data[['horsepower']], y_data, cv=4, scoring='neg_mean_squared_error')
print("The mean of the negative MSEs is", neg_mse.mean(), "and the standard deviation is", neg_mse.std())

# Perform cross-validation to compute R-squared scores using 2-fold cross-validation.
Rc = cross_val_score(lre, x_data[['horsepower']], y_data, cv=2)
print("The mean R-squared score with 2-fold cross-validation is", Rc.mean())

# Predict the target values using cross-validation and return the predictions.
# This helps to evaluate how well the model performs across different data folds.
yhat = cross_val_predict(lre, x_data[['horsepower']], y_data, cv=4)
print("The first five predicted values are:", yhat[0:5])


# In[74]:


# Create a LinearRegression object.
lr = LinearRegression()

# Fit the model using 'horsepower', 'curb-weight', 'engine-size', and 'highway-mpg' as predictors.
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

# Predict the target values using the training data.
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

# Print the first five predicted values for the training set.
print("First five predicted values for the training set:", yhat_train[0:5])

# Predict the target values using the test data.
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

# Print the first five predicted values for the test set.
print("First five predicted values for the test set:", yhat_test[0:5])


# In[75]:


# Set the title for the distribution plot.
Title = 'Distribution Plot of Predicted Value Using Training Data vs Training Data Distribution'

# Generate the distribution plot.
# 'y_train' holds the actual training values.
# 'yhat_train' holds the predicted values from the training data.
# The labels "Actual Values (Train)" and "Predicted Values (Train)" will appear in the plot legend.
# 'Title' sets the title of the plot.
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)


# In[76]:


# Set the title for the distribution plot.
Title = 'Distribution Plot of Predicted Value Using Test Data vs Data Distribution of Test Data'

# Generate the distribution plot.
# 'y_test' contains the actual test values.
# 'yhat_test' contains the predicted values from the test data.
# The labels "Actual Values (Test)" and "Predicted Values (Test)" will appear in the plot legend.
# 'Title' sets the title of the plot.
DistributionPlot(y_test, yhat_test, "Actual Values (Test)", "Predicted Values (Test)", Title)


# In[77]:


from sklearn.preprocessing import PolynomialFeatures

# Split the data into training and test sets, with 55% of the data used for training and 45% for testing.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)

# Create a polynomial feature transformer with a degree of 5.
pr = PolynomialFeatures(degree=5)

# Transform the 'horsepower' feature in both training and test sets to polynomial features.
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])

# Initialize and train a linear regression model using the polynomial features.
poly = LinearRegression()
poly.fit(x_train_pr, y_train)

# Predict the target values for the test set using the trained model.
yhat = poly.predict(x_test_pr)

# Print the first five predicted values.
print("Predicted values:", yhat[0:5])

# Print the first four predicted values and compare them with the actual target values.
print("Predicted values (first 4):", yhat[0:4])
print("True values (first 4):", y_test[0:4].values)


# In[78]:


# Visualize the polynomial regression results using the PollyPlot function.
# This function displays the training data, test data, and the predicted polynomial function.
# x_train['horsepower'] and x_test['horsepower']: Input features for training and test sets.
# y_train and y_test: Actual target values for training and test sets.
# poly: The trained polynomial regression model.
# pr: The polynomial feature transformer.

PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly, pr)


# In[79]:


# Calculate R² score for the training data
train_r2 = poly.score(x_train_pr, y_train)
print("R² of the training data:", train_r2)

# Calculate R² score for the test data
test_r2 = poly.score(x_test_pr, y_test)
print("")
print("R² of the test data:", test_r2)


# In[80]:


# Initialize an empty list to store R² scores for different polynomial degrees
Rsqu_test = []

# Define the range of polynomial orders to test
order = [1, 2, 3, 4]

# Loop through each polynomial order
for n in order:
    # Create a PolynomialFeatures object for the current degree
    pr = PolynomialFeatures(degree=n)
    
    # Transform the training and testing data using the polynomial features
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    
    # Create and train the Linear Regression model
    lr = LinearRegression()
    lr.fit(x_train_pr, y_train)
    
    # Calculate the R² score on the test data and append it to the list
    Rsqu_test.append(lr.score(x_test_pr, y_test))

# Plot R² scores against polynomial orders
plt.plot(order, Rsqu_test)

# Label the x-axis as 'order'
plt.xlabel('order')

# Label the y-axis as 'R^2'
plt.ylabel('R^2')

# Set the title of the plot
plt.title('R^2 Using Test Data')

# Annotate the plot with a text label indicating the maximum R² value
plt.text(3.05, 0.735, 'Maximum R^2 ')


# In[81]:


def f(order, test_data):
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    
    # Create a PolynomialFeatures object with the specified degree
    pr = PolynomialFeatures(degree=order)
    
    # Transform the training and testing features using polynomial features
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    
    # Initialize and fit a LinearRegression model using the transformed training features
    poly = LinearRegression()
    poly.fit(x_train_pr, y_train)
    
    # Plot the results using the PollyPlot function, showing training data, testing data, and the model's predictions
    PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly, pr)


# In[82]:


from ipywidgets import interact

# Create interactive widgets to adjust the parameters of the function "f".
interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))


# In[83]:


# Create a PolynomialFeatures object with a degree of 2 for polynomial transformation
pr1 = PolynomialFeatures(degree=2)

# Transform the training features 'horsepower', 'curb-weight', 'engine-size', and 'highway-mpg' into polynomial features
x_train_pr1 = pr1.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

# Transform the testing features 'horsepower', 'curb-weight', 'engine-size', and 'highway-mpg' into polynomial features
x_test_pr1 = pr1.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

# Output the shape of the transformed training data to see the number of features after the polynomial transformation
x_train_pr1.shape

# Fit a LinearRegression model using the transformed polynomial features from the training data
poly1 = LinearRegression().fit(x_train_pr1, y_train)


# In[84]:


yhat_test1=poly1.predict(x_test_pr1)

Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'

DistributionPlot(y_test, yhat_test1, "Actual Values (Test)", "Predicted Values (Test)", Title)


# In[85]:


from sklearn.linear_model import Ridge

# Create a PolynomialFeatures object with a degree of 2 for polynomial transformation
pr = PolynomialFeatures(degree=2)

# Transform the training features into polynomial features
x_train_pr = pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'normalized-losses', 'symboling']])

# Transform the testing features into polynomial features
x_test_pr = pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'normalized-losses', 'symboling']])

# Initialize a Ridge regression model with an alpha value of 1 (controls regularization strength)
RigeModel = Ridge(alpha=1)

# Fit the Ridge regression model using the transformed polynomial features from the training data
RigeModel.fit(x_train_pr, y_train)

# Predict the target values using the trained Ridge model on the test data
yhat = RigeModel.predict(x_test_pr)

# Print the first four predicted values and the corresponding actual test values for comparison
print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)


# In[86]:


from tqdm import tqdm

# Initialize empty lists to store R^2 scores for the test and training sets
Rsqu_test = []
Rsqu_train = []

# Create an empty list 'dummy1' (potentially for later use, though it's not used here)
dummy1 = []

# Generate an array of alpha values, scaled by a factor of 10, ranging from 0 to 9990
Alpha = 10 * np.array(range(0,1000))

# Initialize a progress bar to visualize the loop's progress over the Alpha values
pbar = tqdm(Alpha)

# Iterate over each alpha value in the Alpha array
for alpha in pbar:
    # Initialize a Ridge regression model with the current alpha value
    RigeModel = Ridge(alpha=alpha)
    
    # Fit the Ridge regression model using the training polynomial features
    RigeModel.fit(x_train_pr, y_train)
    
    # Calculate the R^2 scores for the test and training data
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    # Update the progress bar with the current test and training R^2 scores
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    # Append the current R^2 scores to their respective lists
    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)


# In[87]:


# Plot the R^2 scores for the test (validation) data against the alpha values
plt.plot(Alpha, Rsqu_test, label='validation data')

# Plot the R^2 scores for the training data against the alpha values, using a red line
plt.plot(Alpha, Rsqu_train, 'r', label='training Data')

# Label the x-axis with 'alpha' to indicate that it represents the regularization strength
plt.xlabel('alpha')

# Label the y-axis with 'R^2' to indicate that the y-axis represents the R^2 score
plt.ylabel('R^2')

# Display a legend to differentiate between the lines representing the test and training data
plt.legend()


# In[88]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid for alpha values to search
parameters1 = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000]}]

# Initialize a Ridge regression model
RR = Ridge()

# Perform GridSearchCV to find the best alpha using 4-fold cross-validation
Grid1 = GridSearchCV(RR, parameters1, cv=4)

# Fit GridSearchCV with the training data to find the best model
Grid1.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

# Retrieve the best model with the optimal alpha value
BestRR = Grid1.best_estimator_

# Evaluate the best model's performance on the test data
score = BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)

# Print the score and explanation
print(f'The best model, optimized using Grid Search, achieved a R^2 score of {score:.4f} on the test data.')
print('This R^2 score indicates how well the model predicts the car prices on unseen test data.')
print('A higher R^2 score signifies a better fit of the model to the test data, with 1 being a perfect fit.')

