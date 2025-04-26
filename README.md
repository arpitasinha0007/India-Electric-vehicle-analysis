# India-Electric-vehicle-analysis
# Vehicle Sales Analysis and Prediction

## Overview
This project focuses on analyzing vehicle sales data to derive valuable insights, predict future sales trends, and categorize manufacturers based on their sales performance. The analysis includes various aspects such as data preprocessing, exploratory data analysis (EDA), predictive modeling, and clustering. The goal is to provide a comprehensive solution for understanding trends, sales forecasts, and market distribution in the vehicle industry.

## Project Structure
- *Data Loading & Cleaning*: Handles loading the dataset, addressing missing values, removing duplicates, and extracting features.
- *Exploratory Data Analysis (EDA)*: Analyzes sales trends over time, provides visualizations to understand market share, and identifies key manufacturers.
- *Sales Prediction*: Implements machine learning models to predict vehicle sales.
- *Clustering Manufacturers*: Categorizes manufacturers into groups using clustering techniques.

## Steps and Features

### 1. Data Loading & Cleaning
- *Path Handling*: Ensured proper file path usage with raw string literals (r"").
- *Data Cleaning*: Addressed missing values using dropna(), removed duplicates using drop_duplicates(), and standardized column naming for clarity.
- *Feature Engineering*: Extracted YEAR from the DATE column to facilitate trend analysis.
- *Data Reshaping*: Applied the melt() function to reshape sales data for more manageable analysis.

### 2. Exploratory Data Analysis (EDA)
- *Trend Analysis*: Analyzed trends for different vehicle categories over the years.
- *Market Share Visualization*: Used a pie chart to visualize the market share distribution in the latest year.
- *Top Manufacturers*: Generated a bar plot to identify the top 5 manufacturers based on sales volume.
- *Sales Trend Visualization*: Created a line plot to visualize the sales trend of the leading manufacturer.

### 3. Sales Prediction
Four machine learning models were implemented for sales prediction:
- *Linear Regression*
- *Ridge Regression*
- *Random Forest*
- *ARIMA* (with try-except for robustness)
- The models' performance is evaluated using metrics such as r2_score and RMSE. A line plot comparing predictions was also created.

### 4. Clustering Manufacturers
- *Pivot Table*: Constructed a pivot table for sales data analysis.
- *Clustering*: Focused on filtering data for 2W and 3W vehicles and used the Elbow method to determine the optimal number of clusters (k) for KMeans.
- *Data Scaling*: Applied StandardScaler to scale the features for clustering.

## Technologies Used
- *Python*: Primary programming language for data manipulation and modeling.
- *Pandas*: Used for data cleaning and manipulation.
- *Matplotlib & Seaborn*: Used for data visualization and EDA.
- *Scikit-Learn*: Used for implementing machine learning models and data scaling.
- *Statsmodels*: Used for implementing ARIMA for time-series forecasting.

## Requirements
To run this project, ensure you have the following Python libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

