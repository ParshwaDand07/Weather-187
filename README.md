# Weather Prediction Using Various Regression Models

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

This project utilizes machine learning techniques to predict the average temperature in India using multiple regression models, including Linear Regression, Ridge Regression, Support Vector Regression (SVR), Polynomial Regression, Lasso Regression, Stochastic Gradient Descent (SGD), and ElasticNet. The dataset contains weather data collected from different weather stations across India, including features such as precipitation, temperature, date, and station information.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

To run this project locally, follow these installation steps:

```bash
# Clone the repository
git clone https://github.com/ParshwaDand07/Weather-187

# Navigate to the project directory
cd weather-prediction

# Install dependencies
pip install -r pandas
pip install -r numpy
pip install -r sklearn
pip install -r seaborn
pip install -r matplotlib
pip install -r calendar

# Import necessary modules
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import seaborn as sns
from sklearn import preprocessing
import calendar
import warnings 
import matplotlib.pyplot as plt

# Load the weather dataset
weather_data = pd.read_csv('the weather of 187 countries in 2020.csv')

# Preprocess the data 

#The country that I have selected is "INDIA" for my predictions. There are so many unnecessary columns with so many null values, so I am dropping all such columns.
weather_india = weather_data[weather_data['Country/Region']=='India'].copy()

#We are calculating how much value is in each column in this dataset in percentage.
(weather_india.isnull().sum())/weather_data.shape[0]

#we drop unnecessary features
weather_india_final = weather_india.drop(['ELEVATION', 'PRCP_ATTRIBUTES','TAVG_ATTRIBUTES','TMAX_ATTRIBUTES','TMIN_ATTRIBUTES','DAPR','MDP')

#By using Label Encoding I am just giving a unique number to each unique station.
label_encoder = preprocessing.LabelEncoder()
weather_india_final['STATION']= label_encoder.fit_transform(weather_india_final['STATION'])

#Making 'DATE' column as our index
weather_india_final.set_index('DATE',inplace = True)

#By using Foward fill method I have filled all null values in 'PRCP' column. Still there are some rows with some null values so I have filled that remaining rows with 0.
weather_india_final['PRCP'].fillna(method = 'ffill',inplace=True)
weather_india_final['PRCP'].fillna(0,inplace=True)

#Filling null values of 'TMIN' by taking median of all values in that column.
weather_india_final['TMIN'].fillna(weather_india_final['TMIN'].median(),inplace=True)

# Train different regression models
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=1.0)
elasticnet_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
svr_model = SVR(kernel='linear')
poly_model = PolynomialFeatures(degree=2)
sgd_model = SGDRegressor()

# Train, evaluate and Visualize the results each model

#we have done same for all model
params = {'alpha': [0.0006,0.0009,0.0003,0.001,0.0015,0.002,0.1,0.3,0.6,0.005,0.003,0.09,0.06,0.006,1,5,8,10,20],
          'solver' : ['auto', 'svd', 'cholesky', 'saga','lsqr', 'sparse_cg', 'sag']
         }


Ridge_metrics = pd.DataFrame(columns=['Station', 'MAE', 'MSE','Accuracy'])
stations = sorted(weather_india_final['STATION'].unique())
fig, ax = plt.subplots(1, 6, figsize=(22, 3))
fig.suptitle('Ridge Regression')
for i, station in enumerate(stations):
    df = weather_india_final[weather_india_final['STATION'] == station].copy() 
    df.drop(['STATION'], axis=1, inplace=True)
    
    train = df.loc[:'2020-06-30']
    X_train = train.drop('Target', axis=1)
    y_train = train['Target']
    

    test = df.loc['2020-07-01':]
    X_test = test.drop(['Target'], axis=1)
    y_test = test['Target']
    
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    grid_search = GridSearchCV(estimator=Ridge(), param_grid=params_ridge, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train_scaled, y_train)
    
    preds = grid_search.predict(X_test_scaled)

    metrics = {'Station' : i, 'MAE': mean_absolute_error(y_test, preds), 'MSE' : mean_squared_error(y_test, preds), 'Accuracy' : str(round(100 - (np.mean(np.abs(y_test - preds) / y_test) * 100),2))+'%'}
    Ridge_metrics = pd.concat([Ridge_metrics, pd.DataFrame([metrics])], ignore_index=True)
    #print('Station:', station, 'MAE:', mean_absolute_error(y_test, preds),'MSE:',mean_squared_error(y_test, preds),r2_score(y_test, preds))
    
    combined = pd.concat([y_test, pd.Series(preds, index=test.index)], axis=1)
    combined.columns = ['Actual', 'Predictions']
    print(f'Best Params for Station {i} : {grid_search.best_params_}')

    count = i % 6
    ax[count].plot(combined.index, combined['Actual'], label='Actual')
    ax[count].plot(combined.index, combined['Predictions'], label='Predicted')
    ax[count].set_title(f'Station {station}\n')
    ax[count].tick_params(axis='x', rotation=45)
    
    if (count == 5) or (i == (len(stations) - 1)):
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.05))
        plt.tight_layout()
        plt.show()
        
        if i != (len(stations) - 1):
            fig, ax = plt.subplots(1, 6, figsize=(22, 3))

#Contributing
Fork the project.
Create a new branch (git checkout -b feature).
Make changes and commit them (git commit -m 'Add feature').
Push to the branch (git push origin feature).
Open a pull request.
Contributions are welcome!

#License
This project is licensed under the MIT License. See LICENSE.md for details.

