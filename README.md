# Weather Prediction Using Various Regression Models

## Overview

This project utilizes machine learning techniques to predict the average temperature in India using multiple regression models, including Linear Regression, Polynomial Regression, Stochastic Gradient Descent (SGD), Lasso Regression, Ridge Regression, Support Vector Regression (SVR) and ElasticNet. The dataset contains weather data collected from different weather stations across India, including features such as precipitation, temperature, date, and station information.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)

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
```

```
# Import necessary modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGD Regressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import calendar

# Load the weather dataset
weather_data = pd.read_csv('the weather of 187 countries in 2020.csv')

# Preprocess the data 

#The country that we have selected is "INDIA" for my predictions. There are so many unnecessary columns with so many null values, so we are dropping all such columns.
weather_india = weather_data[weather_data['Country/Region']=='India'].copy()

# We drop unnecessary features
weather_india_final = weather_india.drop(['ELEVATION', 'PRCP_ATTRIBUTES','TAVG_ATTRIBUTES','TMAX_ATTRIBUTES','TMIN_ATTRIBUTES','DAPR','MDP')

# Train different regression models. Below are samples.
linear_model = LinearRegression()
poly_model = PolynomialFeatures(degree=2)
sgd_model = SGDRegressor()
lasso_model = Lasso(alpha=1.0)
ridge_model = Ridge(alpha=1.0)
elasticnet_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
svr_model = SVR(kernel='linear')

#Contributing
Fork the project.
Create a new branch (git checkout -b feature).
Make changes and commit them (git commit -m 'Add feature').
Push to the branch (git push origin feature).
Open a pull request.
Contributions are welcome!
```
