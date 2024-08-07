import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Function to train linear models according to the required type
def train_model(X_train, y_train, model_type = 'linear', **kwargs):

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(**kwargs)
    elif model_type == 'lasso':
        model = Lasso(**kwargs)
    elif model_type == 'elasticnet':
        model = ElasticNet(**kwargs)
    else:
        raise ValueError("Invalid model type. Choose from 'linear', 'ridge', 'lasso', 'elasticnet'.")

    model.fit(X_train, y_train)
    return model


# Function to evaluate the model
def evaluate_model(model, X_test, y_test, X_train, y_train):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    print(f'Mean Squared Error for Training set: {mse_train:.2f}')
    print(f'R^2 Score for Training set: {r2_train:.4f}')
    print(f'Mean Squared Error for Test set: {mse_test:.2f}')
    print(f'R^2 Score for Test set: {r2_test:.4f}')