import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from src.constants import RANDOM_SEED
from src.data_engineering import feature_scaling_standardization


# Function to train linear models according to the required type
def train_model(X_train, y_train, model_type = 'linear', alpha = 1.0, l1_ratio = 0.5):

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha)
    elif model_type == 'lasso':
        model = Lasso(alpha)
    elif model_type == 'elasticnet':
        model = ElasticNet(alpha, l1_ratio=l1_ratio)
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


# Function to do K-Fold Cross-Validation with different Linear models
def evaluate_models_with_kfold(X, y, model_types, k=5, alpha = 1.0, l1_ratio = 0.5):
    kf_cv = KFold(n_splits = k, shuffle = True, random_state = RANDOM_SEED)
    results = {model_type : {'train_mse': [], 'test_mse': []} for model_type in model_types}

    for i, (idx_train,idx_test) in enumerate(kf_cv.split(X)):
        print("-"*30)
        print(f"Evaluating Fold {i+1}:")
        X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
        y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

        #Standardization
        X_train_scaled, X_test_scaled = feature_scaling_standardization(X_train, X_test)

        for model_type in model_types:
            print(f"Training {model_type.capitalize()} model...")
            model = train_model(X_train_scaled, y_train, model_type, alpha, l1_ratio)

            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)

            results[model_type]['train_mse'].append(train_mse)
            results[model_type]['test_mse'].append(test_mse)
        print("-"*30)

    return results


# Functiom to show a bar chart about mean and std of MSE values for each model and for both train and test
def plot_mse_summary(results):

    summary_data = []
    for model_type, res in results.items():
        train_mean = np.mean(res['train_mse'])
        train_std = np.std(res['train_mse'])
        test_mean = np.mean(res['test_mse'])
        test_std = np.std(res['test_mse'])

        summary_data.append({
            'Model': model_type,
            'Dataset': 'Train',
            'Mean_MSE': train_mean,
            'Std_MSE': train_std
        })
        summary_data.append({
            'Model': model_type,
            'Dataset': 'Test',
            'Mean_MSE': test_mean,
            'Std_MSE': test_std
        })

    summary_df = pd.DataFrame(summary_data)

    fig = go.Figure()

    for dataset in ['Train', 'Test']:
        df_subset = summary_df[summary_df['Dataset'] == dataset]
        fig.add_trace(go.Bar(
            x=df_subset['Model'],
            y=df_subset['Mean_MSE'],
            name=f'{dataset} Mean MSE',
            error_y=dict(type='data', array=df_subset['Std_MSE'], visible=True),
        ))

    fig.update_layout(
        title='Mean and Std Comparison about MSE for each model',
        xaxis_title='Model',
        yaxis_title='Mean Squared Error (MSE)',
        barmode='group',
        legend_title='Dataset',
        template='plotly_white'
    )

    #fig.show()

    return fig