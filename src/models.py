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
        model = LinearRegression(n_jobs=-1)
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


# Function to count Non-Zero Coefficients
def count_non_zero_coefficients(model, X_train):

    coeff = model.coef_
    no_zero_count = np.sum(coeff != 0)
    return no_zero_count


# Function to do K-Fold Cross-Validation with different Linear models
def evaluate_models_with_kfold(X, y, model_types, k=5, alpha = 1.0, l1_ratio = 0.5):
    kf_cv = KFold(n_splits = k, shuffle = True, random_state = RANDOM_SEED)
    results = {model_type : {'train_mse': [], 'test_mse': [], 
                             'train_r2': [], 'test_r2': [], 
                             'train_rmse': [], 'test_rmse': [], 
                             'non_zero_coeff': []} for model_type in model_types}
    y_true_pred = {model_type : {
        'y_true': [], 'y_pred': []
    } for model_type in model_types}

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
            y_true_pred[model_type]['y_true'].append(y_test)
            y_true_pred[model_type]['y_pred'].append(y_test_pred)

            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            non_zero_count = count_non_zero_coefficients(model, X_train_scaled)

            results[model_type]['train_mse'].append(train_mse)
            results[model_type]['test_mse'].append(test_mse)
            results[model_type]['train_r2'].append(train_r2)
            results[model_type]['test_r2'].append(test_r2)
            results[model_type]['train_rmse'].append(train_rmse)
            results[model_type]['test_rmse'].append(test_rmse)
            results[model_type]['non_zero_coeff'].append(non_zero_count)

        print("-"*30)

    return results, y_true_pred


# Functiom to show a bar chart about mean and std of MSE values for each model and for both train and test
def plot_mse_summary(results, metric = 'mse'):
    
    summary_data = []
    for model_type, res in results.items():
        train_mean = np.mean(res['train_' + metric])
        train_std = np.std(res['train_' + metric])
        test_mean = np.mean(res['test_' + metric])
        test_std = np.std(res['test_' + metric])

        summary_data.append({
            'Model': model_type,
            'Dataset': 'Train',
            'Mean_' + metric.upper(): train_mean,
            'Std_' + metric.upper(): train_std
        })
        summary_data.append({
            'Model': model_type,
            'Dataset': 'Test',
            'Mean_' + metric.upper(): test_mean,
            'Std_' + metric.upper(): test_std
        })

    summary_df = pd.DataFrame(summary_data)

    fig = go.Figure()

    for dataset in ['Train', 'Test']:
        df_subset = summary_df[summary_df['Dataset'] == dataset]
        fig.add_trace(go.Bar(
            x=df_subset['Model'],
            y=df_subset['Mean_' + metric.upper()],
            name=f'{dataset} Mean ' + metric.upper(),
            error_y=dict(type='data', array=df_subset['Std_' + metric.upper()], visible=True),
        ))

    fig.update_layout(
        title=f'Mean and Std Comparison about {metric.upper()} for each model',
        xaxis_title='Model',
        yaxis_title=f'Mean Squared Error ({metric.upper()})',
        barmode='group',
        legend_title='Dataset',
        template='plotly_white'
    )

    return fig, summary_df


#Function to see models complexity with Non-Zero Coefficients
def plot_model_complexity(results):
    complexity_data = []
    for model_type, res in results.items():
        non_zero_mean = np.mean(res['non_zero_coeff'])
        non_zero_std = np.std(res['non_zero_coeff'])

        complexity_data.append({
            'Model': model_type,
            'Mean_Non_Zero_Coefficients': non_zero_mean,
            'Std_Non_Zero_Coefficients': non_zero_std
        })

    complexity_df = pd.DataFrame(complexity_data)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=complexity_df['Model'],
        y=complexity_df['Mean_Non_Zero_Coefficients'],
        text=complexity_df['Mean_Non_Zero_Coefficients'],
        textposition='inside',  
        error_y=dict(type='data', array=complexity_df['Std_Non_Zero_Coefficients'], visible=True),
        name='Complexity (Non-Zero Coefficients)'
    ))

    fig.update_layout(
        title='Complexity Comparison for each model (Coefficients not zero)',
        xaxis_title='Model',
        yaxis_title='Mean Number of Non Zero Coefficients',
        barmode='group',
        template='plotly_white'
    )

    fig.show()