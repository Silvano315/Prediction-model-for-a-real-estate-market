import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import skew


# Function to view model performances for a specific metric ['mse', 'r2', 'rmse']
def plot_model_performance(results, metric='mse'):
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

    plt.figure(figsize=(14,8))
    sns.barplot(x='Model', y='Mean_' + metric.upper(), hue='Dataset', data=summary_df, errorbar=None)

    #for i, row in summary_df.iterrows():
    #     plt.text(i//2, row['Mean_' + metric.upper()], f"{row['Mean_' + metric.upper()]:.2f}", ha='center', va='bottom')
    
    plt.title(f'Model Performance Comparison (Mean {metric.upper()})')
    plt.ylabel(f'Mean Squared Error ({metric.upper()})')
    plt.xlabel('Model')
    plt.show()

    return summary_df


# Function to view model performances using a violin plot for a specific metric ['mse', 'r2', 'rmse']
def violin_plots_performance(results, metric='mse'):
    detailed_data = []

    for model_type, res in results.items():
        for val in res['train_' + metric]:
            detailed_data.append({
                'Model': model_type,
                'Dataset': 'Train',
                metric.upper(): val
            })
        for val in res['test_' + metric]:
            detailed_data.append({
                'Model': model_type,
                'Dataset': 'Test',
                metric.upper(): val
            })

    detailed_df = pd.DataFrame(detailed_data)

    plt.figure(figsize=(14, 8))

    sns.violinplot(x='Model', y=metric.upper(), hue='Dataset', data=detailed_df, split=False)

    plt.title(f'{metric.upper()} Distribution Across Models')
    plt.ylabel(f'{metric.upper()}')
    plt.xlabel('Model')

    plt.show()


# Function to plot residuals and scatter plot for each fold
def plot_residuals_and_scatter(y_true_fold, y_pred_fold, model_type, fold = None, dataset='Test'):
    residuals = y_true_fold - y_pred_fold
    
    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)
    skewness = skew(residuals)
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot residuals distribution
    sns.histplot(residuals, kde=True, bins=30, ax=axs[0])
    if fold != None:
        axs[0].set_title(f'Residuals Distribution for {model_type} (Fold {fold+1}, {dataset} Set)')
    else:
        axs[0].set_title(f'Residuals Distribution for {model_type}, {dataset} Set)')
    axs[0].set_xlabel('Residuals')
    axs[0].set_ylabel('Density')
    axs[0].axvline(0, color='red', linestyle='--')
    axs[0].text(0.95,0.95, 
                f'Mean: {mean_residuals:.2f}\nStd: {std_residuals:.2f}\nSkewness: {skewness:.2f}',
                transform=axs[0].transAxes, fontsize=12, 
                ha='right', va='top', 
                bbox=dict(facecolor='white', alpha=0.5))
    
    # Plot scatter plot
    axs[1].scatter(y_true_fold, y_pred_fold, alpha=0.5)
    axs[1].plot([y_true_fold.min(), y_true_fold.max()], 
                [y_true_fold.min(), y_true_fold.max()], 'r--')
    axs[1].set_title('True vs Prediction')
    axs[1].set_xlabel('True Values')
    axs[1].set_ylabel('Predicted Values')
    
    plt.show()