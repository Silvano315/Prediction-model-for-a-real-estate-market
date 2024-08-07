import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Function to check if there are duplicates in the dataset
def handle_duplicates(df):

    duplicates = df.duplicated().sum()
    
    if duplicates > 0:
        print(f"Number of duplicates: {duplicates}. Shape DF: {df.shape}")
        df_cleaned = df.drop_duplicates()
        print(f"Number of removed duplicates: {duplicates}. New shape DF: {df_cleaned.shape}")
    else:
        print("No duplicates found!")
        df_cleaned = df

    return df_cleaned

# Function to perform EDA with plotly.express: bar plot for categorical features and histogram for numerical 
def plot_feature_distribution(df, feature, is_categorical):

    if is_categorical:
        value_counts = df[feature].value_counts().reset_index()
        value_counts.columns = [feature, 'count']
        fig = px.bar(value_counts,
                     x=feature, y='count',
                     labels={feature: feature, 'count' : 'Count'},
                     title=f'Distribution of {feature}')
    else:
        fig = px.histogram(df, x = feature, labels={feature:feature},
                           marginal='box', #'violin'
                           title=f'Distribution of {feature}')
    fig.show()