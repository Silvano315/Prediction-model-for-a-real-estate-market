import pandas as pd
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Categorical features with type int64 conversion to 'category'
def convert_to_category(df, features):
    df_copy = df.copy()
    for feature in features:
        df_copy[feature] = df_copy[feature].astype('category')
    return df_copy


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
def plot_feature_distribution(df, feature, is_categorical, comparison = None):

    if is_categorical:
        value_counts = df.groupby(feature).agg(count=(feature, 'size'), price_mean=('price', 'mean')).reset_index()
        value_counts.columns = [feature, 'count', 'price_mean']
        if comparison != None:
            fig = px.bar(value_counts,
                        x=feature, y='count',
                        color='price_mean',
                        labels={feature: feature, 'count': 'Count', 'price_mean': 'Average Price'},
                        title=f'Distribution of {feature}')
        else:
            fig = px.bar(value_counts,
                        x=feature, y='count',
                        labels={feature: feature, 'count': 'Count'},
                        title=f'Distribution of {feature}')
    elif comparison != None:
        fig = px.histogram(df, x = feature, labels={feature:feature},
                           color = comparison,
                           title=f'Distribution of {feature} compared with {comparison}')
    else:
        fig = px.histogram(df, x = feature, labels={feature:feature},
                           marginal='box', #'violin'
                           title=f'Distribution of {feature}')
    fig.show()


# Function to do One-Hot Encoding on the categorical features
def apply_encoding(df):

    cat_features = df.select_dtypes(include='category').columns
    encoding_features = [feature for feature in cat_features if df[feature].nunique() > 2]

    df_encoded = pd.get_dummies(df, columns=encoding_features, drop_first=True, dtype=int)

    return df_encoded

# Function to do Feature Scaling with standardization
def feature_scaling_standardization(X_train, X_test):

    continuous_features = ['area'] 

    continuous_transformer = Pipeline(
        steps=[('scaler', StandardScaler())]
    )

    preprocessing = ColumnTransformer(
        transformers = [
            ('continuos', continuous_transformer, continuous_features)
        ],
        remainder = 'passthrough'
    )

    X_train_scaled = preprocessing.fit_transform(X_train)
    #X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = preprocessing.transform(X_test)
    #X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    feature_names = continuous_features + [col for col in X_train.columns if col not in continuous_features]
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

    return X_train_scaled, X_test_scaled