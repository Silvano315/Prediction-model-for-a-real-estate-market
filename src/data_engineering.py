import pandas as pd

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
