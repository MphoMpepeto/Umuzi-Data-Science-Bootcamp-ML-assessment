#load, clean and scale data for later analysis

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean(filepath):
    # Load dataset
    df = pd.read_csv(filepath)
    print("Data loaded successfully!")
    print("First 5 rows of the dataset:")
    print(df.head())  # Show first 5 rows of the dataset as a preview
    
    # Drop Region and Channel columns from the dataset as they are categorical variables
    df = df.drop(columns=["Channel", "Region"], errors="ignore")
    print("Dropped 'Channel' and 'Region' columns.")
    print("Remaining columns:", df.columns.tolist())
    
    # Scale the data (mean = 0 and SD = 1)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    print("Data has been scaled. Here are the first 5 rows of scaled data:")
    print(df_scaled[:5])  # Show first 5 scaled rows
    
    return df, df_scaled

if __name__ == "__main__":
    filepath = "../Data/Wholesale customers data.csv"
    df, df_scaled = load_and_clean(filepath)