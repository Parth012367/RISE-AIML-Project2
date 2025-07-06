# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Handle missing data (simple example: drop missing)
    df = df.dropna()

    # Encode categorical 'education'
    le = LabelEncoder()
    df['education'] = le.fit_transform(df['education'])

    X = df[['age', 'income', 'education', 'credit_score', 'loan_amount']]
    y = df['loan_approved']

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

