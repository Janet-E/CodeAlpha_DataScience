import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# STEP 1: LOAD DATA & PREPROCESSING
df = pd.read_csv('car data.csv')
# Cleaning column names by removing any accidental spaces
df.columns = df.columns.str.strip()
# Print columns to verify successful loading
print(" STEP 1: DATA LOADED ")
print("Dataset Columns:", df.columns.tolist())
#STEP 2: FEATURE ENGINEERING & ENCODING
print("\n--- STEP 2: FEATURE ENGINEERING ---")
# Feature Engineering: Calculate the age of the car to give the model better data
df['Car_Age'] = 2026 - df['Year']
# Handling Categorical Features: Convert text columns into numbers (One-Hot Encoding)
# This converts Fuel_Type, Selling_type, and Transmission into machine-readable format
categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# STEP 3: SPLITTING DATA & TRAINING MODEL
print("\n STEP 3: SPLITTING DATA & TRAINING MODEL ")
# Define target variable using exact capitalization
target = 'Selling_Price'
# Define numerical features (including our new Car_Age feature)
num_features = ['Present_Price', 'Driven_kms', 'Owner', 'Car_Age']
# Get the names of the newly created encoded columns
encoded_features = [col for col in df_encoded.columns if col not in num_features + [target, 'Car_Name', 'Year']]
# Combine all valid features
all_features = num_features + encoded_features
# Separate features (X) and target (y)
X = df_encoded[all_features]
y = df_encoded[target]
# Perform Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Success! Data split complete.")
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
