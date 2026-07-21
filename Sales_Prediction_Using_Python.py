import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# STEP 1: DATA LOADING & CLEANING
print(" STEP 1: DATA CLEANING & TRANSFORMATION ")
df = pd.read_csv('Advertising.csv')
# Drop index columns if present (e.g., 'Unnamed: 0')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
# Cleaning column names and handling missing values
df.columns = df.columns.str.strip()
df = df.dropna()
print("Dataset Head:\n", df.head())
# STEP 2: FEATURE SELECTION & SPLITTING
print("\n--- STEP 2: FEATURE SELECTION & DATA SPLITTING ---")
# Define features (advertising platforms) and target (Sales)
features = ['TV', 'Radio', 'Newspaper']
target = 'Sales'

X = df[features]
y = df[target]

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training shapes: X_train={X_train.shape}, y_train={y_train.shape}")
# STEP 3: MODEL TRAINING & EVALUATION
print("\n--- STEP 3: REGRESSION MODEL TRAINING ---")
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Evaluation Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f" - Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f" - R-squared (Accuracy Score): {r2:.4f} ({r2*100:.1f}%)")
# STEP 4: ACTIONABLE MARKETING INSIGHTS
print("\n--- STEP 4: BUSINESS INSIGHTS ---")
coefficients = pd.DataFrame({'Platform': features, 'Impact Coefficient': model.coef_})
coefficients = coefficients.sort_values(by='Impact Coefficient', ascending=False)

print("Sales Impact per Dollar Spent:")
print(coefficients.to_string(index=False))

print("\n💡 Actionable Strategy:")
best_platform = coefficients.iloc[0]['Platform']
worst_platform = coefficients.iloc[-1]['Platform']
print(f"-> Maximize ROI by shifting budget focus toward {best_platform}.")
print(f"-> Re-evaluate or reduce spending on {worst_platform} as it yields the lowest returns.")

