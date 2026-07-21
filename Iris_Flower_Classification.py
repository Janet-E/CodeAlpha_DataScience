import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
print("STEP 1: LOADING IRIS DATASET")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
print(f"Dataset loaded with {df.shape[0]} flowers.")
print("Features:", iris.feature_names)
print("\n STEP 2: SPLITTING DATA (TRAIN VS TEST)")
X = df[iris.feature_names]  # Measurements (Sepal/Petal length & width)
y = df['species']           # Target Species (0=Setosa, 1=Versicolor, 2=Virginica)
# Split data: 80% to train the model, 20% to test its accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
print("\n--- STEP 3: TRAINING MACHINE LEARNING MODEL ---")
# Initialize a Random Forest Classifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")
print("\n STEP 4: EVALUATING PERFORMANCE ")
# Ask the model to predict the species of the test flowers it has never seen
predictions = model.predict(X_test)
# Calculate accuracy score
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy Score: {accuracy * 100:.2f}%")
print("\nDetailed Performance Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))
