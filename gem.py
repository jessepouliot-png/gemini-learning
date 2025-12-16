# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Step 2: Load the dataset
# The Iris dataset is used here as a simple, built-in example
iris = load_iris()
X, y = iris.data, iris.target

# Step 3: Split the data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional Step 4: Preprocess the data (Feature Scaling)
# Scaling is important for algorithms like KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Choose and train the model
# Using K-nearest Neighbors with k=3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 6: Make predictions and evaluate the model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Step 7: Use the model to predict a new data point (example)
# A new sample (sepal length, sepal width, petal length, petal width)
new_data = [[5.1, 3.5, 1.4, 0.2]]
# Scale the new data point using the same scaler
new_data_scaled = scaler.transform(new_data) 
prediction = model.predict(new_data_scaled)
print(f"Prediction for new data point: {iris.target_names[prediction[0]]}")
