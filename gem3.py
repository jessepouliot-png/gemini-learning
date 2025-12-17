from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Load the data
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

# 2. Split and Scale the data
# We use the same split and scaling as our previous conversation to ensure consistency.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Initialize and Train the Optimized Model
# Use the best 'k' value found in our Grid Search: n_neighbors = 3
best_k = 3
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)

# 4. Evaluate the model
y_pred = knn_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"--- Optimized K-Nearest Neighbors (KNN) Model ---")
print(f"Optimal 'k' used: {best_k}")
print(f"Test Set Accuracy: {test_accuracy*100:.2f}%")

# 5. Prediction Example
# Predict the species for a new sample [Sepal L, Sepal W, Petal L, Petal W]
new_sample = np.array([[5.8, 2.7, 4.2, 1.3]]) 
new_sample_scaled = scaler.transform(new_sample)
prediction = knn_model.predict(new_sample_scaled)
predicted_species = target_names[prediction[0]]

print(f"\nPrediction for new sample {new_sample[0]}: {predicted_species}")