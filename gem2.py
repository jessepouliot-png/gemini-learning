from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Load the data
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

# 2. Split the data (using the same split settings as our last conversation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a Decision Tree Classifier
# Scaling is optional for Decision Trees but helps with feature importance visualization later.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# We set max_depth to 3 to keep the tree simple and prevent overfitting
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train_scaled, y_train)

# 4. Make predictions
y_pred = dt_model.predict(X_test_scaled)

# 5. Evaluate the model with a Classification Report
print("--- Decision Tree Classification Report ---")
# The report provides precision, recall, f1-score, and support for each class
report = classification_report(y_test, y_pred, target_names=target_names)
print(report)

# 6. Predict a new custom data point
# Example: A flower with features [Sepal L, Sepal W, Petal L, Petal W] = [5.8, 2.7, 4.2, 1.3]
new_sample = np.array([[5.8, 2.7, 4.2, 1.3]]) 
new_sample_scaled = scaler.transform(new_sample)
prediction = dt_model.predict(new_sample_scaled)
predicted_species = target_names[prediction[0]]

print(f"\nPrediction for new sample {new_sample[0]}: {predicted_species}")
