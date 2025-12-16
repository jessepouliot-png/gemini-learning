import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 2. Visualize the Data (Sepal Length vs Sepal Width)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Iris Dataset: Sepal Length vs Sepal Width')
plt.legend(handles=scatter.legend_elements()[0], labels=list(target_names))
plt.grid(True)
plt.savefig('iris_visualization.png')
print("Visualization saved.")

# 3. Optimize Parameters using Grid Search
# Split and Scale Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid (testing k from 1 to 30)
param_grid = {'n_neighbors': range(1, 31)}

# Initialize KNN and Grid Search
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Fit the search
grid_search.fit(X_train_scaled, y_train)

# Output results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Validate on Test Set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test_scaled, y_test)
print(f"Test Set Accuracy with Best Model: {test_accuracy*100:.2f}%")
