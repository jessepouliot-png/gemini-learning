from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load the data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split and Scale the data (essential step for KNN optimization)
# Using the 80/20 split and random_state=42 from our previous discussions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Optimize the Model Parameters (Grid Search)
# This step systematically tests k from 1 to 30 using 5-fold cross-validation (cv=5)
param_grid = {'n_neighbors': range(1, 31)} 
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy') 
grid_search.fit(X_train_scaled, y_train)

# 4. Perform/Evaluate the Optimized Model
best_k = grid_search.best_params_['n_neighbors']
best_score_cv = grid_search.best_score_
best_model = grid_search.best_estimator_

# Evaluate the best model on the unseen test set
y_pred_test = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("--- Full Iris Model Optimization & Performance (KNN) ---")
print(f"1. Data Preparation: Loaded, Split (80/20), and Scaled.")
print(f"2. Optimization Algorithm: Grid Search with 5-Fold Cross-Validation.")
print(f"3. Optimized Model: K-Nearest Neighbors")
print(f"\n--- Optimization Results ---")
print(f"Optimal Parameter 'k' found: {best_k}")
print(f"Best Cross-Validation Score: {best_score_cv:.4f}")
print(f"\n--- Final Performance ---")
print(f"Final Test Set Accuracy: {test_accuracy*100:.2f}%")