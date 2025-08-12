#
# Task 6: K-Nearest Neighbors (KNN) Classification
# AI & ML Internship - Elevate Labs
#
# Objective: Understand and implement KNN for classification problems.
# Tools: Scikit-learn, Pandas, Matplotlib, Seaborn
# Dataset: Iris Dataset
#

# 1. Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import DecisionBoundaryDisplay

print("--- KNN Classification on Iris Dataset ---")

# 2. Load and Prepare the Dataset
iris = load_iris()
# For easier visualization of decision boundaries, we will use only the first two features.
X = iris.data[:, :2] 
y = iris.target

# Create a pandas DataFrame for easier exploration (optional)
df = pd.DataFrame(data=np.c_[iris['data'][:,:2], iris['target']],
                     columns=['Sepal Length (cm)', 'Sepal Width (cm)', 'Target'])
print("\nFirst 5 rows of the dataset:")
print(df.head())


# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# 4. Normalize Features
# Normalization is crucial for distance-based algorithms like KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Experiment with different values of K
k_range = range(1, 26)
accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plotting accuracies to find the optimal K
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracies, marker='o', linestyle='dashed')
plt.title('Accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Let's choose K based on the plot. Usually, an odd number is preferred to avoid ties.
# We will select the K that gives the highest accuracy.
optimal_k = k_range[np.argmax(accuracies)]
print(f"\nOptimal value of K found: {optimal_k}")

# 6. Train the final model with the optimal K
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train_scaled, y_train)
y_pred_optimal = knn_optimal.predict(X_test_scaled)

# 7. Evaluate the Model
# Accuracy Score
accuracy = accuracy_score(y_test, y_pred_optimal)
print(f"\nModel Accuracy with K={optimal_k}: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_optimal)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title(f'Confusion Matrix for KNN (K={optimal_k})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png') # Save the figure
print("\nConfusion matrix plot saved as 'confusion_matrix.png'")


# Classification Report
class_report = classification_report(y_test, y_pred_optimal, target_names=iris.target_names)
print("\nClassification Report:")
print(class_report)

# 8. Visualize Decision Boundaries
print("\nGenerating and saving decision boundary plot...")

# Define color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ['darkred', 'darkgreen', 'darkblue']

# Create an instance of the classifier and fit the data
clf = KNeighborsClassifier(n_neighbors=optimal_k)
clf.fit(X_train_scaled, y_train)

# Plot the decision boundary
fig, ax = plt.subplots(figsize=(10, 8))
DecisionBoundaryDisplay.from_estimator(
    clf,
    X_train_scaled,
    cmap=cmap_light,
    ax=ax,
    response_method="predict",
    xlabel=iris.feature_names[0],
    ylabel=iris.feature_names[1],
)

# Plot the training points
sns.scatterplot(
    x=X_train_scaled[:, 0],
    y=X_train_scaled[:, 1],
    hue=iris.target_names[y_train],
    palette=cmap_bold,
    alpha=1.0,
    edgecolor="black",
    ax=ax
)
plt.title(f"2-Class KNN Decision Boundary (K={optimal_k})")
plt.savefig('decision_boundary.png') # Save the figure
print("Decision boundary plot saved as 'decision_boundary.png'")

print("\n--- Task Completed ---")

