# -K-Nearest-Neighbors-KNN-Classification
# **Task 6: K-Nearest Neighbors (KNN) Classification**

This project is a part of the AI & ML Internship at **Elevate Labs**, sponsored by the Ministry of MSME, Govt of India.

The objective of this task is to implement the K-Nearest Neighbors (KNN) algorithm for a classification problem. We use the popular **Iris dataset** for this purpose and leverage tools like Scikit-learn, Pandas, and Matplotlib.

---

### **Repository Contents**

* `knn_classifier.py`: The main Python script that performs the entire classification task.
* `confusion_matrix.png`: An image showing the confusion matrix of the final model's performance on the test set.
* `decision_boundary.png`: A visualization of the KNN decision boundaries for the two selected features of the Iris dataset.
* `README.md`: This file, explaining the project and task details.

---
**Implementation Steps**

1.  **Load Dataset**: The script loads the Iris dataset directly from `scikit-learn`. For simplicity in visualizing decision boundaries, only the first two features (Sepal Length and Sepal Width) are used.
2.  **Data Splitting**: The dataset is split into training (70%) and testing (30%) sets.
3.  **Feature Scaling**: Feature normalization is performed using `StandardScaler` from scikit-learn. This is a critical step for distance-based algorithms like KNN.
4.  **Find Optimal K**: The model is trained for a range of K values (1 to 25). The accuracy for each K is plotted to find the optimal value that yields the highest accuracy.
5.  **Model Training**: A final KNN classifier is trained using the optimal K value found in the previous step.
6.  **Model Evaluation**: The model's performance is evaluated on the test set using:
    * **Accuracy Score**: To get the overall correctness percentage.
    * **Confusion Matrix**: To understand the model's performance for each class.
    * **Classification Report**: For detailed metrics like precision, recall, and F1-score.
7.  **Visualize Decision Boundaries**: The decision boundaries created by the KNN classifier are plotted and saved as an image. This helps in understanding how the model separates the different classes in the feature space.

---

### **How to Run the Code**

1.  Clone the repository:
    ```bash
    git clone <your-repo-link>
    cd <repo-name>
    ```
2.  Install the required libraries:
    ```bash
    pip install scikit-learn pandas matplotlib seaborn
    ```
3.  Execute the Python script:
    ```bash
    python knn_classifier.py
    ```
4.  The script will print the evaluation metrics to the console and save `confusion_matrix.png` and `decision_boundary.png` in the same directory.

---

### **Interview Questions & Answers**

#### 1. How does the KNN algorithm work?
KNN is a simple, supervised machine learning algorithm used for both classification and regression. It is a non-parametric, instance-based learning algorithm. The core idea is:
* **Training Phase**: It stores all the available data points and their class labels. No actual model is built during this phase.
* **Prediction Phase**: To classify a new data point, it calculates the distance between the new point and all the stored data points. It then selects the 'K' nearest neighbors. The new data point is assigned to the class that is most common among its K nearest neighbors (majority vote).

#### 2. How do you choose the right K?
The choice of K is critical for the model's performance.
* **Small K**: A small K (e.g., K=1) can lead to a model that is very sensitive to noise and outliers, resulting in high variance and overfitting.
* **Large K**: A large K makes the decision boundaries smoother, making the model more resilient to noise but potentially leading to high bias and underfitting.
* **Method**: A common method is to run the KNN algorithm several times with different K values and choose the K that results in the best performance (e.g., highest accuracy) on a validation set. This is often visualized using an "elbow method" plot showing accuracy vs. K.

#### 3. Why is normalization important in KNN?
Normalization (or feature scaling) is crucial for KNN because it relies on distance metrics (like Euclidean distance) to determine the "nearness" of data points. If features are on different scales, the feature with the larger scale will dominate the distance calculation, leading to biased results. For example, if one feature ranges from 0-1 and another from 0-1000, the second feature will have a much larger impact on the distance. Normalizing all features to a similar scale ensures that each feature contributes equally to the distance calculation.

#### 4. What is the time complexity of KNN?
* **Training Time Complexity**: $O(1)$, as it simply stores the dataset.
* **Prediction Time Complexity**: $O(n \cdot d)$, where 'n' is the number of training samples and 'd' is the number of features. For each prediction, the algorithm must compute the distance to every training point. This makes KNN computationally expensive during prediction, especially for large datasets.

#### 5. What are the pros and cons of KNN?
* **Pros**:
    * **Simple to understand and implement.**
    * **No training phase**: The algorithm is instance-based, making it fast to "train."
    * **Flexible**: Easily adapts to new data.
    * **Works well for multi-class problems.**
* **Cons**:
    * **Computationally expensive**: Prediction is slow for large datasets.
    * **High memory requirement**: Needs to store the entire dataset.
    * **Sensitive to irrelevant features**: All features contribute to the distance, so irrelevant features can degrade performance.
    * **Sensitive to the scale of data and outliers.**

#### 6. Is KNN sensitive to noise?
Yes, KNN is sensitive to noise and outliers, especially with a small value of K. A noisy data point can be one of the 'K' nearest neighbors and wrongly influence the classification of a new point. This effect can be mitigated by choosing a larger value for K, which averages out the influence of noisy points.

#### 7.How does KNN handle multi-class problems?
KNN handles multi-class problems naturally. The algorithm's voting mechanism works the same regardless of the number of classes. When classifying a new point, it identifies the K nearest neighbors and their classes. The new point is then assigned to the class that appears most frequently among those neighbors (the mode).

#### 8. What's the role of distance metrics in KNN?
Distance metrics are fundamental to KNN as they define the concept of "similarity" or "nearness" between data points. The choice of metric can significantly impact the model's performance. Common metrics include:
* **Euclidean Distance ($L_2$ norm)**: The most common metric, representing the straight-line distance between two points.
* **Manhattan Distance ($L_1$ norm)**: The distance between two points measured along the axes at right angles.
* **Minkowski Distance**: A generalization of both Euclidean and Manhattan distances.
The appropriate metric often depends on the nature of the data.
