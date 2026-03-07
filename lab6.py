from turtle import pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree


def entropy(y):

    values, counts = np.unique(y, return_counts=True)

    probabilities = counts / counts.sum()

    entropy_value = -np.sum(probabilities * np.log2(probabilities))

    return entropy_value


def gini_index(y):

    values, counts = np.unique(y, return_counts=True)

    probabilities = counts / counts.sum()

    gini = 1 - np.sum(probabilities ** 2)

    return gini


def information_gain(X, y, feature_index):

    parent_entropy = entropy(y)

    feature_values = np.unique(X[:, feature_index])

    weighted_entropy = 0

    for value in feature_values:

        subset_y = y[X[:, feature_index] == value]

        weight = len(subset_y) / len(y)

        weighted_entropy += weight * entropy(subset_y)

    info_gain = parent_entropy - weighted_entropy

    return info_gain


def best_feature(X, y):

    gains = []

    for i in range(X.shape[1]):

        gain = information_gain(X, y, i)

        gains.append(gain)

    best = np.argmax(gains)

    return best


def equal_width_binning(data, bins=10):

    min_val = np.min(data)

    max_val = np.max(data)

    width = (max_val - min_val) / bins

    binned = np.floor((data - min_val) / width)

    return binned.astype(int)



def build_simple_tree(X, y, feature_names):

    root_feature = best_feature(X, y)

    tree = {
        "root_feature_index": root_feature,
        "root_feature_name": feature_names[root_feature]
    }

    return tree


def visualize_tree(X, y, feature_names, class_names):

    model = DecisionTreeClassifier()

    model.fit(X, y)

    plt.figure(figsize=(12,8))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True
    )

    plt.title("A6 - Decision Tree Visualization")

    plt.show()

def decision_boundary_plot(X, y, feature_names):

    model = DecisionTreeClassifier()

    model.fit(X, y)

    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1

    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)

    plt.scatter(X[:,0], X[:,1], c=y)

    plt.xlabel(feature_names[0])

    plt.ylabel(feature_names[1])

    plt.title("A7 - Decision Boundary")

    plt.show()


# MAIN 

if __name__ == "__main__":


    # Load dataset
    data = pd.read_csv(r"C:\Users\sahil\Downloads\Archive\final_combined_dataset.csv")

    # Remove Ward number because it is just an ID
    data = data.drop(columns=["Ward NO"])

    # Fill missing values
    data = data.fillna(0)

    target_column = "Consumption in ML"

    # Convert continuous values into categories
    y = pd.qcut(data[target_column], q=3, labels=[0,1,2])

    # Features
    X = data.drop(columns=[target_column]).values

    feature_names = data.drop(columns=[target_column]).columns
    class_names = ["Low", "Medium", "High"]

   
    # A1 : Entropy
    print("A1 - Entropy:", entropy(y))

    # A2 : Gini Index
    print("A2 - Gini Index:", gini_index(y))

    # A3 : Best Feature
    best = best_feature(X, y)

    print("A3 - Best Feature for Root:", feature_names[best])

    # A4 : Binning Example
    X_binned = X.copy()

    X_binned[:,0] = equal_width_binning(X[:,0], bins=5)

    print("A4 - Binning applied on feature 0")

    # A5 : Simple Decision Tree
    tree = build_simple_tree(X, y, feature_names)

    print("A5 - Simple Tree Root:", tree)

    # A6 : Visualize Decision Tree
    visualize_tree(X, y, feature_names, class_names)

    # A7 : Decision Boundary
    X_two_features = X[:, :2]

    decision_boundary_plot(X_two_features, y, feature_names)