import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.cluster import AgglomerativeClustering, DBSCAN


def run_random_search(X_train, y_train):

    params = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }

    rs = RandomizedSearchCV(SVC(), params, n_iter=2)
    rs.fit(X_train, y_train)

    return rs.best_estimator_


def classification_models(X_train, X_test, y_train, y_test, best_svm):

    models = {
        "SVM": best_svm,
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "NaiveBayes": GaussianNB(),
        "MLP": MLPClassifier(max_iter=200)
    }

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        results.append([
            name,
            accuracy_score(y_train, y_train_pred),
            accuracy_score(y_test, y_test_pred),
            precision_score(y_test, y_test_pred, average='weighted'),
            recall_score(y_test, y_test_pred, average='weighted'),
            f1_score(y_test, y_test_pred, average='weighted')
        ])

    return pd.DataFrame(results, columns=[
        "Model", "Train Acc", "Test Acc", "Precision", "Recall", "F1"
    ])


def regression_models(X_train, X_test, y_train, y_test):

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor()
    }

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append([name, mse, r2])

    return pd.DataFrame(results, columns=["Model", "MSE", "R2"])


def clustering_models(X):

    agg = AgglomerativeClustering(n_clusters=3)
    agg_labels = agg.fit_predict(X)

    db = DBSCAN(eps=0.5, min_samples=5)
    db_labels = db.fit_predict(X)

    return agg_labels, db_labels


def main():

    # LOAD DATA
    data = pd.read_csv(r"C:\Users\sahil\Downloads\final_all_data\Water_Consumption_Data_July_2025_0.csv")

    # CLEAN COLUMN NAMES
    data.columns = data.columns.str.strip()

    print("Columns:", data.columns.tolist())

    # DROP NON-NUMERIC / USELESS COLUMNS
    data = data.drop(columns=["Ward NO", "Ward Name"], errors='ignore')

    # HANDLE MISSING VALUES
    data = data.fillna(0)

    target_column = "Consumption in ML"

    # CLASSIFICATION TARGET
    y_class = pd.qcut(data[target_column], q=3, labels=[0, 1, 2])

    # REGRESSION TARGET
    y_reg = data[target_column]

    # FEATURES
    X = data.drop(columns=[target_column]).values

   
    # A2: RANDOM SEARCH
    print("\nRunning RandomizedSearchCV...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2)

    best_svm = run_random_search(X_train, y_train)

 
    # A3: CLASSIFICATION
    print("Running Classification...")
    clf_df = classification_models(X_train, X_test, y_train, y_test, best_svm)
    print("\n=== Classification Results ===")
    print(clf_df)

    
    # A4: REGRESSION
    print("\nRunning Regression...")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2)

    reg_df = regression_models(X_train_r, X_test_r, y_train_r, y_test_r)
    print("\n=== Regression Results ===")
    print(reg_df)

  
    # A5: CLUSTERING
    print("\nRunning Clustering...")
    agg_labels, db_labels = clustering_models(X)

    print("\n=== Clustering Results ===")
    print("Agglomerative clusters:", np.unique(agg_labels))
    print("DBSCAN clusters:", np.unique(db_labels))


if __name__ == "__main__":
    main()