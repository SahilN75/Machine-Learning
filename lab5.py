import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def train_linear_regression(X_train, y_train):
    model = LinearRegression()  # Create Linear Regression object
    model.fit(X_train, y_train) # Train model
    return model

def evaluate_regression(model, X, y):
    y_pred = model.predict(X) # Predict values 
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse) # Calculate Root Mean Squared Error
    mape = np.mean(np.abs((y - y_pred) / y)) * 100 # Calculate Mean Absolute Percentage Error
    r2 = r2_score(y, y_pred)
    return mse, rmse, mape, r2


def perform_kmeans(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")  # Create KMeans model
    kmeans.fit(X)
    return kmeans


def evaluate_clustering(X, labels): 
    sil = silhouette_score(X, labels) # Calculate Silhouette score
    ch = calinski_harabasz_score(X, labels) # Calculate CH score
    db = davies_bouldin_score(X, labels) # Calculate DB index
    return sil, ch, db


def kmeans_multiple_k(X, k_range):
    sil_scores = []    # Store silhouette scores
    ch_scores = []  # Store CH scores
    db_scores = []  # Store DB scores 
    distortions = [] # Store inertia values

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X)
        
        sil_scores.append(silhouette_score(X, kmeans.labels_))
        ch_scores.append(calinski_harabasz_score(X, kmeans.labels_))
        db_scores.append(davies_bouldin_score(X, kmeans.labels_))
        distortions.append(kmeans.inertia_)

    return sil_scores, ch_scores, db_scores, distortions


#main

if __name__ == "__main__":
    
    file_path = "C:/Users/sahil/Downloads/Archive/Water_Consumption_Data_July_2025_0.csv"
    
    df = load_data(file_path)

    
    df.columns = df.columns.str.strip()#remove extra spaces
    target_column = "Consumption in ML"
    df_numeric = df.drop(columns=["Ward NO", "Ward Name"]) # This removes non-numeric columns

    X = df_numeric.drop(columns=[target_column])
    y = df_numeric[target_column]

    X_train, X_test, y_train, y_test = train_test_split( # Split into training and test data
        X, y, test_size=0.2, random_state=42
    )

    #A1
    single_feature = X_train.columns[0]

    model_single = train_linear_regression(
        X_train[[single_feature]], y_train  # Train using single feature
    )

    #A2
    print("Single Feature Train:", evaluate_regression(model_single, X_train[[single_feature]], y_train))
    print("Single Feature Test:", evaluate_regression(model_single, X_test[[single_feature]], y_test))

    #A3
    model_multi = train_linear_regression(X_train, y_train)  # Train multi-feature model

    print("Multi Feature Train:", evaluate_regression(model_multi, X_train, y_train))
    print("Multi Feature Test:", evaluate_regression(model_multi, X_test, y_test))

    #A4
    X_cluster = df_numeric.drop(columns=[target_column])
    kmeans_2 = perform_kmeans(X_cluster, 2)

    #A5
    sil, ch, db = evaluate_clustering(X_cluster, kmeans_2.labels_)

    print("Silhouette:", sil)
    print("Calinski-Harabasz:", ch)
    print("Davies-Bouldin:", db)

    #A6
    k_range = range(2, 10)

    sil_scores, ch_scores, db_scores, distortions = kmeans_multiple_k(
        X_cluster, k_range
    )

    plt.figure()
    plt.plot(k_range, sil_scores)
    plt.title("Silhouette Score vs K")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.show()

    #A7
    plt.figure()
    plt.plot(k_range, distortions)
    plt.title("Elbow Plot")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.show()