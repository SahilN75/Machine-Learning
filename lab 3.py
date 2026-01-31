
# lab session 03
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import minkowski

# function to calculate dot product manually
def dot_product(a, b):
    result = 0                    # initializing result variable
    for i in range(len(a)):       # looping through vector elements
        result = result + a[i] * b[i]   # multiplying and adding
    return result                 # returning final value


# function to calculate euclidean norm manually
def euclidean_norm(v):
    s = 0                         # variable to store sum
    for val in v:                 # loop through values of vector
        s = s + val * val         # squaring and adding
    return np.sqrt(s)             # square root of total

# function to compute mean vector
def mean_vector(data):
    total_rows = data.shape[0]    # number of samples
    return np.sum(data, axis=0) / total_rows   # average calculation


# function to compute variance vector
def variance_vector(data):
    mu = mean_vector(data)        # computing mean first
    diff = data - mu              # subtract mean
    return np.sum(diff ** 2, axis=0) / data.shape[0]  # variance formula


# function to compute std deviation vector
def std_vector(data):
    return np.sqrt(variance_vector(data))   # sqrt of variance


# function to compute minkowski distance
def minkowski_distance(a, b, p):
    dist = 0                      # distance accumulator
    for i in range(len(a)):       # loop through features
        dist = dist + abs(a[i] - b[i]) ** p   # minkowski formula
    return dist ** (1 / p)        # final distance



# function for manual knn prediction
def custom_knn(x_train, y_train, test_point, k):
    distances = []                # list to store distances

    # loop through training samples
    for i in range(len(x_train)):
        d = euclidean_norm(x_train[i] - test_point)   # distance
        distances.append((d, y_train[i]))             # store with label

    # sorting distances in ascending order
    distances.sort(key=lambda x: x[0])

    # selecting k nearest neighbors
    nearest = distances[:k]

    votes = {}                    # dictionary to count votes

    # counting class votes
    for _, label in nearest:
        votes[label] = votes.get(label, 0) + 1

    # returning label with maximum votes
    return max(votes, key=votes.get)




# function to compute confusion matrix values
def confusion_matrix(y_true, y_pred):
    tp = tn = fp = fn = 0         # initializing counts

    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        else:
            fn += 1

    return tp, tn, fp, fn


# function to compute accuracy
def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


# function to compute precision
def precision(tp, fp):
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


# function to compute recall
def recall(tp, fn):
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


# function to compute f1 score
def f1_score(p, r):
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)



# loading csv file (keep csv in same folder as this code)
file_path = r"C:\Users\sahil\Downloads\LAB3\Household_Profile_Bengaluru_ason_01-03-2011.csv"
df = pd.read_csv(file_path)

# selecting only numeric columns
df_num = df.select_dtypes(include=np.number)

# removing missing values
df_num = df_num.dropna()

# selecting one column to create binary target
base_col = df_num.columns[0]

# creating class label using median split
df_num["class"] = (df_num[base_col] > df_num[base_col].median()).astype(int)

# separating features and target
x = df_num.drop(columns=["class"]).values
y = df_num["class"].values

a = x[0]
b = x[1]

print("dot product (custom):", dot_product(a, b))
print("dot product (numpy):", np.dot(a, b))

print("euclidean norm (custom):", euclidean_norm(a))
print("euclidean norm (numpy):", np.linalg.norm(a))

class0 = x[y == 0]
class1 = x[y == 1]

mean0 = mean_vector(class0)
mean1 = mean_vector(class1)

std0 = std_vector(class0)
std1 = std_vector(class1)

interclass_dist = euclidean_norm(mean0 - mean1)
print("\ninterclass distance:", interclass_dist)


feature = x[:, 0]

plt.hist(feature, bins=10)
plt.xlabel("feature value")
plt.ylabel("frequency")
plt.title("feature distribution")
plt.show()

print("mean of feature:", np.mean(feature))
print("variance of feature:", np.var(feature))

p_vals = range(1, 11)
dist_list = []

for p in p_vals:
    dist_list.append(minkowski_distance(a, b, p))

plt.plot(p_vals, dist_list, marker="o")
plt.xlabel("p value")
plt.ylabel("distance")
plt.title("minkowski distance vs p")
plt.show()


print("custom minkowski p=3:", minkowski_distance(a, b, 3))
print("scipy minkowski p=3:", minkowski(a, b, 3))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)


print("\nknn accuracy:", knn.score(x_test, y_test))


y_pred = knn.predict(x_test)
print("sample predictions:", y_pred[:5])



custom_pred = []

for pt in x_test:
    custom_pred.append(custom_knn(x_train, y_train, pt, 3))


k_values = range(1, 12)
acc_values = []

for k in k_values:
    temp_knn = KNeighborsClassifier(n_neighbors=k)
    temp_knn.fit(x_train, y_train)
    acc_values.append(temp_knn.score(x_test, y_test))

plt.plot(k_values, acc_values, marker="o")
plt.xlabel("k value")
plt.ylabel("accuracy")
plt.title("accuracy vs k")
plt.show()


tp, tn, fp, fn = confusion_matrix(y_test, y_pred)

acc = accuracy(tp, tn, fp, fn)
prec = precision(tp, fp)
rec = recall(tp, fn)
f1 = f1_score(prec, rec)

print("\nconfusion matrix")
print("tp:", tp, "tn:", tn, "fp:", fp, "fn:", fn)

print("\nperformance metrics")
print("accuracy:", acc)
print("precision:", prec)
print("recall:", rec)
print("f1 score:", f1)
