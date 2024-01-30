import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


csvfile = "diabetes_prediction_dataset.csv"

def createMatrix(size,split):
    """
    Features:
    age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level.
    """
    df = pd.read_csv(csvfile)
    df.to_numpy()
    full = np.column_stack((np.ones(size), df))
    X = full[:, :-1]
    Y = full[:, -1].astype(int)

    # Making all the data numeric
    X[:, 1] = np.where(X[:, 1] == 'Female', 0, X[:, 1])
    X[:, 1] = np.where(X[:, 1] == 'Male', 1, X[:, 1])
    X[:, 1] = np.where(X[:, 1] == 'Other', 0.5, X[:, 1])
    X[:, 5] = np.where(X[:, 5] == 'No Info', 0, X[:, 5])
    X[:, 5] = np.where(X[:, 5] == 'never', 1, X[:, 5])
    X[:, 5] = np.where(X[:, 5] == 'ever', 1, X[:, 5])
    X[:, 5] = np.where(X[:, 5] == 'former', 2, X[:, 5])
    X[:, 5] = np.where(X[:, 5] == 'not current', 3, X[:, 5])
    X[:, 5] = np.where(X[:, 5] == 'current', 4, X[:, 5])

    X_train = X[:split , :]
    X_test = X[split:,:]
    y_train = Y[:split]
    y_test = Y[split:]

    return X,Y,X_train,X_test,y_train,y_test

def euclideanDist(point1, point2):
    squared_diff = np.square(point1 - point2)
    sum = np.sum(squared_diff)
    return np.sqrt(sum)

def knn_algorithm(k,X_train,X_testInstance,y_train,iterationNum):
    dist = np.array([euclideanDist(X_testInstance, trainInstance) for trainInstance in X_train])
    indexList = np.argsort(dist) # creating list of the indices of points closest to the test

    countY = np.sum(y_train[indexList[:k]]) # since yes is 1 and no is 0, sum of the first k elements will give num of yes
    countN = k - countY

    print(iterationNum,") Yes: ", countY, "and No: ", countN)
    if countY >= countN:
        return 1
    else:
        return 0

def knn(k):
    split = 80000
    X,Y,X_train,X_test,y_train,y_test = createMatrix(100000,split)
    y_test = y_test.astype(np.int64)

    y_pred = np.array([knn_algorithm(k,X_train,testI,y_train,i) for i,testI in enumerate(X_test)])

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)

def knnWithLibraries():
    split = 70000
    X, Y, X_train,X_test,y_train,y_test = createMatrix(100000,split)
    y_train = y_train.astype(np.int64)
    k_values = np.arange(1, 21)
    cvScores = []

    for k in k_values:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn_model, X_train, y_train, cv=5)
        cvScores.append(np.mean(scores))

    optimal_k = k_values[np.argmax(cvScores)]
    print(f"The optimal k value is: {optimal_k}")

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, cvScores, marker='o', linestyle='-', color='b')
    plt.title('Cross-Validated Accuracy for Different k Values')
    plt.xlabel('k Value')
    plt.ylabel('Cross-Validated Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

def naiveBayes():
    X,Y , X_train, X_test, y_train, y_test = createMatrix(100000,80000)    
    y_train = y_train.astype(np.int64)

    nb_classifier = GaussianNB()

    # Train the classifier on the training data
    nb_classifier.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = nb_classifier.predict(X_test)

    # Evaluate the performance of the classifier
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)

# the function knnWithLibraries has been used to determine the 
# ideal k value. this value (7) has been put into knn(). 
knn(7)
naiveBayes()

