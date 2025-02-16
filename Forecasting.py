from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Loading the dataset
def loadingDataSet(path):
    dataset = pd.read_csv(path)
    print(dataset)
    return dataset

# Checking if there is any missing values
def checkingForMissingValues(dataset):

    missingValues = dataset.isnull().sum()
    print("The Number Of Missing values in Each Column:")
    print(missingValues)
    return missingValues

def dropingMissingData(dataset):

    no_missing_values = dataset.dropna()
    missingValues = no_missing_values.isnull().sum()
    print("The Number Of Missing values in Each Column after dropping missing data:")
    print(missingValues)
    return no_missing_values

# Replacing missing values with the average of the feature
def replacingMissingValuesWithAverage(dataset):

    no_missing_values_dataset = dataset.copy()
    for column in no_missing_values_dataset.select_dtypes(include=['number']).columns:
        no_missing_values_dataset[column] = no_missing_values_dataset[column].fillna(no_missing_values_dataset[column].mean())

    missingValues = no_missing_values_dataset.isnull().sum()
    print("Dataset after replacing missing values with the average:")
    print(no_missing_values_dataset)
    print("The Number Of Missing values in Each Column after dropping missing data:")
    print(missingValues)
    return no_missing_values_dataset

# Function to check if numeric features have the same scale
def scalingData(dataset):

    print("The Descriptive data for each numeric column:")
    print(dataset.describe())
    minMaxScaler = MinMaxScaler()

    # Selecting the columns with numeric values
    columnsWithNumericValues = dataset.select_dtypes(include=['number']).columns

    # Normalizing the columns with numeric values
    dataset[columnsWithNumericValues] = minMaxScaler.fit_transform(dataset[columnsWithNumericValues])
    # Displaying the data set after being normalized
    print("Numeric Columns after Normalization:")
    print(dataset.select_dtypes(include=['number']))

    return dataset

def separatingDataSet(dataset):
    features = dataset[['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']]
    targets = dataset['Rain']
    print("features columns:")
    print(features)
    print("Target columns:")
    print(targets)
    return features, targets

def splittingandScaling(dataset):
    x, y = separatingDataSet(dataset)
    featuresTrain, featuresTest, targetTrain, targetTest = train_test_split(x, y, test_size=0.25, random_state=123)

    #Applying scaling to the training and testing features
    X_train_scaled, X_test_scaled = scalingAfterSplitting(featuresTrain, featuresTest)

    print(f"Training Features(X) Shape: {X_train_scaled.shape}")
    print(f"Testing Features(X) Shape: {X_test_scaled.shape}")
    print(f"Training Target(Y) Shape: {targetTrain.shape}")
    print(f"Testing Target(Y) Shape: {targetTest.shape}")

    return featuresTrain, featuresTest, targetTrain, targetTest

#scaling numeric features after splitting the data
def scalingAfterSplitting(X_train, X_test):

    # Separate numeric columns for scaling
    numeric_columns = X_train.select_dtypes(include=['number']).columns

    # Initialize the MinMaxScaler
    minMaxScaler = MinMaxScaler()

    # Only scale the numeric columns
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_columns] = minMaxScaler.fit_transform(X_train[numeric_columns])
    X_test_scaled[numeric_columns] = minMaxScaler.transform(X_test[numeric_columns])

    return X_train_scaled, X_test_scaled

def encodeTarget(dataset):
    label_encoder = LabelEncoder()
    dataset['Rain'] = label_encoder.fit_transform(dataset['Rain'])
    return dataset

def distance(x,y):
    result = np.sqrt(np.sum((x - y) ** 2))
    return result


def kNN(featuresTrain, targetTrain, featuresTest, k):
    y_predicts=[]

    for testingPoint in featuresTest.values: # looping over every point(row) in the features Test set
        distances_from_the_point = [] # we will store the distance between the testing point and each other point in this array
        # now each testing data has its own array of distances

        # Calculating the distance from test point to all training points
        for trainingPoint in featuresTrain.values:
            distances_from_the_point.append(distance(testingPoint, trainingPoint))

        distances_after_sorting = np.argsort(distances_from_the_point) # sorting the distances
        distances_after_sorting = distances_after_sorting[:k] # selecting K of these distances

        # getting the target value of the nearest k neighbours
        knn_target_value = [targetTrain.iloc[i] for i in distances_after_sorting]

        # Majority vote: the most common class label among the k neighbors
        most_common = Counter(knn_target_value).most_common(1)[0][0]

        y_predicts.append(most_common)  # Append the prediction for this test point

    return np.array(y_predicts)


def evaluateKNN(Y_test,arrayofpredictedvals):
     y_pred = arrayofpredictedvals
     accuracy = accuracy_score(Y_test, y_pred)
     precision = precision_score(Y_test, y_pred)
     recall = recall_score(Y_test, y_pred)
     print(
         f"\nKNN model Performance:\nAccuracy: {accuracy * 100 :.2f}%\nPrecision: {precision * 100 :.2f}%\nRecall: {recall * 100 :.2f}%")
     return accuracy, precision, recall

#########################################################################################################################################

# Training the Decision Tree classifier
def trainDecTree(X_train, Y_train):
    tree_classifier = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    tree_classifier.fit(X_train, Y_train)
    return tree_classifier

# Evaluate the Decision Tree classifier
def evaluateTree(X_test, Y_test, tree_classifier):
    y_pred = tree_classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    return accuracy, precision, recall

# Plot the Decision Tree
def plotTree(tree_classifier, X_train):
    plt.figure(figsize=(30, 15))
    plot_tree(
        tree_classifier,
        feature_names=X_train.columns,
        class_names=['No Rain', 'Rain'],
        filled=True,
        rounded=True,
        fontsize=20
    )
    plt.title("Decision Tree plot")
    plt.show()

# Train the Naive Bayes classifier
def trainNaiveBayes(X_train, y_train):
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    return nb_classifier

# Evaluate the Naive Bayes classifier
def evaluateNB(X_test, Y_test, nb_classifier):
    y_pred = nb_classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    return accuracy, precision, recall

# KNN built-in function
def knnBuiltInImplementation(X_train, X_test, y_train, k=5):

    # Initialize kNN classifier value with the passed parameter as 5 as a default value
    knn = KNeighborsClassifier(n_neighbors=k)

    # Calls built-in function that takes the features from the dataset and trains it
    knn.fit(X_train, y_train)

    # Calls built-in function that predicts on test data
    y_predict = knn.predict(X_test)

    return y_predict , knn


# # Evaluates the performance based on the provided metrics based on the assignment(metrics accuracy , precision , recall)
# def evaluateMetricsForKNN(y_test, y_pred):
#     #Calls built-in functions that takes the predicted and the actual values of the Rain (target feature)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, pos_label=1)
#     recall = recall_score(y_test, y_pred, pos_label=1)
#
#     #Prints all the values of the metrics
#     print("\nPerformance Metrics on KNN:")
#     print(f"Accuracy: {accuracy:.2f}")
#     print(f"Precision: {precision:.2f}")
#     print(f"Recall: {recall:.2f}")
#     return accuracy, precision, recall

#Compares the values for the Implemented knn and the one using the Scikit-learn library
def plot_comparison_bar(metrics, custom_scores, sklearn_scores):
    x = np.arange(len(metrics))
    width = 0.35  # Bars width

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, custom_scores, width, label='Custom kNN', color='blue')
    ax.bar(x + width / 2, sklearn_scores, width, label='Scikit-learn kNN', color='green')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Custom kNN and Scikit-learn kNN')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    plt.show()

#############################################################################################################################


def main():
    # Loading the dataset
    dataset = loadingDataSet("C:/Users/Eng-Wael/PycharmProjects/MLAssig2/weather_forecast_data.csv")

    checkingForMissingValues(dataset)

    #no_missing_values_dataset = replacingMissingValuesWithAverage(dataset)
    no_missing_values_dataset = dropingMissingData(dataset)

    encoded_dataset = encodeTarget(no_missing_values_dataset)

    scaled_dataset = scalingData(encoded_dataset)

    # Splitting the data into training and testing sets
    featuresTrain, featuresTest, targetTrain, targetTest = splittingandScaling(scaled_dataset)


    #########################################

    print("\n--- Decision Tree ---")
    tree_classifier = trainDecTree(featuresTrain, targetTrain)
    dt_accuracy, dt_precision, dt_recall = evaluateTree(featuresTest, targetTest, tree_classifier)
    print(
        f"Decision Tree Performance:\nAccuracy: {dt_accuracy * 100 :.2f}%\nPrecision: {dt_precision * 100 :.2f}%\nRecall: {dt_recall * 100 :.2f}%")
    plotTree(tree_classifier, featuresTrain)

    print("\n--- Naive Bayes ---")
    nb_classifier = trainNaiveBayes(featuresTrain, targetTrain)
    nb_accuracy, nb_precision, nb_recall = evaluateNB(featuresTest, targetTest, nb_classifier)
    print(
        f"Naive Bayes Performance:\nAccuracy: {nb_accuracy * 100 :.2f}%\nPrecision: {nb_precision * 100 :.2f}%\nRecall: {nb_recall * 100 :.2f}%")

    print("\n--- KNN from Scratch ---")
    kNN_predections = kNN(featuresTrain, targetTrain, featuresTest, 5)

    #evaluateKNN(targetTest, kNN_predections)
    knn_accuracy, knn_precision, knn_recall = evaluateKNN(targetTest, kNN_predections)

    # Perform kNN
    k = 5
    y_pred, knn_model = knnBuiltInImplementation(featuresTrain, featuresTest, targetTrain, k)

    print("\n--- Built-in KNN ---")
    # Evaluation of the KNN
    #evaluateMetricsForKNN(targetTest, y_pred)
    knn_b_accuracy, knn_b_precision, knn_b_recall = evaluateKNN(targetTest, y_pred)

    # # Plot the comparison
    # metrics = ['Accuracy', 'Precision', 'Recall']
    # custom_scores = [knn_accuracy, knn_precision, knn_recall]
    # sklearn_scores = [knn_b_accuracy, knn_b_precision, knn_b_recall]  # You can adjust this for other classifiers as well
    # plot_comparison_bar(metrics, custom_scores, sklearn_scores)




# Run the program
if __name__ == "__main__":
    main()
