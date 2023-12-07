import pandas as pd
from sklearn.model_selection import train_test_split
import warnings as wrn
wrn.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, OneHotEncoder




df = pd.read_csv('processed_dataset.csv', names=['Process Name','Process ID', 'Memory Usage (MB)', 'Percent of Loaded DLLs', 'Injection','Empty'], index_col=False,header=0)
df = df.drop(['Empty'],axis='columns')
process_labels, unique_process = pd.factorize(df['Process Name'])
df['Process Name'] = process_labels

X = df.drop(['Process Name','Process ID', 'Injection'],axis='columns')
Y = df['Injection']

positives = df[df["Injection"] == True]
negatives = df[df["Injection"] == False]

positives["Injection"] = 1
negatives["Injection"] = 0

positive_X = positives.drop(['Process Name','Process ID', 'Injection'],axis='columns')
positive_Y = positives['Injection']

negative_X = negatives.drop(['Process Name','Process ID', 'Injection'],axis='columns')
negative_Y = negatives['Injection']

positive_X_train,positive_X_test,positive_Y_train,positive_Y_test = train_test_split(positive_X,positive_Y,test_size=0.2)
negative_X_train,negative_X_test,negative_Y_train,negative_Y_test = train_test_split(negative_X,negative_Y,test_size=0.2)

X_train = pd.concat([positive_X_train, negative_X_train])
X_test = pd.concat([positive_X_test,negative_X_test,positive_X_train])
Y_train = pd.concat([positive_Y_train,negative_Y_train])
Y_test = pd.concat([positive_Y_test,negative_Y_test,positive_Y_train])

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)  # Specify the number of neighbors

# Train the classifier
knn.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)
from sklearn.metrics import confusion_matrix
from sklearn import metrics
# Calculate and display the confusion matrix
cm = confusion_matrix(Y_test, y_pred)
accuracy = metrics.accuracy_score(Y_test, y_pred)

# Compute precision, recall, and F1-score
precision = metrics.precision_score(Y_test, y_pred)
recall = metrics.recall_score(Y_test, y_pred)
f1_score = metrics.f1_score(Y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
print(cm)

newdata =pd.read_csv('new_infected_data.csv', names=['Process Name','Process ID', 'Memory Usage (MB)', 'Percent of Loaded DLLs'], index_col=False,header=0)
newdata_X = newdata.drop(['Process ID', 'Process Name'], axis='columns')
newdata_pred = knn.predict(newdata_X)
print(newdata_pred)


# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.model_selection import learning_curve
# train_sizes, train_scores, test_scores = learning_curve(knn, X, Y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))
#
# # Calculate the mean and standard deviation of train and test scores
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
#
# # Plot the learning curve
# plt.figure(figsize=(8, 6))
# plt.plot(train_sizes, train_mean, label='Training score', color='blue')
# plt.plot(train_sizes, test_mean, label='Cross-validation score', color='red')
#
# # Plot the shaded area indicating the standard deviation
# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='gray', alpha=0.3)
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='pink', alpha=0.3)
#
# # Add labels and title
# plt.xlabel('Number of Training Samples')
# plt.ylabel('Accuracy Score')
# plt.title('Learning Curve for KNN')
# plt.legend(loc='best')
#
# # Show the plot
# plt.show()