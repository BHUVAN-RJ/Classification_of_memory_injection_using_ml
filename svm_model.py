import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings as wrn
wrn.filterwarnings('ignore')
from matplotlib import pyplot as plt
# Read the data from CSV file and explicitly name the columns
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

from sklearn.svm import SVC
model_SVM = SVC(C=10,kernel='sigmoid')
model_SVM.fit(X_train,Y_train)

from sklearn import metrics

print("SVM MODEL EVALUATION")

# Assuming you have trained your model and obtained the predicted labels
y_pred = model_SVM.predict(X_test)

# Compute accuracy
accuracy = metrics.accuracy_score(Y_test, y_pred)

# Compute precision, recall, and F1-score
precision = metrics.precision_score(Y_test, y_pred)
recall = metrics.recall_score(Y_test, y_pred)
f1_score = metrics.f1_score(Y_test, y_pred)


# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
print(metrics.confusion_matrix(Y_test, y_pred))
print("\n\n********************************************\n\n")


# y_pred = model_SVM.predict(X_test)
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(Y_test, y_pred)
# import seaborn as sns
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()



from sklearn.ensemble import RandomForestClassifier
model_Random = RandomForestClassifier(criterion='gini', max_depth=8, min_samples_split=4,random_state=5)
model_Random.fit(X_train,Y_train)

print("Random forest MODEL EVALUATION")

# Assuming you have trained your model and obtained the predicted labels
y_pred = model_Random.predict(X_test)

# Compute accuracy
accuracy = metrics.accuracy_score(Y_test, y_pred)

# Compute precision, recall, and F1-score
precision = metrics.precision_score(Y_test, y_pred)
recall = metrics.recall_score(Y_test, y_pred)
f1_score = metrics.f1_score(Y_test, y_pred)



# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
print(metrics.confusion_matrix(Y_test, y_pred))

newdata =pd.read_csv('new_infected_data.csv', names=['Process Name','Process ID', 'Memory Usage (MB)', 'Percent of Loaded DLLs'], index_col=False,header=0)
newdata_X = newdata.drop(['Process ID', 'Process Name'], axis='columns')
newdata_pred = model_Random.predict(newdata_X)
print(newdata_pred)

# from sklearn.model_selection import learning_curve
# train_sizes, train_scores, test_scores = learning_curve(
#     model_SVM, X, Y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
# )
#
# # Calculate mean and standard deviation of train and test scores
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
#
# # Plot the learning curve
# plt.figure(figsize=(8, 6))
# plt.plot(train_sizes, train_mean, label='Training Accuracy')
# plt.plot(train_sizes, test_mean, label='Validation Accuracy')
# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
# plt.xlabel('Training Set Size')
# plt.ylabel('Accuracy')
# plt.title('Learning Curve')
# plt.legend(loc='best')
# plt.show()

# graphs
# train_sizes, train_scores, test_scores = learning_curve(
#     model_Random, X, Y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
# )
#
# # Calculate mean and standard deviation of train and test scores
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)

# Plot the learning curve
# plt.figure(figsize=(8, 6))
# plt.plot(train_sizes, train_mean, label='Training Accuracy')
# plt.plot(train_sizes, test_mean, label='Validation Accuracy')
# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
# plt.xlabel('Training Set Size')
# plt.ylabel('Accuracy')
# plt.title('Learning Curve')
# plt.legend(loc='best')
# plt.show()
#
# features = ['Process name', 'Memory Usage', '% of DLL loaded']
# print(df.columns)
# importance = model_Random.feature_importances_
# print(importance)
# indices = np.argsort(importance)
# plt.title('Feature Importance')
# plt.barh(range(len(indices)), importance[indices], color='b')
# print(indices)
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel("Relative Importance")
# plt.show()
#
