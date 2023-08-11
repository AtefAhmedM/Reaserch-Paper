import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score

# Loading the data + plotting and visualizations :
data_frame = pd.read_csv('breast_cancer_data.csv')  
#Get a count of the number of malignant(1) and benign(0) cells
for i in range(data_frame.shape[0]):
    if data_frame['diagnosis'][i] == 'M':
        data_frame['diagnosis'][i] = 0
    else:
        data_frame['diagnosis'][i] = 1
#Split the dataset into independent(X) and dependent(Y) datasets
X = data_frame.iloc[:,1:34].values
Y = data_frame.iloc[:,0].values
#Split the dataset into 75% training and 25% testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
# Convert the data to float
X_train = X_train.astype(float)
X_test = X_test.astype(float)
Y_train = Y_train.astype(float)
Y_test = Y_test.astype(float)
print(Y_train)
#Scale the data (Feature Scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
# Decision trees classifier
tree = DecisionTreeClassifier()
tree.fit(X_train, Y_train)
print('Tree training accuracy : ', tree.score(X_train, Y_train))
print('Tree Testing accuracy : ', tree.score(X_test, Y_test))

print('Model :', tree)
cm = confusion_matrix(Y_test,tree.predict(X_test))
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
print(cm)
print('Testing Accuracy =',(TP + TN)/(TP + FP + FN + TN))

print('Model :', tree)
print(classification_report(Y_test, tree.predict(X_test)))
print(accuracy_score(Y_test, tree.predict(X_test)))
print()

