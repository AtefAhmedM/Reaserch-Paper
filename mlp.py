import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
# Loading the data + plotting and visualizations :
data_frame = pd.read_csv('breast_cancer_data.csv')  

print(data_frame.head(10))
#Get a count of the number of malignant(1) and benign(0) cells
for i in range(data_frame.shape[0]):
    if data_frame['diagnosis'][i] == 'M':
        data_frame['diagnosis'][i] = 0
    else:
        data_frame['diagnosis'][i] = 1
#Split the dataset into independent(X) and dependent(Y) datasets
X = data_frame.iloc[:,1:34].values
Y = data_frame.iloc[:,0].values
#Visualize the count
sns.countplot(data_frame['diagnosis'], label = 'count')
#Create a pair plot
sns.pairplot(data_frame.iloc[: ,0:6], hue = 'diagnosis')
#Visualize the correlation
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data_frame.corr(), annot = True, fmt= '.2f')
# The correlation of the columns
print(data_frame.corr())
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

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, Y_train)
print('MLP Training Accuracy : ' , clf.score(X_train, Y_train))
print('MLP Testing Accuracy : ' , clf.score(X_test, Y_test)) 

print('Model :',clf)
cm = confusion_matrix(Y_test, clf.predict(X_test))
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
print(cm)
print('Testing Accuracy =',(TP + TN)/(TP + FP + FN + TN))
print()

print('Model :', clf)
print(classification_report(Y_test, clf.predict(X_test)))
print(accuracy_score(Y_test, clf.predict(X_test)))
print()

