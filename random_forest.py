import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
#plt.style.use('dark_background')

# Plotting and visualizations :
data_frame = pd.read_csv('breast_cancer_data.csv')  
sns.countplot(data_frame['diagnosis'], label = 'count')
sns.pairplot(data_frame.iloc[: ,0:6], hue = 'diagnosis')
plt.show()


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

print(Y_train)


X_train = X_train.astype(float)
X_test = X_test.astype(float)
Y_train = Y_train.astype(float)
Y_test = Y_test.astype(float)

#Scale the data (Feature Scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, Y_train)


print('Random Forest Training Accuracy : ' , model.score(X_train, Y_train))
print('Random Forest Training Accuracy : ' , model.score(X_test, Y_test))
