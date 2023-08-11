import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#################################################### Importing and clening the data ###############################################

# Importing the df and checking if it was imported correctly. 
df = pd.read_csv (r'./breast_cancer_data.csv')


def turn_diagnosis_into_int(df):
    for element in range(len(df['diagnosis'])):
        if df.at[element,'diagnosis'] == 'M':
            df.at[element,'diagnosis'] = 1
        if df.at[element,'diagnosis'] == 'B':
            df.at[element,'diagnosis'] = 0
    return df
    


# Normalize income, age, population
def normalize_columns(df, columns_to_normalize):
    for col in columns_to_normalize:
        df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())
    return df




# ################################# Run data clean up  ########################################
df = turn_diagnosis_into_int(df)
df = normalize_columns(df, df.columns)

print(df.head())

# ######################################## Train the algorithm ################################################

input_data = df.drop(columns=['diagnosis'])
outcome_data = df['diagnosis']
outcome_data=outcome_data.astype('int')
X_train, X_test, y_train, y_test = train_test_split(input_data, outcome_data, test_size=0.2, random_state=0)
y_train = y_train.astype('int')
y_test = y_test.astype('int')


model = SVC()
model.fit(X_train, y_train)
  
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))



parameter_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000], 
              'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(SVC(), parameter_grid, refit = True, verbose = 3)
  
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)

grid_predictions = grid.predict(X_test)
print(classification_report(y_test, grid_predictions))