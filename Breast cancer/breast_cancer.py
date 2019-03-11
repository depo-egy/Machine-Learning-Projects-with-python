'''
In this Machine Learning Project , we have a classification case in which we will work in categorical data.
Our data is the previously saved data in the sklearn library "breast cancer". We will try to classify the cells
to be either malignant or benign.

These cells have some features like: radius , texture , perimeter , smoothness , concavity , .... , and so on.

We will try to extract the most important features the contribute for the model.

'''

# Let's start


# First we will import the required libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


#So cancer is the variable that holds the dataset

#Now we will try to explore this dataset , the following commands can be show using python console

cancer.keys()
cancer.feature_names
cancer['data'].shape #output: (569, 30) ==> 569 rows:observations , 30 coulmns: features
cancer.target # output is 1 or 0 ==> 1:malignat , 0:benign
cancer.target_names # output is malignant or benign

cancer.DESCR # data description

#Now we will convert this dataset into dataframe , so we can make data preprocessing
df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns = np.append(cancer['feature_names'] ,['target']))

df_cancer.head() #show the first 5 rows

#bivariate analysis with visualization
sns.pairplot(df_cancer ,hue = 'target' ,vars = ['mean radius' , 'mean texture' , 'mean perimeter' ,'mean area' ,'mean smoothness'
                               ])

#Check the correlation between variables
plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True) 


#count each target
sns.countplot(df_cancer['target'])

#divide the dataset into features and target for implementing the model:

X = df_cancer.iloc[: , 0:30] # the columns are the features , take 30 features from 0:29 , the 30th column is execluded
X.head()
y = df_cancer.iloc[: , -1]

#split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)

X_train.shape #output==> (455, 30) , observe that the 455 is 80% of 569 which is the training set
X_test.shape #output==> (114,30)

#Implementing the model

from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, y_train)

#Evaluating the model

y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict))

#Improving the model

min_train = X_train.min()
min_train

range_train = (X_train - min_train).max()
range_train

X_train_scaled = (X_train - min_train)/range_train #Scaling the training set


#scatterplot with nonscaled training set data and with scaled training set data
sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test

from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)

y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm,annot=True,fmt="d")

print(classification_report(y_test,y_predict))

#Improving the model by hyperparameter tuning

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)

print(classification_report(y_test,grid_predictions))

#Finished
sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train) 



y = df_cancer.iloc[: , -1] # y : the last column the target with 0 for bengin and 1 for malignant


