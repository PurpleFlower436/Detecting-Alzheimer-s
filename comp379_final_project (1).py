# -*- coding: utf-8 -*-
"""COMP379_Final_Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OylECnTiyL1TMAjksaCR1ThBVCccvl-M
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#For modeling and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression ### IMPORT OTHER MODELS THAT WILL BE TESTED IN THE PROJECT
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

print('The dataset was obtained from Kaggle site. It contains Gender(M/F), Age	Educations(EDUC), SocioEconomic Status(SES), Mini Mental State Exam(MMSE), Clinical Dementia Rating(CDR), Estimated Total Intercranial Volume (ETIV), normalized Whole Brain Volume (nWBV), and Atlas Scaling Factor(ASF) as features')
print('The dataset contains Group as the target variable with three values: Demented, NonDemented and Converted')
      
      
df=pd.read_csv('alzheimer.csv')
print(df.head())

#Find the number of rows and columns in the data
print(df.shape)

# test for null values in the dataset
print(df.isnull().sum())

"""Based on the above, SES has 19 missing values and MMSE has two missing values.  Since the number of missing values is not that large, rather than dropping these features, we will drop samples where these values are missing."""

df.head()

df1 = df.dropna()
df1.head()
df1.shape

#Are there any duplicated values in the dataset
print(df.duplicated().sum())

# Get the frequency of different values in the target variable Group

print(df1['Group'].value_counts())

# Based on above there are three different values Nondemented, Demented and Converted.  To keep it a binary target variable,
# we drop 'Converted' samples since it occurs much less frequently than Nondemented and Demented.

df2=df1.drop(df1[df1["Group"]=="Converted"].index)
print(df2['Group'].value_counts())

# Convert Group and M/F columns from string to integer values so that these can be used for modeling
# Nondemented = 0, Demented = 1
# M = 0, F = 1

df2['Group'].replace(['Nondemented', 'Demented'],[0, 1], inplace=True)
df2['M/F'].replace(['M', 'F'],[0, 1], inplace=True)
print(df2.head())

# Define the Features and Target variables.  The first column is the target variable. Call it y.
# Remaining 9 columns are explanatory variables or Features.  Collectively call them X.

X = df2.iloc[:, 1:10].values
print(X)
y = df2.iloc[:, 0].values
print(y)

"""# Split the dataset into trainingand test in 70/15/15 proportion.


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)

# print size to ensure that split occurred as desired
print(y.size, y_train.size, y_test.size)

"""

# Split the dataset into training, testing and validation in the ratio of 70/15/15.
# 1: Divide the full data into training (70%) and remaining (30%)
# 2: Now divide the remaining into test (50%) and validation (50%). Since remaining contained 30% of full data,
# splitting into two equal halfs will assign 15% of full data to both test and validation.

X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size = 0.3, random_state = 0, stratify = y)
X_test, X_valid, y_test, y_valid = train_test_split(X_remaining, y_remaining, test_size = 0.5, random_state = 0, stratify = y_remaining)

# print size to ensure that split occurred as desired
print(y.size, y_train.size, y_test.size, y_valid.size)

# Scale the features matrix X since different features have widely different values.  Not scaling can lead
# to some features having an unintended influence on the model predictions.

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled  = sc.transform(X_test)
X_valid_scaled  = sc.transform(X_valid)

# Using LOGISTIC REGRESSION

Logistic = LogisticRegression(random_state = 0)


Logistic.fit(X_train_scaled, y_train)
y_pred = Logistic.predict(X_valid_scaled) 

cmLogistic = confusion_matrix(y_valid, y_pred)
print(cmLogistic)
Accuracy_Logistic_Valid = accuracy_score(y_valid, y_pred)

print ("Accuracy Score using Logistic Regression on the validation dataset is %f" %Accuracy_Logistic_Valid)
F1_Logistic_Valid = f1_score(y_valid, y_pred)
print("F1 score using Logistic Regression on the validation dataset is %f" %F1_Logistic_Valid)


Logistic.fit(X_test_scaled, y_test)
y_pred = Logistic.predict(X_test_scaled) 
Accuracy_Logistic_Test = accuracy_score(y_test, y_pred)
print ("Accuracy Score using Logistic Regression on the test dataset is %f" %Accuracy_Logistic_Test)
F1_Logistic_Test = f1_score(y_test, y_pred)
print("F1 score using Logistic Regression on the test dataset is %f" %F1_Logistic_Test)

# Confusion matrix above shows that all Demented and Nondemented patients are correctly predicted by the Logistic model
# as there are no values on the upward sloping diagonal.  All predictions lie on the downward sloping diagonal.  This is also
# indicated by the perfect accuracy score of 1.0.

print("KNN implementation:")
#using KNN with best parameters a.k.a 1 neighbor and weights being uniform
from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors=1, weights='uniform')
knn.fit(X_train_scaled,y_train)
y_predictions = knn.predict(X_valid_scaled)


knn_confustion_matrix = confusion_matrix(y_valid, y_predictions)
print(knn_confustion_matrix)
Accuracyknn = accuracy_score(y_valid, y_predictions)

f1_score_best_model_valid = f1_score(y_valid, y_predictions)


print("Accuracy Score using KNN algorithm on validation set is %f" %Accuracyknn)
print("")
print("F1 score using KNN algorithm on validation set is %f" %f1_score_best_model_valid)



y_predictions_test_set = knn.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predictions_test_set)
print("")
print(cm)
Accuracyknn_test = accuracy_score(y_test, y_predictions_test_set)
f1_score_best_model_test = f1_score(y_test, y_predictions_test_set)

print("Accuracy Score using KNN algorithm on test set is %f" %Accuracyknn_test)
print("F1 score using KNN algorithm on test set is %f" %f1_score_best_model_test)

print("knn using grid search")
#testing for different k values using grid search 
hyperparameters = [1,2,3,4,5,6,7,8,9,10, 11, 12, 13]

for j in hyperparameters:


#using KNN with best parameters a.k.a 1 neighbor and weights being uniform


  knn = KNeighborsClassifier(n_neighbors=j, weights='uniform')
  knn.fit(X_train_scaled,y_train)
  y_predictions = knn.predict(X_valid_scaled)


  knn_confustion_matrix = confusion_matrix(y_valid, y_predictions)
  Accuracyknn = accuracy_score(y_valid, y_predictions)

  f1_score_best_model_valid = f1_score(y_valid, y_predictions)


  print("The k value is k = %d and accuracy score on validation set is  %f"  %(j, Accuracyknn))
  print("")
  print("The k value is k = %d and f1 score on validation set is %f"  %(j, f1_score_best_model_valid))
  print(knn_confustion_matrix)
  print("")


  y_predictions_test_set = knn.predict(X_test_scaled)
  cm = confusion_matrix(y_test, y_predictions_test_set)
  print("")
  Accuracyknn_test = accuracy_score(y_test, y_predictions_test_set)
  f1_score_best_model_test = f1_score(y_test, y_predictions_test_set)

  print("The k value is k = %d and accuracy score on test set is  %f"  %(j, Accuracyknn_test)) #test set
  print("The k value is k = %d and f1 score on test set is %f"  %(j, f1_score_best_model_test))

# Random Forest Implementation

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('alzheimer.csv')
print(df.head(5))

df = df.drop(df[df["Group"] == "Converted"].index)
class_le = LabelEncoder()
y = class_le.fit_transform(df['Group'].values)
df.iloc[:, 0] = y
y = class_le.fit_transform(df['M/F'].values)
df.iloc[:, 1] = y

df = df.dropna(axis=0)

X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)
X_test, X_valid, y_test, y_valid = train_test_split(
    X_remaining, y_remaining, test_size=0.5, random_state=0, stratify=y_remaining)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


feat_labels = df.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1)

forest.fit(X_train, y_train)
y_pred = forest.predict(X_valid)
Accuracy_Random_Forest_Valid = accuracy_score(y_valid, y_pred)
print ("Accuracy Score using Random Forest on the validation dataset is %f" %Accuracy_Random_Forest_Valid)
F1_Random_Forest_Valid = f1_score(y_valid, y_pred)
print("F1 score using Random Forest on the validation dataset is %f" %F1_Random_Forest_Valid)


forest.fit(X_test, y_test)
y_pred1 = forest.predict(X_test)
Accuracy_Random_Forest_Test = accuracy_score(y_test, y_pred1) 
print ("Accuracy Score using Random Forest on the test dataset is %f" %Accuracy_Random_Forest_Test)
F1_Random_Forest_Test = f1_score(y_test, y_pred1)
print("F1 score using Random Forest on the test dataset is %f" %F1_Random_Forest_Test)


importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d %-*s %f" %
          (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')

plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])

plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score


df1 = df.dropna()
df2=df1.drop(df1[df1["Group"]=="Converted"].index)
df2['Group'].replace(['Nondemented', 'Demented'],[0, 1], inplace=True)
df2['M/F'].replace(['M', 'F'],[0, 1], inplace=True)
X = df2.iloc[:, 1:10].values
#print(X)
y = df2.iloc[:, 0].values
X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size = 0.3, random_state = 0, stratify = y)
X_test, X_valid, y_test, y_valid = train_test_split(X_remaining, y_remaining, test_size = 0.5, random_state = 0, stratify = y_remaining)


# split the data into training and testing sets

# standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)


def gridsearch():
    activation1 = ['relu', 'sigmoid', 'softmax']
    activation2 = ['relu', 'sigmoid', 'softmax']
    activation3 = ['relu', 'sigmoid', 'softmax']
    optimizer = ['adam', 'rmsprop']
    loss = ['categorical_cross', 'binary_crossentropy']
   
    hyper_param_arr = []
    for i in activation1:
      for j in activation2:
        for k in activation3:
          for l in optimizer:
            for m in loss:

              model = Sequential()
              model.add(Dense(64, activation=i, input_dim=X_train.shape[1]))
              model.add(Dense(64, activation=j))
              model.add(Dense(1, activation=k))
              model.compile(optimizer=l, loss='binary_crossentropy', metrics=['accuracy'])
              model.fit(X_train, y_train, epochs=10, batch_size=32)
              score = model.evaluate(X_valid, y_valid, batch_size=32)
              hyper_param_arr.append((score, i, j,k,l,m))
    hyper_param_arr.sort(key=lambda a: a[0])
    print(hyper_param_arr)   

gridsearch()

from sklearn.metrics import accuracy_score, f1_score
X_test = scaler.transform(X_test)

#NN model definition
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)

# evaluate the model
score = model.evaluate(X_test, y_test, batch_size=32)
print(score)

#baseline model using Sklearn dummy classifier
dummy_classifier = DummyClassifier(strategy="stratified")
dummy_classifier.fit(X_train,y_train)
label_predicted_by_model = dummy_classifier.predict(X_test)
accuracy_dummy = accuracy_score(y_test, label_predicted_by_model)
print(f'The accuracy of dummy model is: {accuracy_dummy*100}')