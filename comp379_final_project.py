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


# sklearn's train_test_split is designed to split data into 2 parts only.  
# Hence we use train_test_split twice:
# 1: Divide the full data into training (70%) and remaining (30%)
# 2: Now divide the remaining into test (50%) and validation (50%).  Since remaining contained 30% of full data,
# splitting into two equal halfs will assign 15% of full data to both test and validation as required in the HW.

X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size = 0.3, random_state = 0, stratify = y)
X_test, X_valid, y_test, y_valid = train_test_split(X_remaining, y_remaining, test_size = 0.5, random_state = 0, stratify = y_remaining)

# print size to ensure that split occurred as desired
print(y.size, y_train.size, y_test.size, y_valid.size)

# We can see that training data has 221 samples (approx 70% of all data samples = 317), and test and validation data each have  
# 48 samples (which is approx 15% of the entire data of 317 samples)

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

# Now test a few other models and compare accuracy scores across different models. Take the best model and test its performance
# on the test sample.



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

print("")
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


  print("accuracy score on validation set The k value is k = %d is %f"  %(j, Accuracyknn))
  #print("Accuracy Score using KNN algorithm on validation set with number of neighbors is %f" %Accuracyknn, %)
  print("")
  print("f1 score on validation set The k value is k = %d is %f"  %(j, f1_score_best_model_valid))
  print(knn_confustion_matrix)
  #print("F1 score using KNN algorithm on validation set is %f" %f1_score_best_model_valid)
  print("")


  y_predictions_test_set = knn.predict(X_test_scaled)
  cm = confusion_matrix(y_test, y_predictions_test_set)
  print("")
  Accuracyknn_test = accuracy_score(y_test, y_predictions_test_set)
  f1_score_best_model_test = f1_score(y_test, y_predictions_test_set)

  print("accuracy score on test setThe k value is k = %d is %f"  %(j, Accuracyknn_test)) #test set
  print("f1 score on test set The k value is k = %d is %f"  %(j, f1_score_best_model_test))

