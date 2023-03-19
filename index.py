import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

diabetes_dataset = pd.read_csv('/content/diabetes.csv')

diabetes_dataset.head()

diabetes_dataset.shape

diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

X=diabetes_dataset.drop(columns= 'Outcome',axis=1)
Y=diabetes_dataset['Outcome']

print(X)

print(Y)

scaler = StandardScaler()

scaler.fit(X)

standardized_data=scaler.transform(X)

print(standardized_data)

X= standardized_data
Y=diabetes_dataset['Outcome']

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

print("Accuracy Score of Training Data : ", training_data_accuracy)

X_train_prediction = classifier.predict(X_test)
training_data_accuracy = accuracy_score(X_train_prediction,Y_test)

print("Accuracy Score of Test Data : ", training_data_accuracy)

input_data=(4,110,92,0,0,37.6,0.191,30)
#changing the data to numpy Array
input_data_as_numpy_array=np.asarray(input_data)
#Reshaping the Array
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

#Standardizing the Data
std_data=scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if (prediction[0]==0):
  print("Person is not Diabetic")
else:
  print("Person is Diabetic")