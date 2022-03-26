import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

data = pd.read_csv('iris.csv')
X = np.array(data.iloc[:,0:-1])
y = np.array(data.iloc[:,-1])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
ypred = model.predict(X_test)
print("Available Data : ",len(data))
print("Trained Data:  ",len(X_train))
print("Tested data :",len(X_test))
print("Instance\tTest\tPrediction\t")
for i in range(len(X_test)):
    print(i,'\t\t',ypred[i],'\t',y_test[i])
print("Accuracy :",str(round(accuracy_score(y_test,ypred)*100,2)) ,' %')
print("Confusion Matrix :",confusion_matrix(y_test,ypred))
