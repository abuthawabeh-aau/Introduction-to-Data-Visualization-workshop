#Loading dataset 

import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt


from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 

df = pd.read_csv("data/heart-disease.csv")
df.head() 


#splitting dataset(train, test)
X = df.drop(labels="target", axis=1)

# Target variable
y = df.target.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X,   y, test_size = 0.2) 

#train model
model =KNeighborsClassifier()
model.fit(X_train, y_train)


#test model
model_scores = model.score(X_test, y_test)


#Print the error rate

print(model_scores)
