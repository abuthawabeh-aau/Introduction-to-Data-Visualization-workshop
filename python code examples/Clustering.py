#Loading dataset 

import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets 


#splitting dataset(train, test)
iris = datasets.load_iris()
iris.data 
iris.feature_names 
iris.target 
iris.target_names 

x=pd.DataFrame(iris.data) 

x.columns=['Sepal_Length','Sepal_width','Petal_Length','Petal_width'] 

y=pd.DataFrame(iris.target) 
y.columns=['Targets'] 

#train model
model=KMeans(n_clusters=3)
model.fit(x)



#test model
plt.scatter(x.Petal_Length, x.Petal_width)


colormap=np.array(['Red','green','blue'])

plt.scatter(x.Petal_Length, x.Petal_width,c=colormap[model.labels_],s=40)
plt.title('Classification K-means ')