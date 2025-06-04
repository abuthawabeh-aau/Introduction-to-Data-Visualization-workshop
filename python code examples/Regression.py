#Loading dataset 
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

data = pd.read_csv("data/housing.csv")

data.isnull().sum()

target = 'median_house_value'


features_of_interest = ["total_rooms", "median_income", "housing_median_age", "population"]

#splitting dataset(train, test)

X_train, X_test, y_train, y_test = train_test_split(data[features_of_interest ].values, data[target].values)


#train model
clf = LinearRegression()
clf.fit(X_train, y_train)

#test model
predicted = clf.predict(X_test)
expected = y_test

plt.figure(figsize=(4, 3))
plt.scatter(expected, predicted)
plt.plot([0, 8], [0, 8], "--k")
plt.axis("tight")
plt.xlabel("True price ($100k)")
plt.ylabel("Predicted price ($100k)")
plt.tight_layout()

#Print the error rate

print(f"RMS: {np.sqrt(np.mean((predicted - expected) ** 2))!r} ")

plt.show()