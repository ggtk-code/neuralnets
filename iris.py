# Run various classifiers on iris dataset
import pandas as pd
import os
import perceptron as pt
import adaline as ad
import numpy as np

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(path, header = None, encoding = 'utf-8')
print(df)

# form vectors y and X
# y is 2 class labels -1, +1
y = df.iloc[0:150, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
#print(y)

X = df.iloc[0:150, [0, 1, 2, 3]].values
#print(X)

# Now let's fit a perceptron
# print("===================perceptron====================")
# p = pt.Perceptron(4, 0.1) 
# p.Fit(X, y)
# # print the learned model
# print("=========training done=============")
# p.PrintModel()
# p.Accuracy(X, y)

# Lets fit an adaline
print("\n\n===================adaline====================")
a = ad.Adaline(5, 0.0001, std_features = True) 
a.Fit(X, y)
# print the learned model
print("=========training done=============")
a.PrintModel("final model")
a.Accuracy(X, y)
