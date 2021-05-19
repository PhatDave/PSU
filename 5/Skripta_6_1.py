import numpy as np
from sklearn.datasets import fetch_openml
import joblib
import pickle


X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)


# TODO: prikazi nekoliko ulaznih slika


# skaliraj podatke, train/test split
X = X / 255.
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


# TODO: izgradite vlastitu mrezu pomocu sckitlearn MPLClassifier 


# TODO: evaluirajte izgradenu mrezu


# spremi mrezu na disk
filename = "NN_model.sav"
joblib.dump(mlp_mnist, filename)

