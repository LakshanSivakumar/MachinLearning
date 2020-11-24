import tensorflow
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as mp
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "freetime", "activities"]]
data["activities"] = data[["activities"]].replace("yes", 1)
data["activities"] = data[["activities"]].replace("no", 0)

predict = "G3"
print(data)


X = np.array(data.drop([predict], 1))
y = np.array(data[predict])


best = 0
for _ in range(100000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    if acc > best:
        best = acc
        print(best)
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)


pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    rounded = round(predictions[x])
    print(rounded, x_test[x], y_test[x])


p="failures"
style.use("ggplot")
mp.scatter(data[p], data["studytime"])
mp.xlabel(p)
mp.ylabel("studytime")
mp.show()
