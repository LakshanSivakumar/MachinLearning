import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pickle

data = pd.read_csv("diabetes_data_upload.csv")
print(data.head())

le = preprocessing.LabelEncoder()

Gender = le.fit_transform(list(data["Gender"]))
Polyuria = le.fit_transform(list(data["Polyuria"]))
Polydipsia = le.fit_transform(list(data["Polydipsia"]))
sudden_weight_loss = le.fit_transform(list(data["sudden weight loss"]))
weakness = le.fit_transform(list(data["weakness"]))
Polyphagia = le.fit_transform(list(data["Polyphagia"]))
Genital_thrush = le.fit_transform(list(data["Genital thrush"]))
visual_blurring = le.fit_transform(list(data["visual blurring"]))
Itching = le.fit_transform(list(data["Itching"]))
Irritability = le.fit_transform(list(data["Irritability"]))
delayed_healing = le.fit_transform(list(data["delayed healing"]))
partial_paresis = le.fit_transform(list(data["partial paresis"]))
muscle_stiffness = le.fit_transform(list(data["muscle stiffness"]))
Alopecia = le.fit_transform(list(data["Alopecia"]))
Obesity = le.fit_transform(list(data["Obesity"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(Gender, Polyuria, Polydipsia,sudden_weight_loss, weakness, Polyphagia,Genital_thrush,visual_blurring,Itching,Irritability,delayed_healing,partial_paresis, muscle_stiffness,Alopecia, Obesity))
y = list(cls)


best = 0
model = KNeighborsClassifier(n_neighbors=1)
for _ in range(10):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(best)
    if acc > best:
        best = acc
        with open("class.pickle", "wb") as f:
            pickle.dump(model, f)

pickle_in = open("class.pickle", "rb")
model = pickle.load(pickle_in)

predictions = model.predict(x_test)

names = ["Yes", "No"]

for p in range(len(predictions)):
    print("Predicted ", names[predictions[p]], "Data: ", x_test[p], "Actual ", names[y_test[p]])



