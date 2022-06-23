import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data_app = pd.read_csv(
    "/media/natalia/KINGSTON/application_train.csv",
    sep=",",
    encoding="cp1251")

data_bureau = pd.read_csv(
    "/media/natalia/KINGSTON/bureau.csv",
    sep=",",
    encoding="cp1251")

outer_join_data = pd.merge(data_app, data_bureau, on='SK_ID_CURR', how='outer')
target = outer_join_data["TARGET"]
X = outer_join_data.drop(columns=["TARGET"])
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3)

regr = XGBClassifier()
regr.fit(X_train, y_train)
y_hat = regr.predict(X_test)
accuracy = accuracy_score(y_test, y_hat)
print(accuracy*100)
