import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

data_app = pd.read_csv(
    "D:/application_train.csv",
    sep=",",
    encoding="cp1251")

data_bureau = pd.read_csv(
    "D:/bureau.csv",
    sep=",",
    encoding="cp1251")

data_bureau_join = data_bureau.drop(['SK_ID_BUREAU', "CREDIT_CURRENCY", "CREDIT_ACTIVE", "CREDIT_TYPE"], axis=1)
data_app = data_app.drop(['NAME_CONTRACT_TYPE'], axis=1)
data_bureau_join = data_bureau_join.groupby(by='SK_ID_CURR').mean()
joined_data = pd.merge(data_app, data_bureau_join, on='SK_ID_CURR', how='left')
target = joined_data["TARGET"]
X = joined_data.drop(columns=["TARGET", 'SK_ID_CURR'])
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3)

regr = XGBClassifier()
regr.fit(X_train, y_train)
y_hat = regr.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_hat, pos_label=1)
auc = metrics.auc(fpr, tpr)
print(auc)
