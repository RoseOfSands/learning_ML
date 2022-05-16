from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


"""считывание данных для обучения"""
data = pd.read_csv("E:\file", sep="\t", encoding="cp1251")

"""выбор исходных и целевых параметров для модели"""
X = data[
    [
        "X1",
        "X2",
        "X3",
        "X4",
        "X5",
        "X6",
        "X7",
        "X8",
        "X9",
        "X10",
        "X11",
    ]
]
y = data[["Y"]]
y = np.ravel(y)

"""запуск лучшей модели с соответствующими гиперпараметрами, подобранными с помощью GridSearchCV"""
xgbr = XGBRegressor(learning_rate=0.13, n_estimators=150, max_depth=8)

"""обучение модели"""
xgbr.fit(X, y)

"""считывание данных получения предсказаний"""
dfa = pd.read_csv("E:\file2.txt", sep="\t", encoding="cp1251")
y_hat = xgbr.predict(dfa)

"""сглаживание данных"""
pol = np.polyfit(dfa["X1"], y_hat, 6)
y_pol = np.polyval(pol, dfa["X1"])

"""подсчёт критерия, указывающего, насколько близка полученная зависимость к полиномиальной"""
criterion = np.sum((y_hat - y_pol) ** 2)
print(criterion)

"""вывод результата в указанный файл"""
y_hat = pd.DataFrame(y_hat)
y_hat.insert(1, "полином", np.polyval(pol, dfa["X1"]))
y_hat.insert(0, "X1", dfa["X1"])
y_hat.to_csv(
    "E:\file_predicted",
    sep="\t",
    encoding="cp1251",
    index=False,
)
