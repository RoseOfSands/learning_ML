from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


'''считывание данных для обучения модели в виде таблицы'''
data = pd.read_csv("E:\file", sep="\t", encoding="cp1251")

'''выбор исходных и целевых параметров'''
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

'''данные для подбора гиперпараметров'''
pipe = Pipeline([("regressor", XGBRegressor())])
search_space = [
    {
        "regressor": [XGBRegressor()],
        "regressor__learning_rate": np.arange(0.01, 0.15, 0.02),
        "regressor__n_estimators": np.arange(50, 200, 50),
        "regressor__max_depth": np.arange(4, 15, 2),
    },
    {
        "regressor": [LGBMRegressor()],
        "regressor__learning_rate": np.arange(0.01, 0.15, 0.02),
        "regressor__n_estimators": np.arange(50, 200, 50),
        "regressor__max_depth": np.arange(4, 15, 2),
    },
    {
        "regressor": [CatBoostRegressor()],
        "regressor__learning_rate": np.arange(0.05, 0.13, 0.01),
    },
]

'''поиск лучшей модели'''
grid = GridSearchCV(pipe, search_space, scoring="neg_root_mean_squared_error", cv=5, verbose=1)
best_model = grid.fit(X, y)

'''вывод лучшей модели на экран'''
best_model.best_estimator_.get_params()["regressor"]
print(grid.best_params_)
