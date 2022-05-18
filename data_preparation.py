import pandas as pd
import numpy as np


data = pd.read_csv("E:\file", sep="\t", encoding="cp1251")
a = np.array(data["a"])
a1 = np.array(data["a"])
b = np.array(data["b"])
b1 = np.array(data["b"])
c = np.array(data["c"])
c1 = np.array(data["c"])


def proizv(mass, mass1):
    for i in range(len(mass)):
        if i < len(mass) - 1:
            mass1[i] = (mass[i + 1] - mass[i]) / 0.01
        else:
            mass1[i] = mass1[i - 1]
    return mass1


data.insert(8, "a1", proizv(a, a1))
data.insert(10, "b1", proizv(b, b1))
data.insert(12, "c1", proizv(c, c1))


# data[['a1','b1','c1']]=data[[''a1','b1','c1']].rolling(30,3,center=True).mean()
a2 = np.array(data["a1"])
b2 = np.array(data["b1"])
c2 = np.array(data["c1"])

data.insert(9, "a2", proizv(a1, a2))
data.insert(11, "b2", proizv(b1, b2))
data.insert(13, "c2", proizv(c1, c2))

# data[['a2', 'b2', 'c2']]=data[['a2', 'b2', 'c2']].rolling(30,3,center=True).mean()

data.to_csv("E:\file_prepared", sep="\t", encoding="cp1251", index=False)
