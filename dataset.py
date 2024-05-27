import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
data = pd.read_csv("magic04.data", names=cols)

print(data.head())

data["class"] = (data["class"] == "g").astype(int)

for label in cols[:-1]:
    plt.hist(data[data["class"] == 1][label], color="blue", label="gamma", alpha=0.7, density=True)
    plt.hist(data[data["class"] == 0][label], color="red", label="hadron", alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()


train, valid, test = np.split(data.sample(frac=1), [int(0.6*len(data)), int(0.8*len(df))])

def scale_dataset(dataframe):
    x = dataframe[dataframe.cols[:-1]].values
    y = dataframe[dataframe.cols[-1]].values

print(data.head())



