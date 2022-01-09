import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

df = pd.DataFrame(pd.read_csv('PCOS_data_without_infertility.csv'))
df = df.drop(columns='Unnamed: 44')
df = df.dropna()



print(df)
