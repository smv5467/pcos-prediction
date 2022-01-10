import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

df = pd.DataFrame(pd.read_csv('PCOS_data_without_infertility.csv'))
df = df.drop(columns=[
'Unnamed: 44',
'Sl. No',
'Patient File No.',
'BMI',
'FSH/LH',
'Waist:Hip Ratio'
])
df = df.drop(305) #invalid data
df = df.dropna()

predict = 'PCOS (Y/N)'

X = np.array(df.drop(columns=predict))
y = np.array(df[predict])

# splitting up 10% of the data into testing data and 90%  into training data
(X_train, X_test, y_train, y_test) = sklearn.model_selection.train_test_split(
    X,
    y,
    test_size=0.1
)

linear = linear_model.LinearRegression()
linear.fit(X_train, y_train)
accuracy = linear.score(X_test, y_test)

predictions = np.around(np.maximum(linear.predict(X_test), 0))
for x in range(len(predictions)):
    print(predictions[x], X_test[x], y_test[x])


print(accuracy)

#print(df)
