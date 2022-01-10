import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

PCOS_DATA = 'PCOS_data_without_infertility.csv'

def load_pcos_data():
    df = pd.DataFrame(pd.read_csv(PCOS_DATA))
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

    return df

def main():
    pcos_data = load_pcos_data()

    prediction_column = 'PCOS (Y/N)'

    X = np.array(pcos_data.drop(columns=prediction_column))
    y = np.array(pcos_data[prediction_column])

    # splitting up 5% of the data into testing data and 95%  into training data
    (X_train, X_test, y_train, y_test) = sklearn.model_selection.train_test_split(
        X, y, test_size=0.05
    )

    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    accuracy = logistic.score(X_test, y_test)

    predictions = logistic.predict(X_test)
    print('Prediction  Answer')
    for i, predictions in enumerate(predictions):
        print(f'{predictions}         {y_test[i]}')

    print('Accuracy: ', accuracy)

if __name__ == '__main__':
    main()
