import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

PCOS_DATA_FILENAME = 'PCOS_data_without_infertility.csv'
PREDICTION_COLUMN = 'PCOS (Y/N)'
TEST_SIZE = 0.05
UNUSED_COLUMNS = [
    'Unnamed: 44',
    'Sl. No',
    'Patient File No.',
    'BMI',
    'FSH/LH',
    'Waist:Hip Ratio'
]

def load_pcos_data():
    df = pd.DataFrame(pd.read_csv(PCOS_DATA_FILENAME))
    df = df.drop(columns=UNUSED_COLUMNS)
    df = df.drop(305) #invalid data
    df = df.dropna()

    return df


def main():
    pcos_data = load_pcos_data()

    X = np.array(pcos_data.drop(columns=PREDICTION_COLUMN))
    y = np.array(pcos_data[PREDICTION_COLUMN])

    # splitting up 5% of the data into testing data and 95%  into training data
    (X_train, X_test, y_train, y_test) = sklearn.model_selection.train_test_split(
        X, y, test_size=TEST_SIZE
    )

    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)

    predictions = logistic.predict(X_test)
    print(classification_report(
        y_test,
        predictions,
        target_names=['Negative', 'Positive']
    ))


if __name__ == '__main__':
    main()
