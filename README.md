# PCOS Prediction 🥼
Predicts the likelihood of Polycystic Ovary Syndrome based on patient attributes and symptoms using Logistic Regression.  

<img src="https://user-images.githubusercontent.com/78241340/148857968-0782ac4d-02c8-4a3d-a162-6ba22dcb1e64.png" height="200">

# Setup
### Clone the Repository 
`git clone https://github.com/smv5467/pcos-prediction`

### Add Dependencies with Poetry 
#### If you don't have poetry install with:
`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`  

`poetry install `  

### Download Data
Retrieve data from Kaggle: https://www.kaggle.com/prasoonkottarathil/polycystic-ovary-syndrome-pcos  
Download PCOS_data_without_infertility.xlsx  
Open excel file and save as a CSV file under the same name

### Run program
`poetry run python pcos_predictor.py`

