# PCOS Prediction 🥼
Predicts the likelihood of Polycystic Ovary Syndrome based on patient attributes and symptoms using Logistic Regression.  

![image](https://user-images.githubusercontent.com/78241340/148820557-059fb761-d8ca-48e1-8426-c1dc287d2413.png)

<img src="https://user-images.githubusercontent.com/78241340/148718462-7a01bc16-4c2c-4f4c-ac99-b5c71d96bc5b.png" width="250">

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

