🚢 Titanic Survival Prediction

This project uses machine learning techniques to predict whether a passenger on the Titanic would have survived, based on features like age, gender, passenger class, and embarkation port. The dataset is sourced from the Kaggle Titanic competition.

📌 Table of Contents

Project Overview

Problem Statement

Technologies Used

Dataset Description

Data Preprocessing

Modeling

Evaluation

Results

How to Run

Project Structure

Future Work

License




📖 Project Overview

This project aims to build a predictive model to determine which passengers survived the Titanic disaster. The solution showcases key machine learning steps such as data cleaning, feature engineering, model training, and evaluation.



🧠 Problem Statement

Given a dataset of passengers aboard the Titanic, predict whether a passenger survived (binary classification problem). The dataset contains both numerical and categorical features, and some missing values that must be handled during preprocessing.



🧰 Technologies Used

Python 3.10+

Pandas

NumPy

Scikit-learn

Matplotlib / Seaborn (for visualization)

Jupyter Notebook




📊 Dataset Description

The dataset includes the following features:

PassengerId – ID of the passenger

Pclass – Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)

Name, Sex, Age – Demographics

SibSp, Parch – Family aboard

Ticket, Fare – Ticket info

Cabin – Cabin number (many missing values)

Embarked – Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

Survived – Target variable (0 = No, 1 = Yes)




🧹 Data Preprocessing

Handling Missing Values:

'Age': Imputed using median values.

'Embarked': Filled with the most frequent category.

'Cabin': Dropped due to high proportion of missing data.


Feature Engineering:

Converted categorical features like 'Sex' and 'Embarked' using label encoding and one-hot encoding.

Created new features like "FamilySize" from 'SibSp' and 'Parch'.


Feature Selection:

Dropped irrelevant columns like 'Name', 'Ticket', and 'Cabin'.





🤖 Modeling

Two classification models were trained and evaluated:

1. Logistic Regression


2. Random Forest Classifier



Hyperparameter tuning was done using grid search and cross-validation to improve model performance.



📈 Evaluation

Evaluation metric: F1-Score

Random Forest outperformed Logistic Regression.

Achieved an F1-score of 0.78, indicating a balanced trade-off between precision and recall.




✅ Results

Successfully predicted survival of passengers using ML models.

Demonstrated the effectiveness of data preprocessing and ensemble methods.

Built a pipeline that can be extended for similar classification problems.




▶ How to Run

1. Clone this repository:

git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction


2. Install the required packages:

pip install -r requirements.txt


3. Run the Jupyter Notebook:

jupyter notebook Titanic_Survival_Prediction.ipynb





📁 Project Structure

titanic-survival-prediction/
│
├── data/
│   └── train.csv
├── notebooks/
│   └── Titanic_Survival_Prediction.ipynb
├── models/
│   └── random_forest_model.pkl
├── README.md
├── requirements.txt
└── .gitignore






