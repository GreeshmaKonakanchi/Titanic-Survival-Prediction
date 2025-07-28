🚢 Titanic Survival Prediction

A data science project that predicts passenger survival on the Titanic using machine learning. This project involves end-to-end steps of data preprocessing, feature engineering, model training, and evaluation using Logistic Regression and Random Forest classifiers.


---

📌 Project Overview

The goal is to build a predictive model that determines whether a passenger survived the Titanic disaster based on attributes like age, gender, passenger class, and more. This project uses the popular Titanic dataset from Kaggle.


---

🌟 Key Features

Handles missing values using imputation techniques

Encodes categorical variables for model compatibility

Applies two different classifiers: Logistic Regression and Random Forest

Evaluates model performance using accuracy and F1-score

Clean and modular code following best practices



---

🔧 Technology Stack

Programming Language: Python

Libraries Used:

pandas – data manipulation

numpy – numerical computations

scikit-learn – machine learning models & preprocessing

matplotlib, seaborn – visualization (optional)




---

📊 Learning Analytics

Through this project, the following key concepts were practiced:

Data cleaning and preprocessing (handling missing data)

Label Encoding and Feature Scaling

Supervised Learning Algorithms

Model evaluation techniques (confusion matrix, accuracy, F1-score)



---

👨‍🎓 Target Audience

Beginners in Machine Learning and Data Science

Students preparing for ML interviews or Kaggle competitions

Anyone interested in understanding ML pipelines



---

🧠 Modelling

1. Data Preprocessing

Missing Age values filled using median imputation

Categorical variables (Sex, Embarked) encoded using LabelEncoder

Unnecessary columns like Name, Cabin, Ticket dropped


2. Model Training

Logistic Regression: Simple linear model for binary classification

Random Forest: Ensemble model for better generalization



---

🧪 Evaluation

Model	Accuracy	F1 Score

Logistic Regression	~78%	~0.78
Random Forest	~82%	~0.80


Evaluation metrics used:

Confusion Matrix

Precision, Recall, F1-Score




---

⚙️ How to Implement

1. Clone the repository



git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction

2. Install dependencies



pip install -r requirements.txt

3. Run the notebook or script



jupyter notebook Titanic_Prediction.ipynb


---

💻 Platform Compatibility

OS: Windows / Linux / macOS

Python ≥ 3.7

Jupyter Notebook / VSCode / Any IDE



---

🤝 Contribution

Contributions are welcome! If you'd like to:

Improve model performance

Add visualizations

Create a Flask/Streamlit web app for prediction


Feel free to open an issue or submit a pull request.


---

📂 Directory Structure (Suggested)

titanic-survival-prediction/
├── data/                   # Dataset files
├── notebooks/              # Jupyter notebooks
├── src/                    # Scripts (preprocessing, models)
├── requirements.txt
└── README.md


---
