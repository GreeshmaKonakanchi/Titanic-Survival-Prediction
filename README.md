🚢 Titanic Survival Prediction using Machine Learning

A machine learning project that predicts the survival of passengers aboard the Titanic using classification models. This project walks through data preprocessing, feature engineering, model training, evaluation, and performance improvement — ideal for beginners and enthusiasts looking to understand end-to-end ML pipelines.



📖 Project Overview

The goal of this project is to build a predictive model that determines whether a Titanic passenger would have survived, using passenger characteristics such as age, gender, ticket class, and embarkation port. The dataset used is from the famous Kaggle Titanic competition.

This project demonstrates key steps in supervised learning, including handling missing data, feature transformation, applying classification models, and evaluating their performance.



🚀 Key Features

🧹 Preprocessing: Imputation of missing data in 'Age' and 'Embarked' columns.

🏷 Feature Engineering: Encoding categorical features such as 'Sex' and 'Embarked'.

⚙ Modeling: Implementation of Logistic Regression and Random Forest Classifier.

📊 Evaluation: Comparison based on F1-score, accuracy, precision, and recall.

📈 Performance: Achieved an F1-score of 0.78 using Random Forest.




🛠 Technology Stack

Category	Tools/Libraries

Programming Language	Python 3.10+
Data Manipulation	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Machine Learning	Scikit-learn (Logistic Regression, Random Forest)
Development	Jupyter Notebook / VSCode
Version Control	Git, GitHub




📚 Learning Analytics

This project helps you learn:

🔍 How to analyze and clean real-world data

🧠 Feature engineering and selection for classification

🔢 Difference between linear and ensemble classifiers

🧪 Model evaluation metrics (accuracy, precision, recall, F1)

📦 Structuring a machine learning project in Python




🤖 Modeling

Two classification models were implemented:

1. Logistic Regression

Suitable for baseline binary classification.

Performs well on linearly separable data.



2. Random Forest Classifier

An ensemble model combining multiple decision trees.

Handles non-linearity and feature interactions better.




Both models were trained using the training set, with 5-fold cross-validation applied to evaluate robustness.



📈 Evaluation

Models were evaluated using:

Accuracy: Percentage of correct predictions.

Precision: TP / (TP + FP) — focus on correctness of positive predictions.

Recall: TP / (TP + FN) — focus on completeness of positives.

F1-Score: Harmonic mean of precision and recall — balances both.


Model	Accuracy	Precision	Recall	F1-Score

Logistic Regression	0.78	0.76	0.74	0.75
Random Forest	0.81	0.80	0.76	0.78




🏁 Result

The Random Forest Classifier performed the best, achieving an F1-score of 0.78, indicating a strong balance between precision and recall.

Logistic Regression served as a simple yet effective baseline.




⚙ How to Implement

1. Clone the Repository

git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction

2. Install Required Packages

pip install -r requirements.txt

3. Launch Jupyter Notebook

jupyter notebook notebooks/Titanic_Survival_Prediction.ipynb



💻 Platform Compatibility

Platform	Compatible

Windows	✅
macOS	✅
Linux	✅
Google Colab	✅
Jupyter Lab	✅


Requires Python 3.8 or higher.



🤝 Contribution

We welcome ideas, enhancements, and contributions. Reach out via issues or submit pull requests to join us in transforming the learning experience.






