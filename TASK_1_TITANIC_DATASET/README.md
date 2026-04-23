# 🚢 Titanic Survival Prediction

A machine learning project to predict whether a passenger survived
the Titanic disaster based on features like gender, ticket class,
age, fare, and family size.

## 📌 Project Overview
This is Task 1 of my Data Science internship at CodSoft.
The goal is to train and compare multiple classification models and
identify the best performer using proper evaluation metrics.

## 📂 Dataset
- 891 passengers, 12 features
- Target: Survived (0 = Did Not Survive, 1 = Survived)
- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic)

## 🔍 Approach
1. Exploratory Data Analysis (survival by gender, class, age, fare)
2. Feature Engineering — Family_Size and IsAlone columns
3. Preprocessing — missing value handling, encoding, scaling
4. Trained 4 models: Logistic Regression, KNN, SVM, Random Forest
5. Evaluated best model using confusion matrix and classification report

## 📊 Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 81.01% ✅ |
| KNN | 81.01% |
| SVM | 81.01% |
| Random Forest | 80.45% |

## 🔑 Key Findings
- Sex is the strongest predictor — females survived at 74% vs 19% for males
- First Class passengers had significantly better survival odds (~63%)
- Solo travellers survived at only 30.4% vs 50.6% for those with family
- Logistic Regression selected as final model for accuracy and interpretability
- All models beat the 61.6% naive baseline, confirming genuine learning

## 🛠️ Tech Stack
- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## 📁 File Structure
Task1_Titanic_Survival/
├── titanic_survival.ipynb
├── Titanic-Dataset.xls
├── requirements.txt
└── README.md

## 🚀 How to Run
```bash
pip install -r requirements.txt
jupyter notebook titanic_survival.ipynb
```

## 👤 Author
**Devesh** — BCA Student, Mulund College of Commerce
CodSoft Data Science Internship