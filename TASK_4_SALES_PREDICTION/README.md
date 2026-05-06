# 📈 Sales Prediction Using Python

A machine learning project to predict product sales based on advertising
spend across TV, Radio, and Newspaper channels using regression models.

## 📌 Project Overview
This is Task 4 of my Data Science internship at CodSoft.
The goal is to identify which advertising channel drives sales the most
and build a model that accurately forecasts sales from budget allocation.

## 📂 Dataset
- 200 samples, 3 advertising features + 1 target
- Features: TV, Radio, Newspaper (advertising budgets in $k)
- Target: Sales (units sold)
- Source: [Kaggle Advertising Dataset](https://www.kaggle.com/datasets/bumba5341/advertisingcsv)

## 🔍 Approach
1. Exploratory Data Analysis (distributions, scatter plots, correlation heatmap)
2. Preprocessing — train-test split, standard scaling
3. Trained 4 models: Linear Regression, Ridge, Lasso, Random Forest
4. Evaluated using MAE, RMSE and R² score
5. Feature importance analysis
6. Live sales prediction

## 📊 Results

| Model | R² Score |
|---|---|
| Random Forest | 0.9535 ✅ |
| Linear Regression | 0.9059 |
| Ridge Regression | 0.9057 |
| Lasso Regression | 0.9052 |

## 🔑 Key Findings
- TV advertising is the dominant channel — drives 84.5% of sales variance
- Radio is a meaningful secondary channel at 13.7% importance
- Newspaper spend has almost no impact at just 1.8% importance
- Random Forest outperforms linear models by ~4.5% R² — confirming
  non-linear relationships exist between ad spend and sales
- Average prediction error of only 0.92 sales units across test set

## 🛠️ Tech Stack
- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## 📁 File Structure
TASK_4_SALES_PREDICTION/
├── sales_prediction.ipynb
├── advertising.xls
├── requirements.txt
└── README.md

## 🚀 How to Run
```bash
pip install -r requirements.txt
jupyter notebook sales_prediction.ipynb
```

## 👤 Author
**Devesh** — BCA Student, Mulund College of Commerce
CodSoft Data Science Internship