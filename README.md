# 🌸 Iris Flower Classification

A machine learning project to classify Iris flowers into three species —
Setosa, Versicolor, and Virginica — based on sepal and petal measurements.

## 📌 Project Overview
This is Task 3 of my Data Science internship at CodSoft.
The goal is to train and compare multiple classification models and identify
the best performer using proper evaluation metrics.

## 📂 Dataset
- 150 samples, 3 species (50 each)
- Features: sepal_length, sepal_width, petal_length, petal_width
- Source: [Kaggle Iris Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)

## 🔍 Approach
1. Exploratory Data Analysis (pairplot, boxplots, correlation heatmap)
2. Preprocessing (label encoding, train-test split, standard scaling)
3. Trained 4 models: Logistic Regression, KNN, SVM, Random Forest
4. Evaluated best model using confusion matrix, classification report,
   and permutation feature importance

## 📊 Results

| Model               | Accuracy |
|---------------------|----------|
| SVM (RBF kernel)    | 96.67% ✅ |
| Logistic Regression | 93.33%   |
| KNN                 | 93.33%   |
| Random Forest       | 90.00%   |

## 🔑 Key Findings
- Petal features (length and width) are the strongest predictors
- Setosa is perfectly separable from the other two species
- The only misclassification was 1 Versicolor predicted as Virginica,
  consistent with the slight overlap identified during EDA
- SVM with RBF kernel handles the non-linear boundary best

## 🛠️ Tech Stack
- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## 📁 File Structure

## 🚀 How to Run
```bash
pip install -r requirements.txt
jupyter notebook iris_classification.ipynb
```

## 👤 Author
**Devesh** — BCA Student, Mulund College of Commerce  
CodSoft Data Science Internship