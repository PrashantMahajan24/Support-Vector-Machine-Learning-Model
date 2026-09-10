# SVM Algorithm – Diabetes Prediction

## Project Overview

This project uses the **Support Vector Machine (SVM)** algorithm to predict whether a person is likely to have diabetes based on medical information.

The dataset contains different health-related features, and the `Outcome` column is used as the target variable.

* `0` → No Diabetes
* `1` → Diabetes

## Objective

The main objective of this project is to build a classification model using **SVM** and understand how feature scaling can improve machine learning model performance.

## Dataset

The project uses a diabetes dataset containing medical information about patients.

### Features

| Feature                  | Description                |
| ------------------------ | -------------------------- |
| Pregnancies              | Number of pregnancies      |
| Glucose                  | Glucose level              |
| BloodPressure            | Blood pressure             |
| SkinThickness            | Skin thickness             |
| Insulin                  | Insulin level              |
| BMI                      | Body Mass Index            |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age                      | Age of the person          |
| Outcome                  | Target variable            |

## Project Steps

1. Import required Python libraries.
2. Load the diabetes dataset.
3. Separate input features and target variable.
4. Split the data into training and testing sets.
5. Apply feature scaling.
6. Use StandardScaler for standardization.
7. Use MinMaxScaler for normalization.
8. Train the SVM classification model.
9. Make predictions on test data.
10. Evaluate the model using accuracy, confusion matrix, and classification report.

## Machine Learning Algorithm

### Support Vector Machine (SVM)

SVM is a supervised machine learning algorithm mainly used for **classification** problems.

It tries to find the best boundary, called a **hyperplane**, that separates different classes of data.

In this project, SVM is used to classify patients into:

* No Diabetes
* Diabetes

```python
from sklearn.svm import SVC

svm = SVC()

svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
```

## Feature Scaling

Feature scaling is performed before training the SVM model because different features can have different ranges.

The notebook uses:

### StandardScaler

StandardScaler transforms the data based on the mean and standard deviation.

```python
from sklearn.preprocessing import StandardScaler

scaler_std = StandardScaler()

X_train = scaler_std.fit_transform(X_train)
X_test = scaler_std.transform(X_test)
```

### MinMaxScaler

MinMaxScaler scales values into a specified range, normally between 0 and 1.

```python
from sklearn.preprocessing import MinMaxScaler

scaler_min = MinMaxScaler()

X_train = scaler_min.fit_transform(X_train)
X_test = scaler_min.transform(X_test)
```

## Model Evaluation

The SVM model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

matrix = confusion_matrix(y_test, y_pred_svm)
acc = accuracy_score(y_test, y_pred_svm)
report = classification_report(y_test, y_pred_svm)

print(matrix)
print(acc)
print(report)
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook


