import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

student_cols = ["school", "age", "sex", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"]

student_df = pd.read_csv("student-mat.csv", sep=";")

print(student_df.head())

#Encode categorials variables
label_encoders = {}
for column in student_df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    student_df[column] = le.fit_transform(student_df[column])
    label_encoders[column] = le

student_df['failing'] = (student_df['G3'] < 10).astype(int)
student_df = student_df.drop(columns=['G3'])

X = student_df.drop(columns=['failing'])
y = student_df['failing']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)


X_first_five = scaler.transform(X.head(5))
y_pred_first_five_knn = knn_model.predict(X_first_five)

print("K-Nearest Neighbors predictions for the first five records:")
print(y_pred_first_five_knn)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred_first_five_knn = nb_model.predict(X_first_five)

# True labels for the first five records
y_true_first_five = y.head(5).values
print("True labels for the first five records:")
print(y_true_first_five)

y_pred_first_five_nb = nb_model.predict(X_first_five)

print("Gaussian Naive Bayes predictions for the first five records:")
print(y_pred_first_five_knn)

print("Classifications report for K-Nearest Neigbors on the first five records:")
print(classification_report(y_true_first_five, y_pred_first_five_knn, target_names=["Not Failing", "Failing"]))

print("Classifications report for Gaussian Naive Bayes on the first five records:")
print(classification_report(y_true_first_five, y_pred_first_five_nb, target_names=["Not Failing", "Failing"]))

y_pred_knn = knn_model.predict(X_test)
print("K-Nearest Neigbors Classifier:")
print(classification_report(y_test, y_pred_knn))

