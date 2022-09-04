import pandas as pd
import xlsxwriter
import numpy as np
from sklearn import linear_model
from openpyxl import load_workbook
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
import pydotplus
import graphviz
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns

def print_confusion_matrix(name:str, y_true: list, y_pred: list):
    print(f"Confusion Matrix:")
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=y_true.unique()).plot()
    plt.savefig(f"{name}.png")
    print(f"Accuracy: {round(accuracy_score(y_true, y_pred), 2)*100}%")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))



def save_tree(name: str, classifier: tree, features: list, labels: list) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=500)
    tree.plot_tree(classifier, feature_names=features,class_names=labels, filled=True)
    fig.savefig(f"{name}.png")

# setting the filepath where the file should be stored
filepath = r"Z:\CS3481\Assignment 1\assignment1.xlsx"
spl = filepath.split('\\')

# importing the training data into a dataframe
df2 = pd.read_csv("testing.csv")
df = pd.read_csv("training.csv")

# setting the training variables and the variable which they should output
x_train = df2[['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'pred_minus_obs_H_b1', 'pred_minus_obs_H_b2',
               'pred_minus_obs_H_b3', 'pred_minus_obs_H_b4', 'pred_minus_obs_H_b5', 'pred_minus_obs_H_b6',
               'pred_minus_obs_H_b7', 'pred_minus_obs_H_b8', 'pred_minus_obs_H_b9', 'pred_minus_obs_S_b1',
               'pred_minus_obs_S_b2', 'pred_minus_obs_S_b3', 'pred_minus_obs_S_b4', 'pred_minus_obs_S_b5',
               'pred_minus_obs_S_b6', 'pred_minus_obs_S_b7', 'pred_minus_obs_S_b8', 'pred_minus_obs_S_b9']]
y_train = df2['class']

fn = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'pred_minus_obs_H_b1', 'pred_minus_obs_H_b2',
      'pred_minus_obs_H_b3', 'pred_minus_obs_H_b4', 'pred_minus_obs_H_b5', 'pred_minus_obs_H_b6', 'pred_minus_obs_H_b7',
      'pred_minus_obs_H_b8', 'pred_minus_obs_H_b9', 'pred_minus_obs_S_b1', 'pred_minus_obs_S_b2', 'pred_minus_obs_S_b3',
      'pred_minus_obs_S_b4', 'pred_minus_obs_S_b5', 'pred_minus_obs_S_b6', 'pred_minus_obs_S_b7', 'pred_minus_obs_S_b8',
      'pred_minus_obs_S_b9']
cn = ['class']

# calling the regression model
classif = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=5)
# fitting the regression model based on the training dataset
classif.fit(x_train.values, y_train)

x_test = df[['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'pred_minus_obs_H_b1', 'pred_minus_obs_H_b2',
             'pred_minus_obs_H_b3', 'pred_minus_obs_H_b4', 'pred_minus_obs_H_b5', 'pred_minus_obs_H_b6',
             'pred_minus_obs_H_b7', 'pred_minus_obs_H_b8', 'pred_minus_obs_H_b9', 'pred_minus_obs_S_b1',
             'pred_minus_obs_S_b2', 'pred_minus_obs_S_b3', 'pred_minus_obs_S_b4', 'pred_minus_obs_S_b5',
             'pred_minus_obs_S_b6', 'pred_minus_obs_S_b7', 'pred_minus_obs_S_b8', 'pred_minus_obs_S_b9']]
y_test = df['class']

# printing the regression coefficients, scores - only for testing purposes to make sure that the code is running
print(classif.score(x_train, y_train))

y_pred = classif.predict(x_test)

print_confusion_matrix("random", y_test, y_pred)
save_tree("random", classif, x_train.columns, ['d','h','s','o'])