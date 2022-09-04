import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB



def print_confusion_matrix(name: str, y_true: list, y_pred: list):
    print(f"Confusion Matrix:")
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=y_true.unique()).plot()
    plt.savefig(f"{name} Matrix.png")
    print(f"Accuracy: {round(accuracy_score(y_true, y_pred), 3) * 100}%")
    # print("Classification Report:")
    # print(classification_report(y_true, y_pred))


def save_tree(name: str, classifier: GaussianNB, features: list, labels: list) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=500)
    tree.plot_tree(classifier.fit(x_train.values, y_train), class_names=labels, filled=True)
    # tree.plot_tree(classifier, feature_names=features, class_names=labels, filled=True)
    fig.savefig(f"{name} Tree.png")


filepath = r"Z:\CS3481\Assignment 1\assignment1.xlsx"
spl = filepath.split('\\')

df = pd.read_csv("testing.csv")
df2 = pd.read_csv("training.csv")

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

x_test = df[['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'pred_minus_obs_H_b1', 'pred_minus_obs_H_b2',
             'pred_minus_obs_H_b3', 'pred_minus_obs_H_b4', 'pred_minus_obs_H_b5', 'pred_minus_obs_H_b6',
             'pred_minus_obs_H_b7', 'pred_minus_obs_H_b8', 'pred_minus_obs_H_b9', 'pred_minus_obs_S_b1',
             'pred_minus_obs_S_b2', 'pred_minus_obs_S_b3', 'pred_minus_obs_S_b4', 'pred_minus_obs_S_b5',
             'pred_minus_obs_S_b6', 'pred_minus_obs_S_b7', 'pred_minus_obs_S_b8', 'pred_minus_obs_S_b9']]
y_test = df['class']
val = 6
# classif = RandomForestClassifier(n_estimators=val, criterion="gini", max_depth=5, min_samples_leaf=5)
classif = GaussianNB()
classif.fit(x_train.values, y_train)

print(classif.score(x_train, y_train))

y_pred = classif.predict(x_test)

# for i in range(0, val):
#     classif.estimators_[i].fit(x_train.values, y_train)
#     y_pred_tree = classif.estimators_[i].predict(x_test)
#     # print(classif.estimators_[i].score(x_train, y_train))
#     print(f"Accuracy for subtree {i+1}: {round(accuracy_score(y_test, y_pred_tree), 3) * 100}%")
#     print_confusion_matrix(f"submatrix {i+1}", y_test, y_pred_tree)
#     save_tree(f"Subtree {i+1}", classif.estimators_[i], x_train.columns, ['d', 'h', 's', 'o'])

# for i in range(0, 1):
#     data = {'feature names': fn, 'feature importance': classif.feature_importances_}
#     df = pd.DataFrame(data)
#
#     df.sort_values(by=['feature importance'], ascending=False, inplace=True)
#
#     plt.figure(figsize=(16, 9))
#
#     sns.barplot(x=df['feature importance'], y=df['feature names'])
#
#     title = "Feature Importance for Random Forest Model"
#     plt.title(title)
#     plt.xlabel("Feature Importance")
#     plt.ylabel("Feature Name")
#     plt.show()




print_confusion_matrix("GNB", y_test, y_pred)
# save_tree("GNB", classif, x_train.columns, ['d', 'h', 's', 'o'])
