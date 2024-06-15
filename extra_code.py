import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load breast cancer dataset
from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()

# Display dataset description and feature names
print(cancer_dataset['DESCR'])
print("Feature names:", cancer_dataset['feature_names'])

# Create DataFrame
cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'], cancer_dataset['target']],
                         columns=np.append(cancer_dataset['feature_names'], 'target'))

# Save DataFrame to CSV
cancer_df.to_csv('Cancer_df.csv', index=False)

# Display DataFrame information
print("Head of cancer DataFrame:")
print(cancer_df.head(6))

print("Information of cancer DataFrame:")
print(cancer_df.info())

# Display numerical distribution of data
print("Numerical distribution of data:")
print(cancer_df.describe())

# Pairplot of cancer DataFrame
sns.pairplot(cancer_df, hue='target')
plt.savefig('pairplot1.png')
plt.show()

sns.pairplot(cancer_df, hue='target', vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])
plt.savefig('pairplot2.png')
plt.show()

# Count plot of target class
sns.countplot(x='target', data=cancer_df)
plt.savefig('countplot_target.png')
plt.show()

# Count plot of 'mean radius' feature
plt.figure(figsize=(20, 8))
sns.countplot(x='mean radius', data=cancer_df)
plt.savefig('countplot_mean_radius.png')
plt.show()

# Heatmap of DataFrame
plt.figure(figsize=(16, 9))
sns.heatmap(cancer_df, cmap='viridis')
plt.savefig('heatmap_df.png')
plt.show()

# Heatmap of correlation matrix
plt.figure(figsize=(20, 20))
sns.heatmap(cancer_df.corr(), annot=True, cmap='coolwarm', linewidths=2)
plt.savefig('heatmap_corr.png')
plt.show()

# Create a second DataFrame by dropping the target column
cancer_df2 = cancer_df.drop('target', axis=1)
print("The shape of 'cancer_df2' is:", cancer_df2.shape)

# Visualize correlation barplot
plt.figure(figsize=(16, 5))
ax = sns.barplot(x=cancer_df2.corrwith(cancer_df['target']).index, y=cancer_df2.corrwith(cancer_df['target']))
ax.tick_params(labelrotation=90)
plt.savefig('correlation_barplot.png')
plt.show()

# Input and output variables
X = cancer_df.drop('target', axis=1)
y = cancer_df['target']

# Display head of input and output variables
print("Input variables (X):")
print(X.head(6))

print("Output variable (y):")
print(y.head(6))

# Split dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Function to evaluate and display accuracy of a model
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {model_name} = {accuracy}')
    print(f'Confusion matrix of {model_name}:\n {confusion_matrix(y_test, y_pred)}\n')
    return model

# Support Vector Classifier
from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier = evaluate_model(svc_classifier, X_train, y_train, X_test, y_test, "SVC")

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state=51, penalty='l2', max_iter=200)
lr_classifier = evaluate_model(lr_classifier, X_train, y_train, X_test, y_test, "Logistic Regression")

# K-Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_classifier = evaluate_model(knn_classifier, X_train, y_train, X_test, y_test, "KNN")

# Train with Standard scaled Data
knn_classifier2 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_classifier2 = evaluate_model(knn_classifier2, X_train_sc, y_train, X_test_sc, y_test, "Scaled KNN")

# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier = evaluate_model(nb_classifier, X_train, y_train, X_test, y_test, "Naive Bayes")

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=51)
dt_classifier = evaluate_model(dt_classifier, X_train, y_train, X_test, y_test, "Decision Tree")

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=51)
rf_classifier = evaluate_model(rf_classifier, X_train, y_train, X_test, y_test, "Random Forest")

# AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
adb_classifier = AdaBoostClassifier(estimator=DecisionTreeClassifier(criterion='entropy', random_state=200),
                                    n_estimators=2000, learning_rate=0.1, algorithm='SAMME.R', random_state=1)
adb_classifier = evaluate_model(adb_classifier, X_train, y_train, X_test, y_test, "AdaBoost")

# XGBoost Classifier
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier = evaluate_model(xgb_classifier, X_train, y_train, X_test, y_test, "XGBoost")

# XGBoost parameter tuning
params = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
}

# Randomized Search
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=xgb_classifier, param_distributions=params, scoring='roc_auc', n_jobs=-1, verbose=3)
random_search.fit(X_train, y_train)

# Cross validation
cross_validation = cross_val_score(estimator=random_search.best_estimator_, X=X_train_sc, y=y_train, cv=10)
print("Cross validation accuracy of XGBoost model:", cross_validation)
print("Mean cross validation accuracy of XGBoost model:", cross_validation.mean())

# Save model using pickle
import pickle
pickle.dump(random_search.best_estimator_, open('breast_cancer_detector.pickle', 'wb'))

# Load model and make predictions
breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))
y_pred = breast_cancer_detector_model.predict(X_test)

# Confusion matrix and accuracy
print('Confusion matrix of XGBoost model:\n', confusion_matrix(y_test, y_pred), '\n')
print('Accuracy of XGBoost model =', accuracy_score(y_test, y_pred))
